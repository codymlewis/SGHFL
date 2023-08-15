import argparse
import json
import numpy as np
import sklearn.metrics as skm
from tqdm.auto import trange

import flagon
from flagon.common import Config, Parameters, Metrics, count_clients

import src

import os
os.makedirs("results", exist_ok=True)


def create_sh_clients(create_model_fn, nclients, client_ids, seed=None):
    get_customer_data = src.load_data.solar_home()

    def create_client(client_id: str):
        client_X, client_Y = get_customer_data(client_ids[client_id])
        client_data = {"train": {"X": client_X[:300 * 24], "Y": client_Y[:300 * 24]}, "test": {"X": client_X[300 * 24:], "Y": client_Y[300 * 24:]}}
        return src.client.Client(client_data, create_model_fn)
    return create_client


def create_clients(data, create_model_fn, network_arch, seed=None):
    rng = np.random.default_rng(seed)
    idx = iter(src.common.regional_distribution(data['train']['Y'], network_arch, rng))
    test_idx = iter(src.common.regional_test_distribution(data['test']['Y'], network_arch))
    nclients = count_clients(network_arch)
    data = data.normalise()

    def create_client(client_id: str):
        return src.client.Client(data.select({"train": next(idx), "test": next(test_idx)}), create_model_fn)
    return create_client


class CosineSimilarity(flagon.common.Metric):
    def __call__(self, global_parameters, client_parameters, client_samples):
        client_parameters = [np.concatenate([cl.reshape(-1) for cl in cp]) for cp in client_parameters]
        similarity_matrix = skm.pairwise.cosine_similarity(client_parameters) - np.eye(len(client_parameters))
        return similarity_matrix.sum() / (len(client_parameters) * (len(client_parameters) - 1))


def experiment(config, strategy_class, middle_server_class=flagon.MiddleServer):
    aggregate_results = []
    test_results = []
    if config['dataset'] == "fmnist":
        data = src.load_data.mnist()
        data = data.normalise()
    else:
        data_collector_counts, client_ids = src.load_data.solar_home_customer_regions()
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        if config['dataset'] == "fmnist":
            server = flagon.Server(src.common.create_fmnist_model().get_parameters(), config)
            network_arch = {
                "clients": [
                    {"clients": 3, "strategy": strategy_class(), "middle_server_class": middle_server_class} for _ in range(5)
                ]
            }
            clients = create_clients(data, src.common.create_fmnist_model, network_arch, seed=seed)
        else:
            server = flagon.Server(src.common.create_solar_home_model().get_parameters(), config)
            network_arch = {
                "clients": [{"clients": 0} for _ in data_collector_counts.keys()],
            }
            for k, v in data_collector_counts.items():
                network_arch['clients'][k]['clients'] = v
            clients = create_sh_clients(
                src.common.create_solar_home_model, flagon.common.count_clients(network_arch), client_ids, seed=seed
            )
        history = flagon.start_simulation(
            server,
            clients,
            network_arch
        )
        aggregate_results.append(history.aggregate_history[config['num_rounds']])
        test_results.append(history.test_history[config['num_rounds']])
        pbar.set_postfix(aggregate_results[-1])
    return {"train": aggregate_results, "test": test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating the fairness when clients drop out from colloboration.")
    parser.add_argument("-i", "--id", type=int, default=1, help="Which of the experiments in the config to perform (counts from 1).")
    parser.add_argument("-d", "--dataset", type=str, default="fmnist", help="Which of the datasets to perform the experiment with.")
    args = parser.parse_args()

    with open("configs/performance.json", 'r') as f:
        experiment_config = src.common.get_experiment_config(json.load(f), args.id)
    experiment_config["metrics"] = [CosineSimilarity()]
    experiment_config["dataset"] = args.dataset

    results = experiment(
        experiment_config,
        src.strategy.KickbackMomentum if experiment_config.get("mu1") else flagon.strategy.FedAVG,
        src.middle_server.IntermediateFineTuner if experiment_config.get("num_finetune_episodes") else flagon.MiddleServer
    )

    filename = "results/performance_{}{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['metrics', 'round', 'mu1', 'mu2']]),
        "_momentum" if experiment_config.get("mu1") else ""
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")