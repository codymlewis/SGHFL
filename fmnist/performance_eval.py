import argparse
import json
import numpy as np
import sklearn.metrics as skm
from tqdm.auto import trange

import fl

import os
os.makedirs("results", exist_ok=True)


def create_clients(data, create_model_fn, network_arch, seed=None):
    rng = np.random.default_rng(seed)
    idx = iter(fl.common.regional_distribution(data['train']['Y'], network_arch, rng))
    test_idx = iter(fl.common.regional_test_distribution(data['test']['Y'], network_arch))
    data = data.normalise()

    def create_client(client_id: str):
        return fl.client.Client(data.select({"train": next(idx), "test": next(test_idx)}), create_model_fn)
    return create_client


class CosineSimilarity(fl.common.Metric):
    def __call__(self, global_parameters, client_parameters, client_samples):
        client_grads = [
            np.concatenate([gl.reshape(-1) - cl.reshape(-1) for cl, gl in zip(cp, global_parameters)])
            for cp in client_parameters
        ]
        similarity_matrix = np.abs(skm.pairwise.cosine_similarity(client_grads)) - np.eye(len(client_grads))
        return similarity_matrix.sum() / (len(client_grads) * (len(client_grads) - 1))


def experiment(config, strategy_name, middle_server_class=fl.middle_server.MiddleServer):
    aggregate_results = []
    test_results = []
    data = fl.load_data.mnist()
    data = data.normalise()
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        server = fl.server.Server(fl.common.create_fmnist_model().get_parameters(), config)
        network_arch = {
            "clients": [
                {"clients": 3, "strategy": strategy_name, "middle_server_class": middle_server_class}
                for _ in range(5)
            ]
        }
        clients = create_clients(data, fl.common.create_fmnist_model, network_arch, seed=seed)
        history = fl.simulation.start_simulation(server, clients, network_arch)
        aggregate_results.append(history.aggregate_history[config['num_rounds']])
        test_results.append(history.test_history[config['num_rounds']])
        pbar.set_postfix(aggregate_results[-1])
    return {"train": aggregate_results, "test": test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Perform experiments evaluating the fairness when clients drop out from colloboration.")
    parser.add_argument("-i", "--id", type=int, default=1,
                        help="Which of the experiments in the config to perform (counts from 1).")
    args = parser.parse_args()

    with open("configs/performance.json", 'r') as f:
        experiment_config = fl.common.get_experiment_config(json.load(f), args.id)
    print(f"Using config: {experiment_config}")
    experiment_config["metrics"] = [CosineSimilarity()]

    results = experiment(
        experiment_config,
        experiment_config.get("aggregator"),
        fl.middle_server.IntermediateFineTuner if experiment_config.get("num_finetune_episodes") else fl.middle_server.MiddleServer
    )

    filename = "results/performance_{}{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['metrics', 'round', 'mu1']]),
        "_momentum" if experiment_config.get("mu1") else ""
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")
