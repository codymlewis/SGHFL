import argparse
import json
import numpy as np
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
        return fl.client.Client(data.select({"train": next(idx), "test": next(test_idx)}), create_model_fn, seed)
    return create_client


def experiment(config):
    full_results = {}
    data = fl.load_data.mnist()
    data = data.normalise()
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32

        experiment_server = fl.server.Server(
            fl.common.create_fmnist_model().get_parameters(),
            config,
            client_manager=fl.server.DroppingClientManager(config['drop_round'], seed=seed),
            strategy_name=config.get("aggregator")
        )

        if config.get("num_finetune_episodes"):
            middle_server_class = fl.middle_server.IntermediateFineTuner
        else:
            middle_server_class = fl.middle_server.MiddleServer

        network_arch = {
            "clients": [
                {
                    "clients": 3,
                    "strategy": config.get("aggregator"),
                    "middle_server_class": middle_server_class
                } for _ in range(5)
            ]
        }
        history = fl.simulation.start_simulation(
            experiment_server,
            create_clients(data, fl.common.create_fmnist_model, network_arch, seed=seed),
            network_arch
        )
        pbar.set_postfix(history.aggregate_history[config['num_rounds']])

        results, dropped_results = history.test_history[config['num_rounds']]
        for k, v in dropped_results.items():
            results[f"dropped {k}"] = v
        full_results[i] = results
    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform experiments evaluating the fairness when clients drop out from colloboration."
    )
    parser.add_argument("-i", "--id", type=int, default=1,
                        help="Which of the experiments in the config to perform (counts from 1).")
    args = parser.parse_args()

    with open("configs/fairness.json", 'r') as f:
        experiment_config = fl.common.get_experiment_config(json.load(f), args.id)
    print(f"Using config: {experiment_config}")
    experiment_config["experiment"] = "fairness"

    results = experiment(experiment_config)

    filename = "results/fairness_{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['analytics', 'round', 'mu1', 'adapt_loss']]),
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")
