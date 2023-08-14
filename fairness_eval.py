import argparse
import json
import datasets
import numpy as np
from tqdm.auto import trange

import flagon
from flagon.strategy import FedAVG
from flagon.common import Config, Parameters, Metrics, count_clients

import load_data
import strategy
import client
import middle_server
import server
import common

import os
os.makedirs("results", exist_ok=True)


def create_clients(data, create_model_fn, network_arch, seed=None):
    rng = np.random.default_rng(seed)
    idx = iter(common.regional_distribution(data['train']['Y'], network_arch, rng))
    test_idx = iter(common.regional_test_distribution(data['test']['Y'], network_arch))
    nclients = count_clients(network_arch)
    data = data.normalise()

    def create_client(client_id: str):
        return client.Client(data.select({"train": next(idx), "test": next(test_idx)}), create_model_fn, seed)
    return create_client


def experiment(config):
    aggregate_results = []
    test_results = []
    data = load_data.mnist()
    data = data.normalise()
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        if config.get("adaptive_loss"):
            server_class = server.Adaptive
        else:
            server_class = flagon.Server
        experiment_server = server_class(
            common.create_fmnist_model().get_parameters(),
            config,
            client_manager=server.DroppingClientManager(config['drop_round'], seed=seed),
        )
        if config.get("num_finetuning_episodes") and config.get("adaptive_loss"):
            middle_server_class = middle_server.AdaptiveLossIntermediateFineTuner
        elif config.get("num_finetuning_episodes"):
            middle_server_class = middle_server.IntermediateFineTuner
        elif config.get("adaptive_loss"):
            middle_server_class = middle_server.AdaptiveLoss
        else:
            middle_server_class = flagon.MiddleServer
        if config.get("bottom_k"):
            strategy_class = strategy.BottomK
        elif config.get('mu1'):
            strategy_class = strategy.FreezingMomentum()
        else:
            strategy_class = flagon.server.FedAVG()
        network_arch = {
            "clients": [
                {
                    "clients": 3,
                    "strategy": strategy_class,
                    "middle_server_class": middle_server_class
                } for _ in range(5)
            ]
        }
        history = flagon.start_simulation(
            experiment_server,
            create_clients(data, common.create_fmnist_model, network_arch, seed=seed),
            network_arch
        )
        aggregate_results.append(history.aggregate_history[config['num_rounds']])
        test_results.append(history.test_history[config['num_rounds']])
        pbar.set_postfix(aggregate_results[-1])
    return {"train": aggregate_results, "test": test_results}


def fairness_analytics(client_metrics, client_samples, config):
    distributed_metrics = {k: [v] for k, v in client_metrics[0].items()}
    for cm in client_metrics[1:]:
        for k, v in cm.items():
            distributed_metrics[k].append(v)
    analytics = {f"{k} mean": np.mean(v).item() for k, v in distributed_metrics.items() if "std" in k}
    analytics.update({f"{k} std": np.std(v).item() for k, v in distributed_metrics.items()})
    return analytics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating the fairness when clients drop out from colloboration.")
    parser.add_argument("-i", "--id", type=int, default=1, help="Which of the experiments in the config to perform (counts from 1).")
    parser.add_argument("-d", "--dataset", type=str, default="fmnist", help="Which of the datasets to perform the experiment with.")
    args = parser.parse_args()

    with open("configs/fairness.json", 'r') as f:
        experiment_config = common.get_experiment_config(json.load(f), args.id)
    experiment_config["analytics"] = [fairness_analytics]
    experiment_config["dataset"] = args.dataset

    results = experiment(experiment_config)

    filename = "results/fairness_{}{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['analytics', 'round', 'mu1', 'adapt_loss']]),
        "_momentum" if experiment_config.get("mu1") else ""
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")