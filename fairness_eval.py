import argparse
import json
import datasets
import numpy as np
from tqdm.auto import trange

import flagon
from flagon.strategy import FedAVG
from flagon.common import Config, Parameters, Metrics, count_clients

import src

import os
os.makedirs("results", exist_ok=True)


def create_clients(data, create_model_fn, network_arch, seed=None):
    rng = np.random.default_rng(seed)
    idx = iter(src.common.regional_distribution(data['train']['Y'], network_arch, rng))
    test_idx = iter(src.common.regional_test_distribution(data['test']['Y'], network_arch))
    nclients = count_clients(network_arch)
    data = data.normalise()

    def create_client(client_id: str):
        return src.client.Client(data.select({"train": next(idx), "test": next(test_idx)}), create_model_fn, seed)
    return create_client


def experiment(config):
    aggregate_results = []
    test_results = []
    data = src.load_data.mnist()
    data = data.normalise()
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        if config.get("adaptive_loss"):
            server_class = src.server.Adaptive
        else:
            server_class = flagon.Server

        if config.get("bottom_k") and config.get('mu1'):
            strategy_class = src.strategy.BottomKFreezingMomentum
        elif config.get("bottom_k"):
            strategy_class = src.strategy.BottomK
        elif config.get("top_k") and config.get('mu1'):
            strategy_class = src.strategy.TopKFreezingMomentum
        elif config.get("top_k"):
            strategy_class = src.strategy.TopK
        elif config.get('mu1'):
            strategy_class = src.strategy.FreezingMomentum
        else:
            strategy_class = flagon.server.FedAVG

        experiment_server = server_class(
            src.common.create_fmnist_model().get_parameters(),
            config,
            client_manager=src.server.DroppingClientManager(config['drop_round'], seed=seed),
            strategy=strategy_class()
        )

        if config.get("num_finetune_episodes") and config.get("adaptive_loss"):
            middle_server_class = src.middle_server.AdaptiveLossIntermediateFineTuner
        elif config.get("num_finetune_episodes"):
            middle_server_class = src.middle_server.IntermediateFineTuner
        elif config.get("adaptive_loss"):
            middle_server_class = src.middle_server.AdaptiveLoss
        else:
            middle_server_class = flagon.MiddleServer

        network_arch = {
            "clients": [
                {
                    "clients": 3,
                    "strategy": strategy_class(),
                    "middle_server_class": middle_server_class
                } for _ in range(5)
            ]
        }
        history = flagon.start_simulation(
            experiment_server,
            create_clients(data, src.common.create_fmnist_model, network_arch, seed=seed),
            network_arch
        )
        agg_res = {config['num_rounds']: history.aggregate_history[config['num_rounds']]}
        test_res = {config['num_rounds']: history.test_history[config['num_rounds']]}
        if config['drop_round'] <= config['num_rounds']:
            agg_res[config['drop_round']] = history.aggregate_history[config['drop_round']]
            test_res[config['drop_round']] = history.test_history[config['drop_round']]

        aggregate_results.append(agg_res)
        test_results.append(test_res)
        pbar.set_postfix(aggregate_results[-1][config['num_rounds']])
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
        experiment_config = src.common.get_experiment_config(json.load(f), args.id)
    print(f"Using config: {experiment_config}")
    experiment_config["analytics"] = [fairness_analytics]
    experiment_config["dataset"] = args.dataset
    experiment_config["eval_at"] = experiment_config["drop_round"]

    results = experiment(experiment_config)

    filename = "results/fairness_{}{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['analytics', 'round', 'mu1', 'adapt_loss', "eval_at"]]),
        "_momentum" if experiment_config.get("mu1") else ""
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")