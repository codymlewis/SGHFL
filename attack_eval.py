import argparse
from functools import partial
import gc
import json
import numpy as np
import einops
from tqdm.auto import trange

import flagon
from flagon.strategy import FedAVG
from flagon.common import Config, Parameters, Metrics, count_clients, to_attribute_array
from flagon.strategy import FedAVG

import src

import os
os.makedirs("results", exist_ok=True)


def create_clients(data, create_model_fn, network_arch, nadversaries, adversary_type, seed=None):
    Y = data['train']['Y']
    rng = np.random.default_rng(seed)
    idx = iter(src.common.lda(Y, count_clients(network_arch), rng, alpha=1000))
    nclients = count_clients(network_arch)
    corroborator = src.attacks.Corroborator(nclients)

    def create_client(client_id: str) -> src.client.Client:
        if int(client_id) > (nclients - nadversaries - 1):
            return adversary_type(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, corroborator)
        return src.client.Client(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, seed)
    return create_client


def bd_create_clients(data, create_model_fn, network_arch, nadversaries, adversary_type, from_y, to_y, seed=None):
    Y = data['train']['true Y']
    rng = np.random.default_rng(seed)
    idx = iter(src.common.lda(Y, count_clients(network_arch), rng, alpha=1000))
    nclients = count_clients(network_arch)
    corroborator = src.attacks.Corroborator(nclients)

    def create_client(client_id: str) -> src.client.Client:
        client_idx = next(idx)
        if int(client_id) > (nclients - nadversaries - 1):
            return adversary_type(data.select({"train": np.arange(len(data['train'])), "test": np.arange(len(data['test']))}), create_model_fn, corroborator, from_y, to_y, seed)
        return src.attacks.BackdoorClient(data.select({"train": client_idx, "test": np.arange(len(data['test']))}), create_model_fn, from_y, to_y, seed)
    return create_client


def experiment(config):
    results = {}
    if config['dataset'] == "fmnist":
        data = src.load_data.mnist()
    if config.get("from_y"):
        data.map(src.attacks.backdoor_mapping(data, config['from_y'], config['to_y']))
    data = data.normalise()

    strategy_type = {"fedavg": FedAVG, "median": src.strategy.Median, "centre": src.strategy.Centre}[config['aggregator']]
    if config.get("from_y"):
        adversary_type = src.attacks.BackdoorLIE
    else:
        adversary_type = {"empty": src.attacks.EmptyUpdater, "ipm": src.attacks.IPM, "lie": src.attacks.LIE}[config['attack']]
    train_results = []
    test_results = []
    for i in (pbar := trange(config['repeat'])):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        server = flagon.Server(
            {"fmnist": src.common.create_fmnist_model}[config['dataset']]().get_parameters(),
            config,
            strategy=strategy_type(),
        )
        network_arch = {"clients": config['num_clients']}
        history = flagon.start_simulation(
            server,
            (partial(bd_create_clients, from_y=config['from_y'], to_y=config['to_y']) if config.get("from_y") else create_clients)(
                data,
                src.common.create_fmnist_model,
                network_arch,
                nadversaries=config['num_adversaries'],
                adversary_type=adversary_type,
                seed=seed
            ),
            network_arch
        )
        train_results.append(history.aggregate_history[config['num_rounds']])
        test_results.append(history.test_history[config['num_rounds']])
        del server
        del network_arch
        gc.collect()
    return {"train": train_results, "test": test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating attacks upon FL.")
    parser.add_argument("-i", "--id", type=int, default=1, help="Which of the experiments in the config to perform (counts from 1).")
    parser.add_argument("-d", "--dataset", type=str, default="fmnist", help="Which of the datasets to perform the experiment with.")
    args = parser.parse_args()

    with open("configs/attack.json", 'r') as f:
        experiment_config = src.common.get_experiment_config(json.load(f), args.id)
    print(f"Using config: {experiment_config}")
    experiment_config["dataset"] = args.dataset

    results = experiment(experiment_config)

    filename = "results/attack_{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['round']])
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")

    # data = src.load_data.mnist()
    # data.map(src.attacks.backdoor_mapping(data, experiment_config['from_y'], experiment_config['to_y']))