import argparse
from functools import partial
import gc
import json
import numpy as np
from tqdm.auto import trange

import fl
from fl.common import count_clients

import os
os.makedirs("results", exist_ok=True)


def create_clients(data, create_model_fn, network_arch, client_type, nadversaries, adversary_type, seed=None):
    Y = data['train']['Y']
    rng = np.random.default_rng(seed)
    idx = iter(fl.common.lda(Y, count_clients(network_arch), rng, alpha=1000))
    nclients = count_clients(network_arch)
    corroborator = fl.attacks.Corroborator(nclients)

    def create_client(client_id: str) -> fl.client.Client:
        if int(client_id) > (nclients - nadversaries - 1):
            return adversary_type(
                data.select({"train": next(idx), "test": np.arange(len(data['test']))}),
                create_model_fn,
                corroborator,
                seed=seed
            )
        return client_type(
            data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, seed=seed
        )
    return create_client


def create_sh_clients(create_model_fn, network_arch, nadversaries, client_type, adversary_type, seed=None):
    get_customer_data = fl.load_data.solar_home()
    nclients = count_clients(network_arch)
    corroborator = fl.attacks.Corroborator(nclients)

    def create_client(client_id: str) -> fl.client.Client:
        cid = int(client_id)
        client_X, client_Y = get_customer_data(cid + 1)
        client_data = fl.data_manager.Dataset({
            "train": {"X": client_X[:300 * 24], "Y": client_Y[:300 * 24]},
            "test": {"X": client_X[300 * 24:], "Y": client_Y[300 * 24:]}
        })
        if cid > (nclients - nadversaries - 1):
            return adversary_type(client_data, create_model_fn, corroborator, seed=seed)
        return fl.client.Client(client_data, create_model_fn, seed=seed)
    return create_client


def bd_create_sh_clients(create_model_fn, network_arch, client_type, nadversaries, adversary_type, seed=None):
    get_customer_data = fl.load_data.solar_home()
    nclients = count_clients(network_arch)
    corroborator = fl.attacks.Corroborator(nclients)

    def create_client(client_id: str) -> fl.client.Client:
        cid = int(client_id)
        client_X, client_Y = get_customer_data(cid + 1)
        client_data = fl.data_manager.Dataset({
            "train": {"X": client_X[:300 * 24], "Y": client_Y[:300 * 24]},
            "test": {"X": client_X[300 * 24:], "Y": client_Y[300 * 24:]}
        })
        # Backdoor mapping
        backdoor_idx = {
            "train": client_data['train']['X'][:, -1, 3] > 0.9,
            "test": client_data['test']['X'][:, -1, 3] > 0.9
        }
        client_data = client_data.map(fl.attacks.sh_backdoor_mapping())
        if cid > (nclients - nadversaries - 1):
            return adversary_type(client_data, create_model_fn, corroborator, backdoor_idx, seed=seed)
        return fl.attacks.BackdoorClient(client_data, create_model_fn, backdoor_idx, seed=seed)
    return create_client


def experiment(config):
    data = fl.load_data.mnist()
    if config.get("from_y"):
        data.map(fl.attacks.backdoor_mapping(data, config['from_y'], config['to_y']))
    data = data.normalise()

    if config.get("from_y") or config.get("backdoor"):
        adversary_type = fl.attacks.BackdoorLIE
        client_type = fl.attacks.BackdoorClient
    else:
        adversary_type = {
            "empty": fl.attacks.EmptyUpdater, "ipm": fl.attacks.IPM, "lie": fl.attacks.LIE
        }[config['attack']]
        client_type = fl.client.Client

    train_results = []
    test_results = []
    create_model_fn = fl.common.create_fmnist_model

    for i in trange(config['repeat']):
        seed = round(np.pi**i + np.exp(i)) % 2**32
        server = fl.server.Server(
            create_model_fn().get_parameters(),
            config,
            strategy_name=config.get('aggregator'),
        )

        network_arch = {"clients": config['num_clients']}
        create_clients_fn = partial(create_clients, data)

        clients = create_clients_fn(
            create_model_fn,
            network_arch,
            client_type=client_type,
            nadversaries=config['num_adversaries'],
            adversary_type=adversary_type,
            seed=seed
        )
        history = fl.simulation.start_simulation(server, clients, network_arch)

        if config.get("eval_every"):
            train_results.append(history.aggregate_history)
            test_results.append(history.test_history)
        else:
            train_results.append(history.aggregate_history[config['num_rounds']])
            test_results.append(history.test_history[config['num_rounds']])

        del server
        del network_arch
        gc.collect()
    return {"train": train_results, "test": test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments evaluating attacks upon FL.")
    parser.add_argument("-i", "--id", type=int, default=1,
                        help="Which of the experiments in the config to perform (counts from 1).")
    args = parser.parse_args()

    with open("configs/attack.json", 'r') as f:
        experiment_config = fl.common.get_experiment_config(json.load(f), args.id)
    print(f"Using config: {experiment_config}")
    results = experiment(experiment_config)
    filename = "results/attack_{}.json".format(
        '_'.join([f'{k}={v}' for k, v in experiment_config.items() if k not in ['round']])
    )
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {filename}")
