from __future__ import annotations
from typing import Tuple, Dict
import argparse
from functools import partial
import os
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
from sklearn import metrics
from tqdm import trange
from safetensors.numpy import load_file

import fl
import adversaries
from logger import logger
import data_manager


def np_indexof(arr, val):
    index = np.where(arr == val)[0]
    if index.size > 0:
        return index
    return None


def get_adversary_class(attack, nclients, pct_adversaries, pct_saturation):
    if attack == "none":
        adversary_type = fl.Client
    elif attack == "empty":
        adversary_type = adversaries.EmptyUpdater
    else:
        corroborator = adversaries.Corroborator(nclients, round(nclients * (1 - (pct_adversaries * pct_saturation))))
        if attack == "lie":
            adversary_type = partial(adversaries.LIE, corroborator=corroborator)
        else:
            adversary_type = partial(adversaries.IPM, corroborator=corroborator)
    return adversary_type


def l2rpn_setup(
    num_episodes,
    forecast_window,
    batch_size,
    server_aggregator="fedavg",
    middle_server_aggregator="fedavg",
    model="FFN",
    attack="",
    pct_adversaries=0.5,
    pct_saturation=1.0,
    seed=0,
):
    match model:
        case "FFN":
            forecast_model = fl.FFN()
        case "CNN":
            forecast_model = fl.CNN()
        case "LSTM":
            forecast_model = fl.LSTM()
        case "BiGRU":
            forecast_model = fl.BiGRU()
        case "Attention":
            forecast_model = fl.Attention()
        case "CNN_BiGRU_Attention":
            forecast_model = fl.CNN_BiGRU_Attention()
        case _:
            raise NotImplementedError(f"Model {model} is not implemented")

    global_params = forecast_model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 2 * forecast_window + 2)))
    substation_data = load_file('../data/l2rpn_substation.safetensors')
    substation_ids = substation_data['ids']
    adversary_type = get_adversary_class(attack, len(substation_ids), pct_adversaries, pct_saturation)
    regions = data_manager.load_regions('l2rpn', duttagupta=server_aggregator == "duttagupta")
    middle_servers = [
        fl.MiddleServer(
            global_params,
            [
                (adversary_type if (dc + 1 > math.ceil(len(regions) * (1 - pct_adversaries))) and
                    (c + 1 > math.ceil(len(sids) * (1 - pct_saturation))) else fl.Client)(
                    c,
                    forecast_model,
                    info={
                        "load_id": np_indexof(substation_data['load'], si),
                        "gen_id": np_indexof(substation_data['gen'], si),
                        "num_episodes": num_episodes,
                        "forecast_window": forecast_window,
                    }
                ) for c, si in enumerate(sids)
            ],
            aggregator=middle_server_aggregator,
        ) for dc, sids in enumerate(regions)
    ]
    server = fl.Server(
        forecast_model,
        global_params,
        middle_servers,
        batch_size,
        aggregator=server_aggregator,
    )
    return server


def power_setup(
    batch_size,
    client_data,
    regions,
    server_aggregator="fedavg",
    middle_server_aggregator="fedavg",
    attack="",
    pct_adversaries=0.5,
    pct_saturation=1.0,
    seed=0,
):
    forecast_model = fl.PowerNet(classes=client_data[0].Y.shape[-1])
    global_params = forecast_model.init(jax.random.PRNGKey(seed), client_data[0].X[:1])
    adversary_type = get_adversary_class(attack, len(client_data), pct_adversaries, pct_saturation)
    middle_servers = [
        fl.MiddleServer(
            global_params,
            [
                (adversary_type if (dc + 1 > math.ceil(len(regions) * (1 - pct_adversaries))) and
                    (c + 1 > math.ceil(len(cids) * (1 - pct_saturation))) else fl.Client)(
                    c,
                    forecast_model,
                    data=client_data[c],
                ) for c in cids
            ],
            aggregator=middle_server_aggregator,
        ) for dc, cids in enumerate(regions)
    ]
    server = fl.Server(
        forecast_model,
        global_params,
        middle_servers,
        batch_size,
        aggregator=server_aggregator,
    )
    return server


def l2rpn_train(
    server: fl.Server,
    episodes: int,
    timesteps: int,
    rounds: int,
    batch_size: int,
    forecast_window: int,
    drop_episode: int,
) -> float:
    training_data = load_file('../data/l2rpn_training.safetensors')
    for e in trange(episodes):
        server.reset()
        for t in range(timesteps):
            obs_load_p = training_data[f"E{e}T{t}:load_p"]
            obs_gen_p = training_data[f"E{e}T{t}:gen_p"]
            obs_time = training_data[f"E{e}T{t}:time"]
            server.add_data(obs_load_p, obs_gen_p, obs_time)
        if (e + 1) * timesteps > (batch_size + forecast_window):
            for r in range(rounds):
                cs, all_losses = server.step(compute_cs=(e == (episodes - 1)) and (r == (rounds - 1)))
            logger.info(f"Global loss at episode {e + 1}: {np.mean(all_losses):.5f}")
        if e == drop_episode - 1:
            server.drop_clients()
    return cs


def power_train(server: fl.Server, rounds: int, drop_round: int) -> float:
    for r in trange(rounds):
        cs, all_losses = server.step(compute_cs=r == (rounds - 1), client_steps=10)
        logger.info(f"Global loss at round {r + 1}: {np.mean(all_losses):.5f}")
        if r == drop_round - 1:
            server.drop_clients()
    return cs


def l2rpn_test(
    server: fl.Server,
    episodes: int,
    timesteps: int,
    forecast_window: int,
    cs: float,
    args_dict: Dict[str, str | int | float | bool],
    finetune_steps: int = 0,
) -> Tuple[str, str]:
    testing_data = load_file('../data/l2rpn_testing.safetensors')
    server.setup_test(finetune_steps=finetune_steps)
    client_forecasts, true_forecasts = [], []
    dropped_cfs, dropped_tfs = [], []
    for e in trange(episodes):
        server.reset()
        for t in range(timesteps):
            obs_load_p = testing_data[f"E{e}T{t}:load_p"]
            obs_gen_p = testing_data[f"E{e}T{t}:gen_p"]
            obs_time = testing_data[f"E{e}T{t}:time"]
            true_forecast, client_forecast, dropped_tf, dropped_cf = server.add_test_data(
                obs_load_p, obs_gen_p, obs_time,
            )
            true_forecasts.append(true_forecast)
            client_forecasts.append(client_forecast)
            dropped_tfs.append(dropped_tf)
            dropped_cfs.append(dropped_cf)
    client_forecasts = process_forecasts(client_forecasts, forecast_window)
    true_forecasts = process_forecasts(true_forecasts, forecast_window)
    dropped_cfs = process_forecasts(dropped_cfs, forecast_window)
    dropped_tfs = process_forecasts(dropped_tfs, forecast_window)
    return measure_and_format_results(args_dict, client_forecasts, true_forecasts, dropped_cfs, dropped_tfs, cs)


def process_forecasts(forecasts, forecast_window):
    forecasts = np.array(forecasts[forecast_window - 1:-1])
    forecasts = forecasts.reshape(-1, 2)[forecast_window - 1:-1]
    return forecasts


def measure_and_format_results(args_dict, client_forecasts, true_forecasts, dropped_cfs, dropped_tfs, cs):
    header = "mae,r2_score,dropped mae,dropped r2_score," + ",".join(args_dict.keys())
    results = "{},{},{},{},".format(
        metrics.mean_absolute_error(true_forecasts, client_forecasts),
        metrics.r2_score(true_forecasts, client_forecasts),
        metrics.mean_absolute_error(dropped_tfs, dropped_cfs) if dropped_cfs.shape[0] > 0 else 0.0,
        metrics.r2_score(dropped_tfs, dropped_cfs) if dropped_cfs.shape[0] > 0 else 0.0,
    )
    results += ",".join([str(v) for v in args_dict.values()])
    header += ",cosine_similarity"
    results += f",{cs}"
    logger.info(f"{results=}")
    return header, results


def power_test(
    server: fl.Server,
    X_test,
    Y_test,
    cs: float,
    args_dict: Dict[str, str | int | float | bool],
    finetune_steps: int = 0,
) -> Tuple[str, str]:
    if finetune_steps > 0:
        server.finetune(finetune_steps)
    mae, res_sum_squares, total_sum_squares = 0.0, 0.0, 0.0
    nclients = 0
    for client in server.clients:
        for pred_forecasts in client.forecast(X_test):
            mae += np.abs(Y_test - pred_forecasts).mean()
            res_sum_squares += ((pred_forecasts - Y_test)**2).sum()
            total_sum_squares += ((Y_test - Y_test.mean())**2).sum()
            nclients += 1
    mae /= nclients
    r2_score = 1 - (res_sum_squares / total_sum_squares)
    ndropped_clients = len(server.all_clients) - len(server.clients)
    dropped_mae, dropped_rss, dropped_tss = 0.0, 0.0, 0.0
    ndclients = 0
    if ndropped_clients:
        for client in server.all_clients[:-ndropped_clients]:
            for pred_forecasts in client.forecast(X_test):
                dropped_mae += np.abs(Y_test - pred_forecasts).mean()
                dropped_rss += ((pred_forecasts - Y_test)**2).sum()
                dropped_tss += ((Y_test - Y_test.mean())**2).sum()
                ndclients += 1
        dropped_mae /= ndclients
        dropped_r2_score = 1 - (dropped_rss / dropped_tss)
    else:
        dropped_r2_score = 0.0
    header = "mae,rmse,r2_score,dropped mae,dropped rmse,dropped r2_score," + ",".join(args_dict.keys())
    results = f"{mae},{r2_score},{dropped_mae},{dropped_r2_score},"
    results += ",".join([str(v) for v in args_dict.values()])
    header += ",cosine_similarity"
    results += f",{cs}"
    logger.info(f"{results=}")
    return header, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments with Smart grid datasets.")
    parser.add_argument("-s", "--seed", type=int, default=64, help="Seed for RNG in the experiment.")
    parser.add_argument("-d", "--dataset", type=str, default="l2rpn", help="The dataset to train on.")
    parser.add_argument("-e", "--episodes", type=int, default=10, help="Number of episodes of training to perform.")
    parser.add_argument("-t", "--timesteps", type=int, default=100,
                        help="Number of steps per actor to perform in simulation.")
    parser.add_argument("--forecast-window", type=int, default=24,
                        help="Number of prior forecasts to include in the FL models data to inform its prediction.")
    parser.add_argument("--rounds", type=int, default=50, help="Number of rounds of FL training per episode.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for FL training.")
    parser.add_argument("--pct-adversaries", type=float, default=0.5,
                        help="Percentage of clients to assign as adversaries, if performing an attack evaluation")
    parser.add_argument("--pct-saturation", type=float, default=1.0,
                        help="The percentage of clients under adversary middle servers to assign as adversaries.")
    parser.add_argument("--server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL server.")
    parser.add_argument("--middle-server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL middle server.")
    parser.add_argument("--model", type=str, default="FFN",
                        help="Model architecture to train.")
    parser.add_argument("--attack", type=str, default="none",
                        help="Perform model poisoning on the federated learning model.")
    parser.add_argument('--drop-point', type=float, default=1.1,
                        help="Percent of episodes to pass before dropping clients")
    parser.add_argument('-if', '--intermediate-finetuning', action="store_true",
                        help="Whether to perform intermediate finetuning")
    args = parser.parse_args()

    print(f"Running experiment with {vars(args)}")

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    rngkey = jax.random.PRNGKey(args.seed)
    if args.dataset == "l2rpn":
        drop_episode = round(args.episodes * args.drop_point)
    else:
        drop_episode = round(args.rounds * args.drop_point)

    if args.dataset == "l2rpn":
        server = l2rpn_setup(
            args.episodes,
            args.forecast_window,
            args.batch_size,
            server_aggregator=args.server_aggregator,
            middle_server_aggregator=args.middle_server_aggregator,
            model=args.model,
            attack=args.attack,
            pct_adversaries=args.pct_adversaries,
            pct_saturation=args.pct_saturation,
            seed=args.seed,
        )
    else:
        client_data, X_test, Y_test = data_manager.load_data(args.dataset)
        server = power_setup(
            args.batch_size,
            client_data,
            data_manager.load_regions(args.dataset, duttagupta=args.server_aggregator == "duttagupta"),
            server_aggregator=args.server_aggregator,
            middle_server_aggregator=args.middle_server_aggregator,
            attack=args.attack,
            pct_adversaries=args.pct_adversaries,
            pct_saturation=args.pct_saturation,
            seed=args.seed,
        )
    num_clients = server.num_clients

    if args.dataset == "l2rpn":
        cs = l2rpn_train(
            server, args.episodes, args.timesteps, args.rounds, args.batch_size, args.forecast_window, drop_episode
        )
    else:
        cs = power_train(server, args.rounds, drop_episode)

    logger.info("Testing the trained model.")
    args_dict = vars(args).copy()
    del args_dict["intermediate_finetuning"]
    del args_dict["seed"]

    if args.dataset == "l2rpn":
        header, results = l2rpn_test(server, args.episodes, args.timesteps, args.forecast_window, cs, args_dict)
    else:
        header, results = power_test(server, X_test, Y_test, cs, args_dict)

    # Record the results
    os.makedirs("results", exist_ok=True)
    filename = "results/results.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(header + "\n")
    with open(filename, 'a') as f:
        f.write(results + "\n")
    logger.info(f"Results written to {filename}")

    if args.intermediate_finetuning:
        logger.info("Evaluating with finetuning...")
        args_dict["server_aggregator"] += " IF"
        if args.dataset == "l2rpn":
            _, results = l2rpn_test(
                server, args.episodes, args.timesteps, args.forecast_window, cs, args_dict, finetune_steps=5
            )
        else:
            _, results = power_test(server, X_test, Y_test, cs, args_dict, finetune_steps=5)
        with open(filename, 'a') as f:
            f.write(results + "\n")
        logger.info(f"Results written to {filename}")

    logger.info(f"Experiment took {time.time() - start_time} seconds")
