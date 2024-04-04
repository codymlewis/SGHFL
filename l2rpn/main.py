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


def np_indexof(arr, val):
    index = np.where(arr == val)[0]
    if index.size > 0:
        return index
    return None


def setup(
    substation_data,
    num_episodes,
    forecast_window,
    rounds,
    batch_size,
    server_aggregator="fedavg",
    middle_server_aggregator="fedavg",
    intermediate_finetuning=0,
    attack="",
    pct_adversaries=0.5,
    pct_saturation=1.0,
    seed=0,
):
    num_middle_servers = 10
    forecast_model = fl.ForecastNet()
    global_params = forecast_model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 2 * forecast_window + 2)))
    substation_ids = substation_data['ids']

    if attack == "none":
        adversary_type = fl.Client
    elif attack == "empty":
        adversary_type = adversaries.EmptyUpdater
    else:
        corroborator = adversaries.Corroborator(len(substation_ids), round(len(substation_ids) * (1 - 0.5)))
        if attack == "lie":
            adversary_type = partial(adversaries.LIE, corroborator=corroborator)
        elif attack == "ipm":
            adversary_type = partial(adversaries.IPM, corroborator=corroborator)
    middle_servers = [
        fl.MiddleServer(
            global_params,
            [
                (adversary_type if (dc + 1 > math.ceil(num_middle_servers * (1 - pct_adversaries))) and
                    (c + 1 > math.ceil(len(sids) * (1 - pct_saturation))) else fl.Client)(
                    c,
                    forecast_model,
                    np_indexof(substation_data['load'], si),
                    np_indexof(substation_data['gen'], si),
                    num_episodes,
                    forecast_window
                ) for c, si in enumerate(sids)
            ],
            aggregator=middle_server_aggregator,
        ) for dc, sids in enumerate(np.array_split(substation_ids, num_middle_servers))
    ]
    server = fl.Server(
        forecast_model,
        global_params,
        middle_servers,
        rounds,
        batch_size,
        finetune_episodes=intermediate_finetuning,
        aggregator=server_aggregator,
    )
    return server


def train(
    server: fl.Server,
    episodes: int,
    timesteps: int,
    batch_size: int,
    forecast_window: int,
    drop_episode: int,
) -> float:
    training_data = load_file('data/training.safetensors')
    for e in trange(episodes):
        server.reset()
        for t in range(timesteps):
            obs_load_p = training_data[f"E{e}T{t}:load_p"]
            obs_gen_p = training_data[f"E{e}T{t}:gen_p"]
            obs_time = training_data[f"E{e}T{t}:time"]
            server.add_data(obs_load_p, obs_gen_p, obs_time)
        if (e + 1) * timesteps > (batch_size + forecast_window):
            cs = server.step()
        if e == drop_episode - 1:
            server.drop_clients()
    return cs


def test(
    server: fl.Server,
    episodes: int,
    timesteps: int,
    forecast_window: int,
    cs: float,
    args_dict: Dict[str, str | int | float | bool]
) -> Tuple[str, str]:
    testing_data = load_file('data/testing.safetensors')
    server.setup_test()
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

    header = "mae,rmse,r2_score,dropped mae,dropped rmse,dropped r2_score," + ",".join(args_dict.keys())
    results = "{},{},{},{},{},{},".format(
        metrics.mean_absolute_error(true_forecasts, client_forecasts),
        math.sqrt(metrics.mean_squared_error(true_forecasts, client_forecasts)),
        metrics.r2_score(true_forecasts, client_forecasts),
        metrics.mean_absolute_error(dropped_tfs, dropped_cfs) if dropped_cfs.shape[0] > 0 else 0.0,
        math.sqrt(metrics.mean_squared_error(dropped_tfs, dropped_cfs)) if dropped_cfs.shape[0] > 0 else 0.0,
        metrics.r2_score(dropped_tfs, dropped_cfs) if dropped_cfs.shape[0] > 0 else 0.0,
    )
    results += ",".join([str(v) for v in args_dict.values()])
    header += ",cosine_similarity"
    results += f",{cs}"
    logger.info(f"{results=}")
    return header, results


def process_forecasts(forecasts, forecast_window):
    forecasts = np.array(forecasts[forecast_window - 1:-1])
    forecasts = forecasts.reshape(-1, 2)[forecast_window - 1:-1]
    return forecasts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments with a modified IEEE 118 bus power network.")
    parser.add_argument("-s", "--seed", type=int, default=64, help="Seed for RNG in the experiment.")
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
    parser.add_argument("--intermediate-finetuning", type=int, default=0,
                        help="Finetune the FL models for n episodes prior to testing")
    parser.add_argument("--server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL server.")
    parser.add_argument("--middle-server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL middle server.")
    parser.add_argument("--attack", type=str, default="none",
                        help="Perform model poisoning on the federated learning model.")
    parser.add_argument('--drop-point', type=float, default=1.1,
                        help="Percent of episodes to pass before dropping clients")
    args = parser.parse_args()

    print(f"Running experiment with {vars(args)}")

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    rngkey = jax.random.PRNGKey(args.seed)
    drop_episode = round(args.episodes * args.drop_point)

    server = setup(
        load_file('data/substation.safetensors'),
        args.episodes,
        args.forecast_window,
        args.rounds,
        args.batch_size,
        server_aggregator=args.server_aggregator,
        middle_server_aggregator=args.middle_server_aggregator,
        intermediate_finetuning=args.intermediate_finetuning,
        attack=args.attack,
        pct_adversaries=args.pct_adversaries,
        pct_saturation=args.pct_saturation,
        seed=args.seed,
    )
    num_clients = server.num_clients

    cs = train(server, args.episodes, args.timesteps, args.batch_size, args.forecast_window, drop_episode)

    logger.info("Testing the trained model.")
    header, results = test(
        server, args.episodes, args.timesteps, args.forecast_window, cs, vars(args)
    )

    # Record the results
    os.makedirs("results", exist_ok=True)
    filename = "results/l2rpn.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(header + "\n")
    with open(filename, 'a') as f:
        f.write(results + "\n")
    print(f"Results written to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")
