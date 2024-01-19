from __future__ import annotations
import argparse
from functools import partial
import os
import math
import time
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from l2rpn_baselines.PPO_SB3 import evaluate
import numpy as np
import jax
import jax.numpy as jnp
from sklearn import metrics
from tqdm import trange

import fl
import adversaries
from logger import logger


def np_indexof(arr, val):
    index = np.where(arr == val)[0]
    if index.size > 0:
        return index
    return None


def setup(
    env,
    num_episodes,
    forecast_window,
    rounds,
    batch_size,
    num_middle_servers,
    server_aggregator="fedavg",
    middle_server_aggregator="fedavg",
    server_km=False,
    middle_server_km=False,
    intermediate_finetuning=0,
    compute_cs=False,
    attack="",
    seed=0
):
    forecast_model = fl.ForecastNet()
    global_params = forecast_model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 2 * forecast_window + 2)))
    substation_ids = set(env.load_to_subid) | set(env.gen_to_subid)

    if attack == "":
        adversary_type = fl.Client
    elif attack == "empty":
        adversary_type = adversaries.EmptyUpdater
    else:
        corroborator = adversaries.Corroborator(len(substation_ids), round(len(substation_ids) * (1 - 0.5)))
        if attack == "lie":
            adversary_type = partial(adversaries.LIE, corroborator=corroborator)
        elif attack == "ipm":
            adversary_type = partial(adversaries.IPM, corroborator=corroborator)
    clients = [
        (adversary_type if i + 1 > (len(substation_ids) * 0.5) else fl.Client)(
            i,
            forecast_model,
            np_indexof(env.load_to_subid, si),
            np_indexof(env.gen_to_subid, si),
            num_episodes,
            forecast_window,
        )
        for i, si in enumerate(substation_ids)
    ]
    if num_middle_servers:
        lower_clients = clients
        ms_cids = np.array_split(np.arange(len(lower_clients)), num_middle_servers)
        middle_servers = [
            fl.MiddleServer(
                global_params,
                [lower_clients[i] for i in cids],
                aggregate_fn=getattr(fl, middle_server_aggregator),
                kickback_momentum=middle_server_km,
            )
            for cids in ms_cids
        ]
        clients = middle_servers  # Middle servers are the clients for the top level server
    server = fl.Server(
        forecast_model,
        global_params,
        clients,
        rounds,
        batch_size,
        kickback_momentum=server_km,
        compute_cs=compute_cs,
        finetune_episodes=intermediate_finetuning,
        aggregate_fn=getattr(fl, server_aggregator),
    )
    return server


def add_data(train_env, agent, server, num_timesteps):
    obs = train_env.reset()
    reward = train_env.reward_range[0]
    done = False
    server.reset()
    for t in range(num_timesteps):
        server.add_data(obs)
        act = agent.act(obs, reward, done)
        obs, reward, done, info = train_env.step(act)
        if done:
            obs = train_env.reset()


def test_model(test_env, agent, server, forecast_window):
    server.setup_test()
    obs = test_env.reset()
    reward = train_env.reward_range[0]
    done = False
    client_forecasts, true_forecasts = [], []
    # for i in itertools.count():
    for i in range(100):
        true_forecast, client_forecast = server.add_test_data(obs)
        true_forecasts.append(true_forecast)
        client_forecasts.append(client_forecast)
        act = agent.act(obs, reward, done)
        obs, reward, done, info = test_env.step(act)
        if done:
            # break
            obs = test_env.reset()
    return np.array(client_forecasts[forecast_window - 1:-1]), np.array(true_forecasts[forecast_window - 1:-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments with a modified IEEE 118 bus power network.")
    parser.add_argument("-s", "--seed", type=int, default=64, help="Seed for RNG in the experiment.")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of episodes of training to perform.")
    parser.add_argument("-t", "--timesteps", type=int, default=100,
                        help="Number of steps per actor to perform in simulation.")
    parser.add_argument("--forecast-window", type=int, default=24,
                        help="Number of prior forecasts to include in the FL models data to inform its prediction.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds of FL training per episode.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for FL training.")
    parser.add_argument("--server-km", action="store_true", help="Use Kickback momentum at the FL server")
    parser.add_argument("--middle-server-km", action="store_true", help="Use Kickback momentum at the FL middle server")
    parser.add_argument("--intermediate-finetuning", type=int, default=0,
                        help="Finetune the FL models for n episodes prior to testing")
    parser.add_argument("--server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL server.")
    parser.add_argument("--middle-server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL middle server.")
    parser.add_argument("--num-middle-servers", type=int, default=10, help="Number of middle server for the HFL")
    parser.add_argument("--fairness", action="store_true", help="Perform the fairness evaluation.")
    parser.add_argument("--attack", type=str, default="",
                        help="Perform model poisoning on the federated learning model.")
    args = parser.parse_args()

    print(f"Running experiment with {vars(args)}")

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    rngkey = jax.random.PRNGKey(args.seed)
    env_name = "l2rpn_case14_sandbox"  # Change to l2rpn_idf_2023

    env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=LinesCapacityReward)
    if args.fairness:
        env_opponent_kwargs = {
            "opponent_attack_cooldown": 12*24,
            "opponent_attack_duration": 12*4,
            "opponent_budget_per_ts": 0.5,
            "opponent_init_budget": 0.,
            "opponent_action_class": grid2op.Action.PowerlineSetAction,
            "opponent_class": grid2op.Opponent.RandomLineOpponent,
            "opponent_budget_class": grid2op.Opponent.BaseActionBudget,
            "kwargs_opponent": {"lines_attacked": env.name_line}
        }
    else:
        env_opponent_kwargs = {}
    if not os.path.exists(grid2op.get_current_local_dir() + f"/{env_name}_test"):
        env.train_val_split_random(pct_val=0.0, add_for_test="test", pct_test=10.0)
    train_env = grid2op.make(
        env_name + "_train", backend=LightSimBackend(), reward_class=LinesCapacityReward, **env_opponent_kwargs
    )

    agent, _ = evaluate(
        env,
        nb_episode=0,
        load_path="./agent/saved_model",
        name="test",
        nb_process=1,
        verbose=False,
    )
    server = setup(
        env,
        args.episodes,
        args.forecast_window,
        args.rounds,
        args.batch_size,
        args.num_middle_servers,
        server_aggregator=args.server_aggregator,
        server_km=args.server_km,
        middle_server_aggregator=args.middle_server_aggregator,
        middle_server_km=args.middle_server_km,
        intermediate_finetuning=args.intermediate_finetuning,
        compute_cs=not args.attack and not args.fairness,
        attack=args.attack,
        seed=args.seed,
    )
    num_clients = server.num_clients

    logger.info("Generating data with simulations of the grid and training the models")
    for e in (pbar := trange(args.episodes)):
        add_data(train_env, agent, server, args.timesteps)
        if (e + 1) * args.timesteps > (args.batch_size + args.forecast_window):
            cs = server.step()

    # The testing phase
    logger.info("Testing the trained model.")
    test_env = grid2op.make(
        env_name + "_test", backend=LightSimBackend(), reward_class=LinesCapacityReward, **env_opponent_kwargs
    )
    client_forecasts, true_forecasts = test_model(test_env, agent, server, args.forecast_window)
    client_forecasts = client_forecasts.reshape(-1, 2)[args.forecast_window - 1:-1]
    true_forecasts = true_forecasts.reshape(-1, 2)[args.forecast_window - 1:-1]
    header = "seed,mae,rmse,r2_score,mape"
    results = "{},{},{},{},{}".format(
        args.seed,
        metrics.mean_absolute_error(true_forecasts, client_forecasts),
        math.sqrt(metrics.mean_squared_error(true_forecasts, client_forecasts)),
        metrics.r2_score(true_forecasts, client_forecasts),
        metrics.mean_absolute_percentage_error(true_forecasts, client_forecasts),
    )
    if cs:
        header += ",cosine_similarity"
        results += f",{cs}"
    logger.info(f"{results=}")

    header += ",args"
    results += f",{vars(args)}"

    # Record the results
    os.makedirs("results", exist_ok=True)
    file_suffix = "attack" if args.attack else "fairness" if args.fairness else "performance"
    filename = f"results/l2rpn_{file_suffix}.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(header + "\n")
    with open(filename, 'a') as f:
        f.write(results + "\n")
    print(f"Results written to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")
