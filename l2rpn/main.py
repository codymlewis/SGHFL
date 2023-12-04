from __future__ import annotations
import argparse
from functools import partial
import os
import itertools
import math
import time
import grid2op
from grid2op import Converter
from lightsim2grid import LightSimBackend
import numpy as np
import jax
import jax.numpy as jnp
from sklearn import metrics
from tqdm import trange

import fl
import adversaries
import drl
from logger import logger


def np_indexof(arr, val):
    index = np.where(arr == val)[0]
    if index.size > 0:
        return index
    return None


def fl_setup(
    env,
    num_episodes,
    forecast_window,
    fl_rounds,
    fl_batch_size,
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
        fl_rounds,
        fl_batch_size,
        kickback_momentum=server_km,
        compute_cs=compute_cs,
        finetune_episodes=intermediate_finetuning,
        aggregate_fn=getattr(fl, server_aggregator),
    )
    return server


def add_rl_data(rl_state, train_env, converter, last_obs, i, transitions, rngkey):
    pi, transitions.values[i] = rl_state.apply_fn(
        rl_state.params,
        np.concatenate((last_obs.to_vect(), transitions.client_forecasts[max(0, i - 1)].reshape(-1)))
    )
    transitions.actions[i] = pi.sample(seed=rngkey)
    transitions.log_probs[i] = pi.log_prob(transitions.actions[i])
    obs, transitions.rewards[i], transitions.dones[i], info = train_env.step(
        converter.convert_act(transitions.actions[i])
    )
    transitions.obs[i] = obs.to_vect()
    return obs


def add_data(train_env, converter, rl_state, server, transitions, num_actors, num_timesteps, forecast_window, rngkeys):
    counter = itertools.count()
    for a in range(num_actors):
        last_obs = train_env.reset()
        if server:
            server.reset()
        for t in range(num_timesteps):
            i = next(counter)
            obs = add_rl_data(rl_state, train_env, converter, last_obs, i, transitions, next(rngkeys))
            if server:
                server.add_data(obs, i, transitions)
            last_obs = obs
            if transitions.dones[i]:
                if server:
                    server.reset()
                last_obs = train_env.reset()
    return rl_state


def test_fl_and_rl_model(test_env, rl_state, server, forecast_window, rngkey):
    server.setup_test()
    obs = test_env.reset()
    client_forecasts, true_forecasts = [], []
    for i in itertools.count():
        true_forecast, client_forecast = server.add_test_data(obs)
        true_forecasts.append(true_forecast)
        client_forecasts.append(client_forecast)
        rngkey, _rngkey = jax.random.split(rngkey)
        pi, _ = rl_state.apply_fn(rl_state.params, np.concatenate((obs.to_vect(), client_forecasts[-1].reshape(-1))))
        action = pi.sample(seed=_rngkey)
        obs, reward, done, info = test_env.step(converter.convert_act(action))
        if done:
            break
        if i % 100 == 0 and i > 0:
            logger.info(f"Reached the {i}th test iteration")
    return i + 1, np.array(client_forecasts[forecast_window - 1:-1]), np.array(true_forecasts[forecast_window - 1:-1])


def test_rl_model(test_env, rl_state, rngkey):
    obs = test_env.reset()
    for i in itertools.count():
        rngkey, _rngkey = jax.random.split(rngkey)
        pi, _ = rl_state.apply_fn(rl_state.params, obs.to_vect())
        action = pi.sample(seed=_rngkey)
        obs, reward, done, info = test_env.step(converter.convert_act(action))
        if done:
            break
        if i % 100 == 0 and i > 0:
            logger.info(f"Reached the {i}th test iteration")
    logger.info(f"Finished at the {i}th test iteration")
    return i + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments with a modified IEEE 118 bus power network.")
    parser.add_argument("-s", "--seed", type=int, default=64, help="Seed for RNG in the experiment.")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of episodes of training to perform.")
    parser.add_argument("-a", "--actors", type=int, default=15,
                        help="Number of new simulations to perform during each episode.")
    parser.add_argument("-t", "--timesteps", type=int, default=100,
                        help="Number of steps per actor to perform in simulation.")
    parser.add_argument("--rl-steps", type=int, default=10, help="Number of steps of RL training per episode.")
    parser.add_argument("--rl-batch-size", type=int, default=128, help="Batch size for RL training.")
    parser.add_argument("--forecast-window", type=int, default=24,
                        help="Number of prior forecasts to include in the FL models data to inform its prediction.")
    parser.add_argument("--fl-rounds", type=int, default=10, help="Number of rounds of FL training per episode.")
    parser.add_argument("--fl-batch-size", type=int, default=128, help="Batch size for FL training.")
    parser.add_argument("--fl-server-km", action="store_true", help="Use Kickback momentum at the FL server")
    parser.add_argument("--fl-middle-server-km", action="store_true", help="Use Kickback momentum at the FL middle server")
    parser.add_argument("--intermediate-finetuning", type=int, default=0,
                        help="Finetune the FL models for n episodes prior to testing")
    parser.add_argument("--fl-server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL server.")
    parser.add_argument("--fl-middle-server-aggregator", type=str, default="fedavg",
                        help="Aggregation algorithm to use at the FL middle server.")
    parser.add_argument("--no-fl", action="store_true", help="Specify to not use federated learning for this experiment.")
    parser.add_argument("--num-middle-servers", type=int, default=10, help="Number of middle server for the HFL")
    parser.add_argument("--fairness", action="store_true", help="Perform the fairness evaluation.")
    parser.add_argument("--attack", type=str, default="",
                        help="Perform model poisoning on the federated learning model.")
    args = parser.parse_args()

    print(f"Running experiment with {vars(args)}")

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    rngkey = jax.random.PRNGKey(args.seed)
    env_name = "rte_case14_realistic"  # Change to l2rpn_idf_2023
    perform_fl = not args.no_fl

    env = grid2op.make(env_name)
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
    train_env = grid2op.make(env_name + "_train", backend=LightSimBackend(), **env_opponent_kwargs)

    obs = train_env.reset()
    converter = Converter.ToVect(env.action_space)
    obs_shape = obs.to_vect().shape
    act_shape = (env.action_space.n,)

    if perform_fl:
        server = fl_setup(
            env,
            args.episodes,
            args.forecast_window,
            args.fl_rounds,
            args.fl_batch_size,
            args.num_middle_servers,
            server_aggregator=args.fl_server_aggregator,
            server_km=args.fl_server_km,
            middle_server_aggregator=args.fl_middle_server_aggregator,
            middle_server_km=args.fl_middle_server_km,
            intermediate_finetuning=args.intermediate_finetuning,
            compute_cs=not args.attack and not args.fairness,
            attack=args.attack,
            seed=args.seed,
        )
        num_clients = server.num_clients
    else:
        server = None
        num_clients = 0
    rl_state = drl.setup(env, obs, num_clients, args.seed)

    logger.info("Generating data with simulations of the grid and training the models")
    for e in (pbar := trange(args.episodes)):
        # We generate all of the random generation keys that we will need pre-emptively
        rngkeys = jax.random.split(rngkey, args.actors * args.timesteps + 1)
        rngkey = rngkeys[0]
        # Allocate the memory for our data batch and the index where each sample is stored
        transitions = drl.TransitionBatch.init(
            args.timesteps, args.actors, num_clients, obs_shape, act_shape, args.seed + e
        )
        # Now we perform the actor loop from Algorithm 1 in http://arxiv.org/abs/1707.06347
        rl_state = add_data(
            train_env,
            converter,
            rl_state,
            server,
            transitions,
            args.actors,
            args.timesteps,
            args.forecast_window,
            iter(rngkeys[1:]),
        )

        if perform_fl:
            cs = server.step()
        loss, rl_state = drl.reinforcement_learning(rl_state, transitions, args.rl_steps, args.rl_batch_size)
        pbar.set_postfix_str(f"RL Loss: {loss:.5f}")

    # The testing phase
    logger.info("Testing how long the trained model can run the power network.")
    test_env = grid2op.make(env_name + "_test", backend=LightSimBackend(), **env_opponent_kwargs)
    if perform_fl:
        rl_score, client_forecasts, true_forecasts = test_fl_and_rl_model(
            test_env, rl_state, server, args.forecast_window, rngkey
        )
        client_forecasts = client_forecasts.reshape(-1, 2)[args.forecast_window - 1:-1]
        true_forecasts = true_forecasts.reshape(-1, 2)[args.forecast_window - 1:-1]
        header = "seed,rl_score,mae,rmse,r2_score"
        results = "{},{},{},{},{}".format(
            args.seed,
            rl_score,
            metrics.mean_absolute_error(true_forecasts, client_forecasts),
            math.sqrt(metrics.mean_squared_error(true_forecasts, client_forecasts)),
            metrics.r2_score(true_forecasts, client_forecasts),
        )
        if cs:
            header += ",cosine_similarity"
            results += f",{cs}"
    else:
        rl_score = test_rl_model(test_env, rl_state, rngkey)
        header = "seed,rl_score"
        results = f"{args.seed},{rl_score}"
    print(f"Ran the network for {rl_score} time steps")
    logger.info(f"{results=}")

    header += ",args"
    results += f",{vars(args)}"

    # Record the results
    os.makedirs("results", exist_ok=True)
    file_suffix = "attack" if args.attack else "fairness" if args.fairness else "performance"
    filename = f"results/l2rpn{'_fl' if perform_fl else ''}_{file_suffix}.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(header + "\n")
    with open(filename, 'a') as f:
        f.write(results + "\n")
    print(f"Results written to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")
