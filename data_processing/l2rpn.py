from typing import Dict
import os
import time
import re
import logging
import itertools
import json
from tqdm import trange
import numpy as np
from safetensors.numpy import save_file
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from l2rpn_baselines.PPO_SB3 import train

import duttagupta


def gen_data(env, agent, episodes: int, timesteps: int) -> Dict:
    data = {}
    for e in trange(episodes):
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        for t in range(timesteps):
            data[f"E{e}T{t}:load_p"] = obs.load_p
            data[f"E{e}T{t}:gen_p"] = obs.gen_p
            data[f"E{e}T{t}:time"] = np.array([obs.hour_of_day, obs.minute_of_hour])
            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
            if done:
                obs = env.reset()
    return data


def download():
    episodes = 10
    timesteps = 100

    start_time = time.time()
    os.makedirs("../data/", exist_ok=True)
    env_name = "l2rpn_idf_2023"
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        reward_class=LinesCapacityReward,
        chronics_class=MultifolderWithCache,
    )
    if not os.path.exists(grid2op.get_current_local_dir() + f"/{env_name}_test"):
        env.train_val_split_random(pct_val=0.0, add_for_test="test", pct_test=50.0)
    logging.info("Training agent...")
    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()
    agent = train(
        env,
        iterations=1_000,
        net_arch=[200, 200, 200],
    )
    logging.info("Done.")

    logging.info("Saving substation information...")
    substation_data = {
        "ids": np.union1d(env.load_to_subid, env.gen_to_subid),
        "load": env.load_to_subid,
        "gen": env.gen_to_subid,
    }
    substation_data_fn = "../data/l2rpn_substation.safetensors"
    save_file(substation_data, substation_data_fn)
    logging.info(f"Substation data written to {substation_data_fn}")

    logging.info("Generating training dataset...")
    train_env = grid2op.make(env_name + "_train", backend=LightSimBackend(), reward_class=LinesCapacityReward)
    training_data = gen_data(train_env, agent, episodes, timesteps)
    training_data_fn = '../data/l2rpn_training.safetensors'
    save_file(training_data, training_data_fn)
    logging.info(f"Training data written to {training_data_fn}")

    num_middle_servers = 10
    regions_arr = np.array_split(substation_data['ids'], num_middle_servers)
    regions = {}
    for region, cids in enumerate(regions_arr):
        for cid in cids:
            regions[str(cid)] = region
    regions_fn = "../data/l2rpn_regions.json"
    with open(regions_fn, 'w') as f:
        json.dump(regions, f)
    logging.info(f"Regions written to {regions_fn}")

    client_data = {}
    for cid in substation_data['ids']:
        load_idx = np.where(substation_data['load'] == cid)[0]
        if load_idx.shape[0] == 0:
            load_idx = np.array([-1])
        gen_idx = np.where(substation_data['gen'] == cid)[0]
        if gen_idx.shape[0] == 0:
            gen_idx = np.array([-1])
        client_data[str(cid)] = np.array([
            [
                training_data[f"E{e}T{t}:load_p"][load_idx].mean(axis=0) if np.any(load_idx != -1) else 0.0,
                training_data[f"E{e}T{t}:gen_p"][gen_idx].mean(axis=0) if np.any(gen_idx != -1) else 0.0,
            ]
            for e, t in itertools.product(range(episodes), range(timesteps))
        ])
    duttagupta_regions_fn = "../data/l2rpn_duttagupta_regions.json"
    with open(duttagupta_regions_fn, 'w') as f:
        json.dump(duttagupta.find_regions(client_data), f)
    logging.info(f"Duttagupta regions written to {duttagupta_regions_fn}")

    logging.info("Generating testing dataset...")
    test_env = grid2op.make(env_name + "_test", backend=LightSimBackend(), reward_class=LinesCapacityReward)
    testing_data = gen_data(test_env, agent, episodes, timesteps)
    testing_data_fn = '../data/l2rpn_testing.safetensors'
    save_file(testing_data, testing_data_fn)
    logging.info(f"Testing data written to {testing_data_fn}")

    logging.info(f"Data generation took {time.time() - start_time} seconds")


if __name__ == "__main__":
    download()
