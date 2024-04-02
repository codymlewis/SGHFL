import os
import argparse
import time
import json
import re
from tqdm import trange
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from l2rpn_baselines.PPO_SB3 import train


def gen_data(env, episodes, timesteps):
    data = {}
    for e in trange(episodes):
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        data[e] = {}
        for t in range(timesteps):
            data[e][t] = obs.to_json()
            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
            if done:
                obs = env.reset()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform experiments with a modified IEEE 118 bus power network.")
    parser.add_argument('-e', '--episodes', type=int, default=10, help="Number of episodes of training to perform.")
    parser.add_argument("-t", "--timesteps", type=int, default=100,
                        help="Number of steps per actor to perform in simulation.")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs("data", exist_ok=True)
    # env_name = "l2rpn_idf_2023"
    env_name = "l2rpn_case14_sandbox"  # Just for testing
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        reward_class=LinesCapacityReward,
        chronics_class=MultifolderWithCache,
    )
    if not os.path.exists(grid2op.get_current_local_dir() + f"/{env_name}_test"):
        env.train_val_split_random(pct_val=0.0, add_for_test="test", pct_test=50.0)
    print("Training agent...")
    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()
    agent = train(
        env,
        iterations=1_000,
        net_arch=[200, 200, 200],
    )
    print("Done.")

    print("Generating training dataset...")
    train_env = grid2op.make(env_name + "_train", backend=LightSimBackend(), reward_class=LinesCapacityReward)
    training_data = gen_data(train_env, args.episodes, args.timesteps)
    training_data_fn = 'data/training.json'
    with open(training_data_fn, 'w') as f:
        json.dump(training_data, f, indent=4)
    print(f"Training data written to {training_data_fn}")

    print("Generating testing dataset...")
    test_env = grid2op.make(env_name + "_test", backend=LightSimBackend(), reward_class=LinesCapacityReward)
    testing_data = gen_data(test_env, args.episodes, args.timesteps)
    testing_data_fn = 'data/testing.json'
    with open(testing_data_fn, 'w') as f:
        json.dump(testing_data, f, indent=4)
    print(f"Testing data written to {testing_data_fn}")

    print(f"Data generation took {time.time() - start_time} seconds")
