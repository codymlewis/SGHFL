import re
import os
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from l2rpn_baselines.PPO_SB3 import train


if __name__ == "__main__":
    os.makedirs("./agent", exist_ok=True)
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(
        env_name,
        reward_class=LinesCapacityReward,
        backend=LightSimBackend(),
        chronics_class=MultifolderWithCache
    )

    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()
    train(
        env,
        iterations=1_000,
        # logs_dir="./agent/logs",
        save_path="./agent/saved_model",
        name="test",
        net_arch=[200, 200, 200],
        save_every_xxx_steps=2000,
    )
