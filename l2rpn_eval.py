from __future__ import annotations
import collections
import os
import itertools
import logging
from typing import NamedTuple, Tuple, Iterator
import grid2op
from grid2op import Converter
from lightsim2grid import LightSimBackend
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import chex
import optax
from tqdm import tqdm, trange

logger = logging.getLogger("L2RPN experiment")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('| %(name)s %(levelname)s @ %(asctime)s in %(filename)s:%(lineno)d | %(message)s'))
ch.setStream(tqdm)
ch.terminator = ""
logger.addHandler(ch)


class ForecastNet(nn.Module):
    "Neural network for predicting future power load and generation"

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(6)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        return x


class ForecastBatch(NamedTuple):
    X: chex.Array
    Y: chex.Array
    indexer: Iterator

    def create(dataset_size: int, forecast_window: int) -> ForecastBatch:
        return ForecastBatch(
            np.zeros((dataset_size, 2 * forecast_window + 2)),
            np.zeros((dataset_size, 2)),
            itertools.count()
        )

    def add(self, x, y):
        i = next(self.indexer) % self.Y.shape[0]
        self.X[i] = x
        self.Y[i] = y

    def __len__(self) -> int:
        return self.Y.shape[0]


@jax.jit
def forecast_learner_step(
    state: train_state.TrainState,
    X: chex.Array,
    Y: chex.Array,
) -> Tuple[float, train_state.TrainState]:
    def loss_fn(params):
        predictions = state.apply_fn(params, X)
        return jnp.mean(0.5 * (predictions - Y)**2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


@jax.jit
def fedavg(all_params):
    return jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *all_params)


class Client:
    def __init__(self, model, load_id, gen_id, num_episodes, forecast_window, buffer_size: int = 1000):
        self.past_load = collections.deque([], maxlen=forecast_window)
        self.past_gen = collections.deque([], maxlen=forecast_window)
        self.data = ForecastBatch.create(num_episodes * buffer_size, forecast_window)
        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.PRNGKey(0), self.data.X[:1]),
            tx=optax.adam(0.01),
        )
        self.load_id = load_id
        self.gen_id = gen_id


def np_indexof(arr, val):
    index = np.where(arr == val)[0]
    if index.size > 0:
        return index.item()
    return -1


if __name__ == "__main__":
    seed = 64
    rng = np.random.default_rng(seed)
    num_episodes = 10
    forecast_window = 24
    env_name = "rte_case14_realistic"  # Change to lrpn_idf_2023

    env = grid2op.make(env_name)
    if not os.path.exists(grid2op.get_current_local_dir() + f"/{env_name}_test"):
        env.train_val_split_random(pct_val=0.0, add_for_test="test", pct_test=10.0)
    train_env = grid2op.make(env_name + "_train", backend=LightSimBackend())
    obs = train_env.reset()
    converter = Converter.ToVect(env.action_space)
    obs_shape = obs.to_vect().shape
    act_shape = (env.action_space.n,)
    forecast_model = ForecastNet()
    global_params = forecast_model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 2 * forecast_window + 2)))
    clients = [
        Client(
            forecast_model,
            np_indexof(env.load_to_subid, si), np_indexof(env.gen_to_subid, si),
            num_episodes,
            forecast_window,
        )
        for si in (set(env.load_to_subid) | set(env.gen_to_subid))
    ]

    logger.info("Generating data with simulations of the grid")
    for e in (pbar := trange(num_episodes)):
        obs = train_env.reset()
        past_load, past_gen = collections.deque([], maxlen=forecast_window), collections.deque([], maxlen=forecast_window)
        for i in itertools.count():
            obs, reward, done, info = train_env.step(converter.convert_act(rng.normal(size=act_shape)))
            for client in clients:
                load_p = obs.load_p[client.load_id] if client.load_id >= 0 else 0.0
                gen_p = obs.gen_p[client.gen_id] if client.gen_id >= 0 else 0.0
                if len(past_load) == forecast_window:
                    client.data.add(
                        np.concatenate(
                            (np.array([obs.hour_of_day, obs.minute_of_hour]), np.array(past_load), np.array(past_gen))),
                        np.array([load_p, gen_p])
                    )
                past_load.append(load_p)
                past_gen.append(gen_p)

            if done:
                break

    logger.info("Starting federated training of the forecast model")
    for i in (pbar := trange(1000)):
        all_params = []
        all_losses = []
        for client in clients:
            client.state = client.state.replace(params=global_params)
            idx = rng.choice(len(client.data), 128, replace=False)
            loss, client.state = forecast_learner_step(client.state, client.data.X[idx], client.data.Y[idx])
            all_params.append(client.state.params)
            all_losses.append(loss)
        global_params = fedavg(all_params)
        pbar.set_postfix_str(f"Loss: {np.mean(all_losses):.5f}")
