from __future__ import annotations
import collections
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import chex
import flax.linen as nn
from flax.training import train_state
import optax
import clax
from sklearn import metrics

from logger import logger


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


class ForecastBatch:
    def __init__(self, X: chex.Array, Y: chex.Array, index: int):
        self.X = X
        self.Y = Y
        self.i = index

    def create(dataset_size: int, forecast_window: int) -> ForecastBatch:
        return ForecastBatch(
            np.zeros((dataset_size, 2 * forecast_window + 2)),
            np.zeros((dataset_size, 2)),
            0
        )

    def add(self, x, y):
        self.i = (self.i + 1) % self.Y.shape[0]
        self.X[self.i] = x
        self.Y[self.i] = y

    def __len__(self) -> int:
        return min(self.i, self.Y.shape[0])


@jax.jit
def learner_step(
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


@jax.jit
def median(all_params):
    return jax.tree_util.tree_map(lambda *x: jnp.median(jnp.array(x), axis=0), *all_params)


@jax.jit
def topk(all_params, k=0.5):
    avg_params = jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *all_params)

    def prune(x):
        K = round((1 - k) * x.size)
        return jnp.where(x >= jnp.partition(x.reshape(-1), K)[K], x, 0)

    return jax.tree_util.tree_map(prune, avg_params)


@jax.jit
def centre(all_params):
    nclusters = len(all_params) // 4 + 1
    return jax.tree_util.tree_map(lambda *x: clax.kmeans.fit(jnp.array(x), nclusters)['centroids'].mean(axis=0), *all_params)


class Client:
    def __init__(self, client_id, model, load_id, gen_id, num_episodes, forecast_window, buffer_size: int = 1000, seed=0):
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
        self.forecast_window = forecast_window
        self.id = client_id
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.past_load = collections.deque([], maxlen=self.forecast_window)
        self.past_gen = collections.deque([], maxlen=self.forecast_window)

    def add_data(self, obs, i, transitions):
        load_p = obs.load_p[self.load_id].sum() if self.load_id is not None else 0.0
        gen_p = obs.gen_p[self.gen_id].sum() if self.gen_id is not None else 0.0
        if len(self.past_load) == self.forecast_window:
            sample = np.concatenate((
                np.array([obs.hour_of_day, obs.minute_of_hour]),
                np.array(self.past_load),
                np.array(self.past_gen),
            ))
            self.data.add(sample, np.array([load_p, gen_p]))
            transitions.client_forecasts[max(0, i - 1), self.id] = forecast(self.state, sample)
        self.past_load.append(load_p)
        self.past_gen.append(gen_p)

    def add_test_data(self, obs):
        true_forecast, predicted_forecast = np.zeros(2), np.zeros(2)
        load_p = obs.load_p[self.load_id].sum() if self.load_id is not None else 0.0
        gen_p = obs.gen_p[self.gen_id].sum() if self.gen_id is not None else 0.0
        if len(self.past_load) == self.forecast_window:
            true_forecast = np.array([load_p, gen_p])
        self.past_load.append(load_p)
        self.past_gen.append(gen_p)
        if len(self.past_load) == self.forecast_window:
            sample = np.concatenate((
                np.array([obs.hour_of_day, obs.minute_of_hour]),
                np.array(self.past_load),
                np.array(self.past_gen),
            ))
            self.data.add(sample, np.array([load_p, gen_p]))
            predicted_forecast = forecast(self.state, sample)
        return true_forecast, predicted_forecast

    def step(self, global_params, batch_size=128):
        self.state = self.state.replace(params=global_params)
        idx = self.rng.choice(len(self.data), batch_size, replace=False)
        loss, self.state = learner_step(self.state, self.data.X[idx], self.data.Y[idx])
        return loss, self.state.params

    def set_params(self, global_params):
        self.state = self.state.replace(params=global_params)


class Server:
    def __init__(
        self,
        model,
        global_params,
        clients,
        rounds,
        batch_size,
        aggregate_fn=fedavg,
        kickback_momentum=False,
        compute_cs=False,
        finetune_episodes=0,
    ):
        self.model = model
        self.global_params = global_params
        self.clients = clients
        self.rounds = rounds
        self.batch_size = batch_size
        if kickback_momentum:
            self.aggregate = KickbackMomentum(global_params, aggregate_fn=aggregate_fn)
        else:
            self.aggregate = aggregate_fn
        self.finetune_episodes = finetune_episodes
        self.compute_cs = compute_cs

    def reset(self):
        for client in self.clients:
            client.reset()

    def setup_test(self):
        for client in self.clients:
            client.set_params(self.global_params)
            for _ in range(self.finetune_episodes):
                client.step(self.global_params, self.batch_size)
            client.reset()

    def add_data(self, obs, i, transitions):
        for client in self.clients:
            client.add_data(obs, i, transitions)

    def add_test_data(self, obs):
        true_forecasts, predicted_forecasts = [], []
        for client in self.clients:
            true_forecast, predicted_forecast = client.add_test_data(obs)
            if isinstance(true_forecast, list):
                true_forecasts.extend(true_forecast)
                predicted_forecasts.extend(predicted_forecast)
            else:
                true_forecasts.append(true_forecast)
                predicted_forecasts.append(predicted_forecast)
        return np.array(true_forecasts), np.array(predicted_forecasts)

    def step(self):
        logger.info("Server is starting federated training of the forecast model")
        for _ in range(self.rounds):
            all_losses, all_params = self.inner_step()
            self.global_params = self.aggregate(all_params)
        logger.info(f"Done. FL Server Loss: {np.mean(all_losses):.5f}")
        if self.compute_cs:
            return cosine_similarity(self.global_params, all_params)

    def inner_step(self):
        all_params = []
        all_losses = []
        for client in self.clients:
            loss, params = client.step(self.global_params, self.batch_size)
            all_params.append(params)
            all_losses.append(loss)
        return all_losses, all_params

    @property
    def num_clients(self):
        amount_clients = 0
        for c in self.clients:
            if isinstance(c, MiddleServer):
                amount_clients += len(c.clients)
            else:
                amount_clients += 1
        return amount_clients


class KickbackMomentum:
    def __init__(self, global_params, mu1=0.9, mu2=0.1, aggregate_fn=fedavg):
        self.global_params = global_params
        self.momentum = None
        self.prev_parameters = None
        self.mu1 = mu1
        self.mu2 = mu2
        self.aggregate = aggregate_fn

    def __call__(self, all_params):
        if self.momentum is None:
            self.momentum = jax.tree_util.tree_map(jnp.zeros_like, self.global_params)
        else:
            self.momentum = calc_inner_momentum(self.momentum, self.global_params, self.prev_parameters, self.mu1)
        self.prev_parameters = self.global_params.copy()
        grads = self.aggregate([tree_sub(cp, self.global_params) for cp in all_params])
        self.global_params = calc_outer_momentum(self.momentum, grads, self.global_params, self.mu2)
        return self.global_params


@jax.jit
def calc_inner_momentum(momentum, params, prev_params, mu):
    return jax.tree_util.tree_map(lambda m, p, pp: mu * m + (p - pp), momentum, params, prev_params)


@jax.jit
def calc_outer_momentum(momentum, grads, prev_params, mu):
    return jax.tree_util.tree_map(lambda m, g, pp: pp + mu * m + g, momentum, grads, prev_params)


@jax.jit
def tree_sub(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a - b, tree_a, tree_b)


def cosine_similarity(global_params, client_parameters):
    client_grads = [jax.flatten_util.ravel_pytree(tree_sub(global_params, cp))[0] for cp in client_parameters]
    similarity_matrix = np.abs(metrics.pairwise.cosine_similarity(client_grads)) - np.eye(len(client_grads))
    return similarity_matrix.sum() / (len(client_grads) * (len(client_grads) - 1))


class MiddleServer:
    def __init__(self, global_params, clients, aggregate_fn=fedavg, kickback_momentum=False):
        self.clients = clients
        if kickback_momentum:
            self.aggregate = KickbackMomentum(global_params, aggregate_fn=aggregate_fn)
        else:
            self.aggregate = aggregate_fn

    def reset(self):
        for client in self.clients:
            client.reset()

    def set_params(self, global_params):
        for client in self.clients:
            client.set_params(global_params)

    def add_data(self, obs, i, transitions):
        for client in self.clients:
            client.add_data(obs, i, transitions)

    def add_test_data(self, obs):
        true_forecasts, predicted_forecasts = [], []
        for client in self.clients:
            true_forecast, predicted_forecast = client.add_test_data(obs)
            true_forecasts.append(true_forecast)
            predicted_forecasts.append(predicted_forecast)
        return true_forecasts, predicted_forecasts

    def step(self, global_params, batch_size):
        all_params = []
        all_losses = []
        for client in self.clients:
            loss, params = client.step(global_params, batch_size)
            all_params.append(params)
            all_losses.append(loss)
        new_params = self.aggregate(all_params)
        new_loss = np.mean(all_losses)
        return new_loss, new_params


@jax.jit
def forecast(state, sample):
    return state.apply_fn(state.params, sample)


def reset_clients(clients):
    if clients is None:
        return
    for c in clients:
        c.reset()