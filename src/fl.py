from __future__ import annotations
import collections
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jspo
import chex
import flax.linen as nn
from flax.training import train_state
import optax
import scipy.stats as sps
import sklearn.mixture as skm
from sklearn import metrics

import data_manager
from logger import logger


class ForecastNet(nn.Module):
    "Neural network for predicting future power load and generation in L2RPN"
    classes: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(6)(x)
        x = nn.relu(x)
        x = nn.Dense(self.classes)(x)
        return x


class PowerNet(nn.Module):
    "Neural network for predicting future power load and generation in solar_home and apartment"
    classes: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.standardize(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        return x


@jax.jit
def learner_step(
    state: train_state.TrainState,
    X: chex.Array,
    Y: chex.Array,
    delta: float = 1.0,
) -> Tuple[float, train_state.TrainState]:
    def loss_fn(params):
        predictions = state.apply_fn(params, X)
        return jnp.mean(jnp.where(
                jnp.abs(predictions - Y) < delta,
                0.5 * (predictions - Y)**2,
                delta * (jnp.abs(predictions - Y) - 0.5 * delta)
            ))

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
def topk(all_params, k=0.9):
    avg_params = jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *all_params)

    def prune(x):
        K = round((1 - k) * x.size)
        return jnp.where(x >= jnp.partition(x.reshape(-1), K)[K], x, 0)

    return jax.tree_util.tree_map(prune, avg_params)


@jax.jit
def krum(all_params):
    n = len(all_params)
    clip = round(0.5 * n)
    X = jnp.array([jax.flatten_util.ravel_pytree(p)[0] for p in all_params])
    unflattener = jax.flatten_util.ravel_pytree(all_params[0])[1]
    distances = jnp.sum(X**2, axis=1)[:, None] + jnp.sum(X**2, axis=1)[None] - 2 * jnp.dot(X, X.T)
    _, scores = jax.lax.scan(lambda unused, d: (None, jnp.sum(jnp.sort(d)[1:((n - clip) - 1)])), None, distances)
    idx = jnp.argpartition(scores, n - clip)[:(n - clip)]
    return unflattener(jnp.mean(X[idx], axis=0))


@jax.jit
def trimmed_mean(all_params, c=0.5):
    reject_i = round((c / 2) * len(all_params))
    X = jnp.array([jax.flatten_util.ravel_pytree(p)[0] for p in all_params])
    unflattener = jax.flatten_util.ravel_pytree(all_params[0])[1]
    sorted_X = jnp.sort(X, axis=0)
    return unflattener(jnp.mean(sorted_X[reject_i:-reject_i], axis=0))


@jax.jit
def phocas(all_params, c=0.5):
    X = jnp.array([jax.flatten_util.ravel_pytree(p)[0] for p in all_params])
    unflattener = jax.flatten_util.ravel_pytree(all_params[0])[1]
    # First find the trimmed mean
    reject_i = round((c / 2) * len(all_params))
    sorted_X = jnp.sort(X, axis=0)
    trmean_X = jnp.mean(sorted_X[reject_i:-reject_i], axis=0)
    # Then use it for Phocas
    tm_closest_idx = jnp.argsort(jnp.linalg.norm(X - trmean_X, axis=0))[:round((1 - c) * len(all_params))]
    return unflattener(jnp.mean(X[tm_closest_idx], axis=0))


@jax.jit
def geomedian(all_params):
    X = jnp.array([jax.flatten_util.ravel_pytree(p)[0] for p in all_params])
    unflattener = jax.flatten_util.ravel_pytree(all_params[0])[1]
    return unflattener(jspo.minimize(
        lambda x: jnp.linalg.norm(X - x),
        x0=jnp.mean(X, axis=0),
        method="BFGS",
        options={"maxiter": 5}
    ).x)


@jax.jit
def ssfgm(all_params, rho: float = 0.01, gamma: float = 30):
    """
    Assumptions:
    - Attacking clients are in the minority
    - Updates are i.i.d.
    - Most updates are closer to the honest mean than the target
    """
    # Clip the updates
    X = jnp.array([jax.flatten_util.ravel_pytree(p)[0] for p in all_params])
    unflattener = jax.flatten_util.ravel_pytree(all_params[0])[1]
    X = (X.T * jnp.minimum(1, gamma / jnp.linalg.norm(X, axis=-1))).T
    c = jnp.minimum(
        1.0,
        jnp.mean(
            (jnp.sqrt(0.6) * (jnp.std(X, axis=0) + 1e-8)) /
            (jnp.abs(jnp.median(X, axis=0) - jnp.mean(X, axis=0)) + 1e-8),
        )
    )
    # Eliminate samples that are too close to eachother, leaving only one representative
    dists = jnp.sum(X**2, axis=1)[:, None] + jnp.sum(X**2, axis=1)[None] - 2 * jnp.dot(X, X.T)
    sigma = jnp.std(X)
    far_enough_idx = jnp.all((dists + (jnp.eye(X.shape[0]) * rho * sigma)) >= (rho * sigma), axis=0)
    overlap_X = jnp.sum((X.T * ~far_enough_idx).T, axis=0) / jnp.maximum(jnp.sum(~far_enough_idx), 1.0)
    X = jnp.concatenate((X, overlap_X.reshape(1, -1)))
    far_enough_idx = jnp.concatenate((far_enough_idx, jnp.array([True])))
    k = (jnp.round(jnp.sum(far_enough_idx) * c) - 1 + jnp.sum(~far_enough_idx)).astype(int)
    return unflattener(jspo.minimize(
        lambda x: jnp.sum(jnp.where(
            (
                jnp.linalg.norm(X - x, axis=1) <= jnp.sort(jnp.where(far_enough_idx, jnp.linalg.norm(X - x, axis=1), 0.0))[k]
            ) & far_enough_idx,
            jnp.linalg.norm(X - x, axis=1),
            0.0,
        )),
        x0=jnp.mean(X, axis=0),
        method="BFGS",
        options={"maxiter": 5}
    ).x)


@jax.jit
def topk_ssfgm(all_params, k: float = 0.9, rho: float = 0.01, gamma: float = 30):
    def prune(x):
        K = round((1 - k) * x.size)
        return jnp.where(x >= jnp.partition(x.reshape(-1), K)[K], x, 0)

    pruned_params = [jax.tree_util.tree_map(prune, params) for params in all_params]
    return ssfgm(pruned_params, rho, gamma)


@jax.jit
def space_sample_mean(all_params):
    X = jnp.array([jax.flatten_util.ravel_pytree(p)[0] for p in all_params])
    unflattener = jax.flatten_util.ravel_pytree(all_params[0])[1]
    # Eliminate samples that are too close to eachother, leaving only one representative
    dists = jnp.sum(X**2, axis=1)[:, None] + jnp.sum(X**2, axis=1)[None] - 2 * jnp.dot(X, X.T)
    rho = jnp.max(dists) / jnp.sqrt(X.shape[0])
    sigma = jnp.std(X)
    far_enough_idx = jnp.all((dists + (jnp.eye(X.shape[0]) * rho * sigma)) >= (rho * sigma), axis=0)
    X_sum = jnp.sum((X.T * far_enough_idx).T, axis=0) + jnp.sum((X.T * ~far_enough_idx).T, axis=0)
    X_mean = X_sum / (jnp.sum(far_enough_idx) + 1)
    return unflattener(X_mean)


class KickbackMomentum:
    def __init__(self, global_params, mu1=0.5, mu2=0.1, aggregate_fn=fedavg):
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
def calc_outer_momentum(momentum, grads, params, mu):
    return jax.tree_util.tree_map(lambda m, g, p: p + mu * m + g, momentum, grads, params)


class FedProx:
    def __init__(self, global_params, mu=0.00001, aggregate_fn=fedavg):
        self.global_params = global_params
        self.prev_parameters = None
        self.mu = mu
        self.aggregate = aggregate_fn

    def __call__(self, all_params):
        self.prev_parameters = self.global_params.copy()
        grads = self.aggregate([tree_sub(cp, self.global_params) for cp in all_params])
        self.global_params = calc_fedprox(tree_add(self.global_params, grads), grads, self.prev_parameters, self.mu)
        return self.global_params


@jax.jit
def calc_fedprox(params, grads, prev_params, mu):
    return jax.tree_util.tree_map(lambda p, g, pp: pp + g - mu * (p - pp), params, grads, prev_params)


@jax.jit
def tree_sub(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a - b, tree_a, tree_b)


@jax.jit
def tree_add(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)


def cosine_similarity(global_params, client_grads):
    client_grads = [jax.flatten_util.ravel_pytree(cg)[0] for cg in client_grads]
    similarity_matrix = np.abs(metrics.pairwise.cosine_similarity(client_grads)) - np.eye(len(client_grads))
    return similarity_matrix.sum() / (len(client_grads) * (len(client_grads) - 1))


class MRCS:
    def __init__(self, global_params, mu=0.7, aggregate_fn=fedavg):
        self.global_params = global_params
        self.momentum = None
        self.mu = mu
        self.aggregate = aggregate_fn

    def __call__(self, all_params):
        all_grads = [tree_sub(cp, self.global_params) for cp in all_params]
        p_vals = []
        for grads in all_grads:
            if self.momentum is None:
                p_vals.append(1)
            else:
                sim = cs(self.momentum, grads)
                p_vals.append(max(0, sim))
        p_vals = jnp.array(p_vals)
        if jnp.sum(p_vals) == 0:
            p_vals = jnp.ones_like(p_vals)
        p_vals = p_vals / jnp.sum(p_vals)
        agg_grads = tree_average(all_grads, p_vals)
        if self.momentum is None:
            self.momentum = jax.tree_util.tree_map(jnp.zeros_like, self.global_params)
        self.momentum = polyak_average(self.momentum, agg_grads, self.mu)
        self.global_params = tree_add(self.global_params, self.momentum)
        return self.global_params


@jax.jit
def cs(A, B):
    A = jax.flatten_util.ravel_pytree(A)[0]
    B = jax.flatten_util.ravel_pytree(B)[0]
    return jnp.sum(A * B) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))


@jax.jit
def tree_average(trees, weights):
    return jax.tree_util.tree_map(lambda *x: jnp.sum((weights * jnp.array(x).T).T, axis=0), *trees)


@jax.jit
def polyak_average(old_value, new_value, mu):
    return jax.tree_util.tree_map(lambda o, n: (1 - mu) * o + mu * n, old_value, new_value)


def get_aggregator(aggregator, params=None):
    match aggregator:
        case "fedavg":
            return fedavg
        case "median":
            return median
        case "topk":
            return topk
        case "krum":
            return krum
        case "trimmed_mean":
            return trimmed_mean
        case "phocas":
            return phocas
        case "geomedian":
            return geomedian
        case "kickback_momentum":
            return KickbackMomentum(params)
        case "fedprox":
            return FedProx(params)
        case "mrcs":
            return MRCS(params)
        case "ssfgm":
            return ssfgm
        case "topk_ssfgm":
            return topk_ssfgm
        case "space_sample_mean":
            return space_sample_mean
        case "duttagupta":
            return fedavg
        case "li":
            return fedavg
        case "lissfgm":
            return ssfgm
        case _:
            raise NotImplementedError(f"{aggregator} not implemented")


class Client:
    def __init__(self, client_id, model, info=None, data=None, seed=0):
        assert (info is None) or (data is None), "Either info or data needs to be provided"
        if info is not None:
            self.past_load = collections.deque([], maxlen=info['forecast_window'])
            self.past_gen = collections.deque([], maxlen=info['forecast_window'])
            buffer_size = info.get("buffer_size") if info.get("buffer_size") is not None else 1000
            self.data = data_manager.Dataset.online(info['num_episodes'] * buffer_size, info['forecast_window'])
            self.load_id = info['load_id']
            self.gen_id = info['gen_id']
            self.forecast_window = info['forecast_window']
        if data is not None:
            self.data = data
        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.PRNGKey(0), self.data.X[:1]),
            tx=optax.adam(0.01),
        )
        self.id = client_id
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.past_load = collections.deque([], maxlen=self.forecast_window)
        self.past_gen = collections.deque([], maxlen=self.forecast_window)

    def add_data(self, obs_load_p, obs_gen_p, obs_time):
        load_p = obs_load_p[self.load_id].sum() if self.load_id is not None else 0.0
        gen_p = obs_gen_p[self.gen_id].sum() if self.gen_id is not None else 0.0
        if len(self.past_load) == self.forecast_window:
            sample = np.concatenate((
                obs_time,
                np.array(self.past_load),
                np.array(self.past_gen),
            ))
            self.data.add(sample, np.array([load_p, gen_p]))
        self.past_load.append(load_p)
        self.past_gen.append(gen_p)

    def add_test_data(self, obs_load_p, obs_gen_p, obs_time):
        true_forecast, predicted_forecast = np.zeros(2), np.zeros(2)
        load_p = obs_load_p[self.load_id].sum() if self.load_id is not None else 0.0
        gen_p = obs_gen_p[self.gen_id].sum() if self.gen_id is not None else 0.0
        if len(self.past_load) == self.forecast_window:
            true_forecast = np.array([load_p, gen_p])
        self.past_load.append(load_p)
        self.past_gen.append(gen_p)
        if len(self.past_load) == self.forecast_window:
            sample = np.concatenate((
                obs_time,
                np.array(self.past_load),
                np.array(self.past_gen),
            ))
            self.data.add(sample, np.array([load_p, gen_p]))
            predicted_forecast = forecast(self.state, sample)
        return true_forecast, predicted_forecast

    def step(self, global_params, batch_size=128, steps=1):
        self.state = self.state.replace(params=global_params)
        for _ in range(steps):
            idx = self.rng.choice(len(self.data), batch_size, replace=False)
            loss, self.state = learner_step(self.state, self.data.X[idx], self.data.Y[idx])
        return loss, self.state.params

    def li_step(self, global_params, idx, batch_size=128, steps=1):
        self.state = self.state.replace(params=global_params)
        for _ in range(steps):
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
        batch_size,
        aggregator="fedavg",
    ):
        self.model = model
        self.global_params = global_params
        self.clients = clients
        self.all_clients = clients.copy()  # To maintain track of clients after dropping
        self.batch_size = batch_size
        self.aggregate = get_aggregator(aggregator, global_params)

    def reset(self):
        for client in self.clients:
            client.reset()

    def setup_test(self, finetune_steps=0):
        client_list = self.clients if not hasattr(self, "all_clients") else self.all_clients
        for client in client_list:
            client.set_params(self.global_params)
            for _ in range(finetune_steps):
                client.step(self.global_params, self.batch_size)
            client.reset()

    def add_data(self, obs_load_p, obs_gen_p, obs_time):
        for client in self.clients:
            client.add_data(obs_load_p, obs_gen_p, obs_time)

    def add_test_data(self, obs_load_p, obs_gen_p, obs_time):
        true_forecasts, predicted_forecasts = [], []
        for client in self.clients:
            true_forecast, predicted_forecast = client.add_test_data(obs_load_p, obs_gen_p, obs_time)
            if isinstance(true_forecast, list):
                true_forecasts.extend(true_forecast)
                predicted_forecasts.extend(predicted_forecast)
            else:
                true_forecasts.append(true_forecast)
                predicted_forecasts.append(predicted_forecast)
        true_forecasts, predicted_forecasts = np.array(true_forecasts), np.array(predicted_forecasts)
        # Evaluation of the dropped client's performance
        ndropped_clients = len(self.all_clients) - len(self.clients)
        d_true_forecasts, d_predicted_forecasts = [], []
        if ndropped_clients > 0:
            for client in self.all_clients[-ndropped_clients:]:
                d_true_forecast, d_predicted_forecast = client.add_test_data(obs_load_p, obs_gen_p, obs_time)
                if isinstance(d_true_forecast, list):
                    d_true_forecasts.extend(d_true_forecast)
                    d_predicted_forecasts.extend(d_predicted_forecast)
                else:
                    d_true_forecasts.append(d_true_forecast)
                    d_predicted_forecasts.append(d_predicted_forecast)
        d_true_forecasts, d_predicted_forecasts = np.array(d_true_forecasts), np.array(d_predicted_forecasts)
        return true_forecasts, predicted_forecasts, d_true_forecasts, d_predicted_forecasts

    def step(self, compute_cs=False, client_steps=1):
        all_losses, all_grads = self.inner_step(client_steps)
        self.global_params = tree_add(self.global_params, self.aggregate(all_grads))
        cs = cosine_similarity(self.global_params, all_grads) if compute_cs else -3.0
        return cs, all_losses

    def inner_step(self, client_steps=1):
        all_grads = []
        all_losses = []
        for client in self.clients:
            loss, params = client.step(self.global_params, self.batch_size, client_steps)
            grads = tree_sub(params, self.global_params)
            all_grads.append(grads)
            all_losses.append(loss)
        return all_losses, all_grads

    def finetune(self, finetune_steps):
        client_list = self.clients if not hasattr(self, "all_clients") else self.all_clients
        for client in client_list:
            client.set_params(self.global_params)
            for _ in range(finetune_steps):
                client.step(self.global_params, self.batch_size)

    def drop_clients(self):
        logger.info("Dropping clients")
        nclients = len(self.clients)
        for _ in range(round(nclients * 0.4)):
            self.clients.pop()

    @property
    def num_clients(self):
        amount_clients = 0
        for c in self.clients:
            if isinstance(c, MiddleServer):
                amount_clients += len(c.clients)
            else:
                amount_clients += 1
        return amount_clients


class MiddleServer:
    def __init__(self, global_params, clients, aggregator="fedavg"):
        self.clients = clients
        if aggregator == "li":
            self.alpha = np.ones(len(clients))
            self.beta = np.ones_like(self.alpha)
            for i, client in enumerate(clients):
                if len(client.data) > 0:
                    uniform_distribution = np.linspace(
                        np.min(client.data.Y, axis=0),
                        np.max(client.data.Y, axis=0),
                        num=len(client.data),
                    )
                    self.beta[i] = np.mean([
                        np.maximum(sps.wasserstein_distance(client.data.Y[:, i], uniform_distribution[:, i]), 1e-15)
                        for i in range(client.data.Y.shape[-1])
                    ])
            self.rng = np.random.default_rng(42)
            self.K = min(3, len(clients))
            self.step = self.li_step
        elif aggregator == "lissfgm":
            self.rng = np.random.default_rng(42)
            self.step = self.lissfgm_step
        else:
            self.step = self.normal_step
        self.aggregate = get_aggregator(aggregator, global_params)

    def reset(self):
        for client in self.clients:
            client.reset()

    def set_params(self, global_params):
        for client in self.clients:
            client.set_params(global_params)

    def add_data(self, obs_load_p, obs_gen_p, obs_time):
        for client in self.clients:
            client.add_data(obs_load_p, obs_gen_p, obs_time)

    def add_test_data(self, obs_load_p, obs_gen_p, obs_time):
        true_forecasts, predicted_forecasts = [], []
        for client in self.clients:
            true_forecast, predicted_forecast = client.add_test_data(obs_load_p, obs_gen_p, obs_time)
            true_forecasts.append(true_forecast)
            predicted_forecasts.append(predicted_forecast)
        return true_forecasts, predicted_forecasts

    def normal_step(self, global_params, batch_size, steps=1):
        all_params = []
        all_losses = []
        for client in self.clients:
            loss, params = client.step(global_params, batch_size, steps=steps)
            all_params.append(params)
            all_losses.append(loss)
        new_params = self.aggregate(all_params)
        new_loss = np.mean(all_losses)
        return new_loss, new_params

    def li_step(self, global_params, batch_size, steps=1):
        "https://ieeexplore.ieee.org/document/9874660"
        # Calculate v_i and D_train
        N = 5
        client_data_idxs = []
        client_v = np.zeros_like(self.alpha)
        for c, client in enumerate(self.clients):
            if len(client.data) <= batch_size:
                client_data_idxs.append(np.arange(len(client.data)))
                continue
            entropy_vals = np.zeros(N)
            client_data_idx = np.zeros((N, batch_size), dtype=int)
            for i in range(N):
                client_data_idx[i] = self.rng.choice(len(client.data), batch_size, replace=False)
                x_i = client.data.X[client_data_idx[i]]
                y_i = np.maximum(client.data.Y[client_data_idx[i]], 1e-15)
                entropy_vals[i] = -np.mean(y_i * np.log2(y_i)) + \
                    np.mean(y_i * np.log2(np.maximum(client.state.apply_fn(global_params, x_i), 1e-15)))
            entropy_vals = entropy_vals.reshape(-1, 1)
            bgm_model = skm.BayesianGaussianMixture()
            bgm_model.fit(entropy_vals)
            scores = bgm_model.score_samples(entropy_vals)
            client_data_idxs.append(client_data_idx[np.argmax(scores)])
            client_v[c] = np.max(scores)
        self.alpha = np.maximum(self.alpha + client_v, 1e-15)
        beta_vals = self.rng.beta(self.alpha, self.beta, size=len(self.clients))
        if self.K == len(self.clients):
            selected_clients = np.arange(len(self.clients))
        else:
            selected_clients = np.argpartition(beta_vals, self.K)[:self.K]
        all_params = []
        all_losses = []
        for selected_client_idx in selected_clients:
            loss, params = self.clients[selected_client_idx].li_step(
                global_params, client_data_idxs[selected_client_idx], batch_size, steps=steps
            )
            all_params.append(params)
            all_losses.append(loss)
        new_params = self.aggregate(all_params)
        new_loss = np.mean(all_losses)
        return new_loss, new_params

    def lissfgm_step(self, global_params, batch_size, steps=1):
        N = 5
        client_data_idxs = []
        for c, client in enumerate(self.clients):
            if len(client.data) <= batch_size:
                client_data_idxs.append(np.arange(len(client.data)))
                continue
            entropy_vals = np.zeros(N)
            client_data_idx = np.zeros((N, batch_size), dtype=int)
            for i in range(N):
                client_data_idx[i] = self.rng.choice(len(client.data), batch_size, replace=False)
                x_i = client.data.X[client_data_idx[i]]
                y_i = np.maximum(client.data.Y[client_data_idx[i]], 1e-15)
                entropy_vals[i] = -np.mean(y_i * np.log2(y_i)) + \
                    np.mean(y_i * np.log2(np.maximum(client.state.apply_fn(global_params, x_i), 1e-15)))
            entropy_vals = entropy_vals.reshape(-1, 1)
            bgm_model = skm.BayesianGaussianMixture()
            bgm_model.fit(entropy_vals)
            scores = bgm_model.score_samples(entropy_vals)
            client_data_idxs.append(client_data_idx[np.argmax(scores)])
        all_params = []
        all_losses = []
        for c, client in enumerate(self.clients):
            loss, params = client.li_step(global_params, client_data_idxs[c], batch_size, steps=steps)
            all_params.append(params)
            all_losses.append(loss)
        new_params = self.aggregate(all_params)
        new_loss = np.mean(all_losses)
        return new_loss, new_params

    def forecast(self, X_test):
        predicted_forecasts = []
        for client in self.clients:
            predicted_forecasts.append(forecast(client.state, X_test))
        return predicted_forecasts


@jax.jit
def forecast(state, sample):
    return state.apply_fn(state.params, sample)


def reset_clients(clients):
    if clients is None:
        return
    for c in clients:
        c.reset()
