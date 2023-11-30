from __future__ import annotations
from typing import NamedTuple, Sequence, Tuple
from functools import partial
import argparse
import collections
import os
import itertools
import logging
import math
import functools
import time
import grid2op
from grid2op import Converter
from lightsim2grid import LightSimBackend
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn
from flax.training import train_state
import chex
import optax
import distrax
import clax
from sklearn import metrics
from tqdm import tqdm, trange

logger = logging.getLogger("L2RPN experiment")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('| %(name)s %(levelname)s @ %(asctime)s in %(filename)s:%(lineno)d | %(message)s'))
ch.setStream(tqdm)
ch.terminator = ""
logger.addHandler(ch)


# Federated Learning

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
    def __init__(self, client_id, model, load_id, gen_id, num_episodes, forecast_window, buffer_size: int = 1000):
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
        idx = rng.choice(len(self.data), batch_size, replace=False)
        loss, self.state = forecast_learner_step(self.state, self.data.X[idx], self.data.Y[idx])
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


def np_indexof(arr, val):
    index = np.where(arr == val)[0]
    if index.size > 0:
        return index
    return None


@jax.jit
def forecast(state, sample):
    return state.apply_fn(state.params, sample)


# FL Adversaries

class EmptyUpdater(Client):
    def step(self, global_params, batch_size=128):
        self.state = self.state.replace(params=global_params)
        idx = rng.choice(len(self.data), batch_size, replace=False)
        loss, _ = forecast_learner_step(self.state, self.data.X[idx], self.data.Y[idx])
        return loss, global_params


class Adversary(Client):
    def __init__(
        self, client_id, model, load_id, gen_id, num_episodes, forecast_window, corroborator, buffer_size: int = 1000
    ):
        super().__init__(client_id, model, load_id, gen_id, num_episodes, forecast_window, buffer_size)
        self.corroborator = corroborator
        self.corroborator.register(self)

    def honest_step(self, global_params, batch_size=128):
        return super().step(global_params, batch_size)


class LIE(Adversary):
    def step(self, global_params, batch_size=128):
        mu, sigma, loss = self.corroborator.calc_grad_stats(global_params, self.id, batch_size)
        return loss, lie(mu, sigma, self.corroborator.z_max)


@jax.jit
def lie(mu, sigma, z_max):
    return jax.tree_util.tree_map(lambda m, s: m + z_max * s, mu, sigma)


class IPM(Adversary):
    def step(self, global_params, batch_size=128):
        mu, sigma, loss = self.corroborator.calc_grad_stats(global_params, self.id, batch_size)
        return loss, ipm(global_params, self.corroborator.nadversaries)


@jax.jit
def ipm(params, mu, nadversaries):
    grads = jax.tree_util.tree_map(lambda p, m: p - m, params, mu)
    return jax.tree_util.tree_map(lambda p, g: p + (1 / nadversaries) * g, params, grads)


class Corroborator:
    def __init__(self, nclients, nadversaries):
        self.nclients = nclients
        self.adversaries = []
        self.nadversaries = nadversaries
        self.mu = None
        self.sigma = None
        self.loss = None
        self.parameters = None
        s = self.nclients // 2 + 1 - self.nadversaries
        self.z_max = jsp.stats.norm.ppf((self.nclients - s) / self.nclients)
        self.adv_ids = []
        self.updated_advs = []

    def register(self, adversary):
        self.adversaries.append(adversary)
        self.adv_ids.append(adversary.id)

    def calc_grad_stats(self, global_params, adv_id, batch_size):
        if self.updated_advs:
            self.updated_advs.append(adv_id)
            if set(self.updated_advs) == set(self.adv_ids):  # if everyone has updated, stop using the cached value
                self.updated_advs = []
            return self.mu, self.sigma, self.loss

        honest_parameters = []
        honest_losses = []
        for a in self.adversaries:
            loss, parameters = a.honest_step(global_params, batch_size)
            honest_parameters.append(parameters)
            honest_losses.append(loss)

        # Does some aggregation
        self.mu = fedavg(honest_parameters)
        self.sigma = tree_std(honest_parameters, self.mu)
        self.loss = np.average(honest_losses)
        self.updated_advs.append(adv_id)
        return self.mu, self.sigma, self.loss


def tree_std(trees, tree_mean):
    diffs = [jax.tree_util.tree_map(lambda p, m: (p - m)**2, tree, tree_mean) for tree in trees]
    var = jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *diffs)
    std = jax.tree_util.tree_map(lambda x: jnp.sqrt(x), var)
    return std


# Deep reinforcement learning

class ActorCritic(nn.Module):
    "An Actor Critic neural network model."
    n_actions: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        actor_mean = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.n_actions, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.n_actions,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(x)
        critic = nn.relu(critic)
        critic = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.constant(0.0)
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class TransitionBatch(NamedTuple):
    "Class to store the current batch data, and produce minibatchs from it."
    obs: chex.Array
    client_forecasts: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array
    rng: np.random.Generator

    def init(
            num_timesteps: int,
            num_actors: int,
            num_clients: int,
            obs_shape: Sequence[int],
            act_shape: Sequence[int],
            seed: int = 0,
    ) -> TransitionBatch:
        n = num_timesteps * num_actors
        return TransitionBatch(
            np.zeros((n,) + obs_shape, dtype=np.float32),
            np.zeros((n, num_clients, 2), dtype=np.float32),
            np.zeros((n,) + act_shape, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.random.default_rng(seed),
        )

    def sample(self, batch_size: int = 128) -> TransitionMinibatch:
        idx_start = self.rng.choice(self.obs.shape[0] - batch_size)
        idx = np.arange(idx_start, idx_start + batch_size)
        return TransitionMinibatch(
            self.obs[idx],
            self.client_forecasts[idx],
            self.actions[idx],
            self.rewards[idx],
            self.values[idx],
            self.log_probs[idx],
            self.dones[idx],
        )


class TransitionMinibatch(NamedTuple):
    "Class to store a minibatch of the transition data."
    obs: chex.Array
    client_forecasts: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array


@jax.jit
def rl_learner_step(
    state: train_state.TrainState,
    transitions: TransitionBatch,
    gamma: float = 0.99,
    lamb: float = 0.95,
    eps: float = 0.2,
    coef1: float = 1.0,
    coef2: float = 0.01,
) -> Tuple[float, train_state.TrainState]:
    """
    This is the last two lines of Algorithm 1 in http://arxiv.org/abs/1707.06347, also including the loss function
    calculation.
    """
    # Calculate advantage
    samples = jnp.concatenate((transitions.obs, transitions.client_forecasts.reshape(transitions.obs.shape[0], -1)), axis=-1)
    _, last_val = state.apply_fn(state.params, samples[-1])

    def calc_advantages(last_advantage, done_and_delta):
        done, delta = done_and_delta
        advantage = delta + gamma * lamb * done * last_advantage
        return advantage, advantage

    next_values = jnp.concatenate((transitions.values[1:], last_val.reshape(1)), axis=0)
    deltas = transitions.rewards + gamma * next_values * transitions.dones - transitions.values
    _, advantages = jax.lax.scan(calc_advantages, 0.0, (transitions.dones, deltas))

    def loss_fn(params):
        pis, values = jax.vmap(functools.partial(state.apply_fn, params))(samples)
        log_probs = jax.vmap(lambda pi, a: pi.log_prob(a))(pis, transitions.actions)
        # Value loss
        targets = advantages + transitions.values
        value_losses = jnp.mean((values - targets)**2)
        # Actor loss
        ratio = jnp.exp(log_probs - transitions.log_probs)
        norm_advantages = (advantages - advantages.mean(-1)) / (advantages.std(-1) + 1e-8)
        actor_loss1 = (ratio.T * norm_advantages).T
        actor_loss2 = (jnp.clip(ratio, 1 - eps, 1 + eps).T * norm_advantages).T
        actor_loss = jnp.mean(jnp.minimum(actor_loss1, actor_loss2))
        # Entropy loss
        entropy = jax.vmap(lambda pi: pi.entropy())(pis).mean()
        # Then the full loss
        loss = actor_loss - coef1 * value_losses + coef2 * entropy
        return -loss  # Flip the sign to maximise the loss

    # With that we can calculate a standard gradient descent update
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


# Experiment


def create_rl_model(env, obs, num_clients, seed):
    rl_model = ActorCritic(env.action_space.n)
    rl_state = train_state.TrainState.create(
        apply_fn=rl_model.apply,
        params=rl_model.init(jax.random.PRNGKey(seed), jnp.concatenate((obs.to_vect(), jnp.zeros(2 * num_clients)))),
        # We use AMSGrad instead of Adam due to greater stability in noise https://arxiv.org/abs/1904.09237
        tx=optax.amsgrad(1e-4),
    )
    return rl_state


def setup_fl(
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
    forecast_model = ForecastNet()
    global_params = forecast_model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 2 * forecast_window + 2)))
    substation_ids = set(env.load_to_subid) | set(env.gen_to_subid)

    if attack == "":
        adversary_type = Client
    elif attack == "empty":
        adversary_type = EmptyUpdater
    else:
        corroborator = Corroborator(len(substation_ids), round(len(substation_ids) * (1 - 0.5)))
        if attack == "lie":
            adversary_type = partial(LIE, corroborator=corroborator)
        elif attack == "ipm":
            adversary_type = partial(IPM, corroborator=corroborator)
    clients = [
        (adversary_type if i + 1 > (len(substation_ids) * 0.5) else Client)(
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
            MiddleServer(
                global_params,
                [lower_clients[i] for i in cids],
                aggregate_fn=globals()[middle_server_aggregator],
                kickback_momentum=middle_server_km,
            )
            for cids in ms_cids
        ]
        clients = middle_servers  # Middle servers are the clients for the top level server
    server = Server(
        forecast_model,
        global_params,
        clients,
        fl_rounds,
        fl_batch_size,
        kickback_momentum=server_km,
        compute_cs=compute_cs,
        finetune_episodes=intermediate_finetuning,
        aggregate_fn=globals()[server_aggregator],
    )
    return server


def reset_clients(clients):
    if clients is None:
        return
    for c in clients:
        c.reset()


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


def reinforcement_learning(rl_state, transitions, rl_steps, rl_batch_size):
    for i in range(rl_steps):
        trans_batch = transitions.sample(rl_batch_size)
        loss, rl_state = rl_learner_step(rl_state, trans_batch)
    return loss, rl_state


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
        server = setup_fl(
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
    rl_state = create_rl_model(env, obs, num_clients, args.seed)

    logger.info("Generating data with simulations of the grid and training the models")
    for e in (pbar := trange(args.episodes)):
        # We generate all of the random generation keys that we will need pre-emptively
        rngkeys = jax.random.split(rngkey, args.actors * args.timesteps + 1)
        rngkey = rngkeys[0]
        # Allocate the memory for our data batch and the index where each sample is stored
        transitions = TransitionBatch.init(
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
        loss, rl_state = reinforcement_learning(rl_state, transitions, args.rl_steps, args.rl_batch_size)
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
