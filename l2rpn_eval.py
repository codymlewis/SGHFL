from __future__ import annotations
from typing import NamedTuple, Sequence, Tuple
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
import flax.linen as nn
from flax.training import train_state
import chex
import optax
import distrax
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
        self.forecast_window = forecast_window

    def reset(self):
        self.past_load = collections.deque([], maxlen=self.forecast_window)
        self.past_gen = collections.deque([], maxlen=self.forecast_window)


def np_indexof(arr, val):
    index = np.where(arr == val)[0]
    if index.size > 0:
        return index.item()
    return -1


@jax.jit
def forecast(state, sample):
    return state.apply_fn(state.params, sample)


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
        # We use AMSGrad instead of Adam, due to greater stability in noise https://arxiv.org/abs/1904.09237
        tx=optax.amsgrad(1e-4),
    )
    return rl_state


def setup_fl(env, num_episodes, forecast_window, seed):
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
    return global_params, clients


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


def add_fl_data(clients, obs, i, transitions, forecast_window):
    for c, client in enumerate(clients):
        load_p = obs.load_p[client.load_id] if client.load_id >= 0 else 0.0
        gen_p = obs.gen_p[client.gen_id] if client.gen_id >= 0 else 0.0
        if len(client.past_load) == forecast_window:
            sample = np.concatenate((
                np.array([obs.hour_of_day, obs.minute_of_hour]),
                np.array(client.past_load),
                np.array(client.past_gen)
            ))
            client.data.add(sample, np.array([load_p, gen_p]))
            transitions.client_forecasts[max(0, i - 1), c] = forecast(client.state, sample)
        client.past_load.append(load_p)
        client.past_gen.append(gen_p)


def add_data(train_env, converter, rl_state, clients, transitions, num_actors, num_timesteps, forecast_window, rngkeys):
    counter = itertools.count()
    for a in range(num_actors):
        last_obs = train_env.reset()
        reset_clients(clients)
        for t in range(num_timesteps):
            i = next(counter)
            obs = add_rl_data(rl_state, train_env, converter, last_obs, i, transitions, next(rngkeys))
            if clients:
                add_fl_data(clients, obs, i, transitions, forecast_window)
            last_obs = obs
            if transitions.dones[i]:
                reset_clients(clients)
                last_obs = train_env.reset()
    return rl_state


def federated_learning(global_params, clients, fl_rounds, fl_batch_size):
    if clients and len(clients[0].data) >= fl_batch_size * fl_rounds:
        logger.info("Starting federated training of the forecast model")
        for _ in range(fl_rounds):
            all_params = []
            all_losses = []
            for client in clients:
                client.state = client.state.replace(params=global_params)
                idx = rng.choice(len(client.data), fl_batch_size, replace=False)
                loss, client.state = forecast_learner_step(client.state, client.data.X[idx], client.data.Y[idx])
                all_params.append(client.state.params)
                all_losses.append(loss)
            global_params = fedavg(all_params)
        logger.info(f"Done. FL Loss: {np.mean(all_losses):.5f}")
    return global_params


def reinforcement_learning(rl_state, transitions, rl_steps, rl_batch_size):
    for i in range(rl_steps):
        trans_batch = transitions.sample(rl_batch_size)
        loss, rl_state = rl_learner_step(rl_state, trans_batch)
    return loss, rl_state


def test_fl_and_rl_model(env_name, rl_state, global_params, clients, forecast_window, rngkey):
    test_env = grid2op.make(env_name + "_test", backend=LightSimBackend())
    for client in clients:
        client.state = client.state.replace(params=global_params)
        client.reset()
    obs = test_env.reset()
    client_forecasts, true_forecasts = [], []
    for i in itertools.count():
        client_forecasts.append(np.zeros((len(clients), 2), dtype=np.float32))
        for c, client in enumerate(clients):
            load_p = obs.load_p[client.load_id] if client.load_id >= 0 else 0.0
            gen_p = obs.gen_p[client.gen_id] if client.gen_id >= 0 else 0.0
            if i >= forecast_window:
                true_forecasts.append(np.array([load_p, gen_p]))
            client.past_load.append(load_p)
            client.past_gen.append(gen_p)
            if len(client.past_load) == forecast_window:
                sample = np.concatenate((
                    np.array([obs.hour_of_day, obs.minute_of_hour]),
                    np.array(client.past_load),
                    np.array(client.past_gen)
                ))
                client_forecasts[-1][c] = forecast(client.state, sample)
        rngkey, _rngkey = jax.random.split(rngkey)
        pi, _ = rl_state.apply_fn(rl_state.params, np.concatenate((obs.to_vect(), client_forecasts[-1].reshape(-1))))
        action = pi.sample(seed=_rngkey)
        obs, reward, done, info = test_env.step(converter.convert_act(action))
        if done:
            break
        if i % 100 == 0 and i > 0:
            logger.info(f"Reached the {i}th test iteration")
    return i + 1, np.array(client_forecasts[forecast_window - 1:-1]), np.array(true_forecasts)


def test_rl_model(env_name, rl_state, rngkey):
    test_env = grid2op.make(env_name + "_test", backend=LightSimBackend())
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
    parser.add_argument("--no-fl", action="store_true", help="Specify to not use federated learning for this experiment.")
    args = parser.parse_args()

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    rngkey = jax.random.PRNGKey(args.seed)
    env_name = "rte_case14_realistic"  # Change to lrpn_idf_2023
    perform_fl = not args.no_fl

    env = grid2op.make(env_name)
    if not os.path.exists(grid2op.get_current_local_dir() + f"/{env_name}_test"):
        env.train_val_split_random(pct_val=0.0, add_for_test="test", pct_test=10.0)
    train_env = grid2op.make(env_name + "_train", backend=LightSimBackend())
    obs = train_env.reset()
    converter = Converter.ToVect(env.action_space)
    obs_shape = obs.to_vect().shape
    act_shape = (env.action_space.n,)

    if perform_fl:
        global_params, clients = setup_fl(env, args.episodes, args.forecast_window, args.seed)
        num_clients = len(clients)
    else:
        global_params, clients = None, None
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
            clients,
            transitions,
            args.actors,
            args.timesteps,
            args.forecast_window,
            iter(rngkeys[1:]),
        )

        if perform_fl:
            global_params = federated_learning(global_params, clients, args.fl_rounds, args.fl_batch_size)
        loss, rl_state = reinforcement_learning(rl_state, transitions, args.rl_steps, args.rl_batch_size)
        pbar.set_postfix_str(f"RL Loss: {loss:.5f}")

    # The testing phase
    logger.info("Testing how long the trained model can run the power network.")
    if perform_fl:
        rl_score, client_forecasts, true_forecasts = test_fl_and_rl_model(
            env_name, rl_state, global_params, clients, args.forecast_window, rngkey
        )
        client_forecasts = client_forecasts.reshape(-1, 2)
        header = "seed,rl_score,mae,rmse,r2_score"
        results = "{},{},{},{},{}".format(
            args.seed,
            rl_score,
            metrics.mean_absolute_error(true_forecasts, client_forecasts),
            math.sqrt(metrics.mean_squared_error(true_forecasts, client_forecasts)),
            metrics.r2_score(true_forecasts, client_forecasts),
        )
    else:
        rl_score = test_rl_model(env_name, rl_state, rngkey)
        header = "seed,rl_score"
        results = f"{args.seed},{rl_score}"
    print(f"Ran the network for {rl_score} time steps")
    logger.info(f"{results=}")

    # Record the results
    os.makedirs("results", exist_ok=True)
    filename = f"results/l2rpn{'_fl' if perform_fl else ''}.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(header + "\n")
    with open(filename, 'a') as f:
        f.write(results + "\n")
    print(f"Results written to {filename}")

    print(f"Experiment took {time.time() - start_time} seconds")
