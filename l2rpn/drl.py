from __future__ import annotations
import functools
from typing import NamedTuple, Sequence, Tuple
import math
import numpy as np
import jax
import jax.numpy as jnp
import chex
import flax.linen as nn
from flax.training import train_state
import optax
import distrax


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


def reinforcement_learning(rl_state, transitions, rl_steps, rl_batch_size):
    for i in range(rl_steps):
        trans_batch = transitions.sample(rl_batch_size)
        loss, rl_state = rl_learner_step(rl_state, trans_batch)
    return loss, rl_state


def setup(env, obs, num_clients, seed):
    rl_model = ActorCritic(env.action_space.n)
    rl_state = train_state.TrainState.create(
        apply_fn=rl_model.apply,
        params=rl_model.init(jax.random.PRNGKey(seed), jnp.concatenate((obs.to_vect(), jnp.zeros(2 * num_clients)))),
        # We use AMSGrad instead of Adam due to greater stability in noise https://arxiv.org/abs/1904.09237
        tx=optax.amsgrad(1e-4),
    )
    return rl_state
