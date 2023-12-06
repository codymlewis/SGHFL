"""
Commonly used functions/definitions used within this library.
"""


from typing import List, Callable
from functools import partial
import math
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import optax
import flax.linen as nn
import einops
from tqdm import tqdm
import sklearn.metrics as skm
import scipy.optimize as sp_opt
from flagon.common import count_clients


from .typing import (
    Config,
    Parameters,
    History,
)

from .functions import (
    to_attribute_array,
    count_clients,
)


from .logging import logger


from .metric import Metric

from .plot import plot_network


CACHED_SOLVERS = {}


def crossentropy_loss(model):
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def l2_loss(model):
    def _apply(params, X, Y):
        return jnp.linalg.norm(model.apply(params, X) - Y)
    return _apply


def log_cosh_loss(model):
    def _apply(params, X, Y):
        return jnp.mean(jnp.log(jnp.cosh(model.apply(params, X) - Y)))
    return _apply


def ridge_loss(model, alpha=1.0):
    def _apply(params, X, Y):
        dist = jnp.mean(jnp.linalg.norm(model.apply(params, X) - Y, axis=1)**2)
        reg = jnp.sum(jnp.array([jnp.sum(p.reshape(-1)**2) for p in jax.tree_util.tree_leaves(params)]))
        return dist + alpha * reg
    return _apply


def reg_loss(model, alpha=1.0):
    def _apply(params, X, Y):
        dist = jnp.mean((Y - model.apply(params, X))**2)
        reg = jnp.sum(jnp.array([jnp.sum(p.reshape(-1)**2) for p in jax.tree_util.tree_leaves(params)])) / len(Y)
        return dist + alpha * reg
    return _apply


def mean_squared_error(model):
    def _apply(params, X, Y):
        return jnp.mean((Y - model.apply(params, X))**2)
    return _apply


def root_mean_squared_error(model):
    def _apply(params, X, Y):
        return jnp.sqrt(jnp.mean(Y - (model.apply(params, X))**2))
    return _apply


def mean_absolute_error(model):
    def _apply(params, X, Y):
        return jnp.mean(jnp.abs(Y - model.apply(params, X)))
    return _apply


def accuracy(model):
    def _apply(params, X, Y):
        preds = jnp.argmax(model.apply(params, X), axis=-1)
        return jnp.mean(preds == Y)
    return _apply


class Metrics:
    def __init__(self, model, metrics):
        metrics = [globals()[m] if isinstance(m, str) else m for m in metrics]
        self.metrics = [jax.jit(m(model)) for m in metrics]
        self.metric_names = [m.__name__ for m in metrics]
        self.batch_count = 0
        self.measurements = [0.0 for m in self.metrics]

    def add_batch(self, params, X, Y):
        for i, metric in enumerate(self.metrics):
            self.measurements[i] += metric(params, X, Y)
        self.batch_count += 1

    def compute(self):
        results = {mn: m / self.batch_count for mn, m in zip(self.metric_names, self.measurements)}
        self.measurements = [0.0 for m in self.metrics]
        self.batch_count = 0
        return results


class Model:
    def __init__(self, model, params, opt, loss_fun, metrics=[accuracy], seed=None, no_cache=False):
        loss_fun_name = loss_fun if isinstance(loss_fun, str) else loss_fun.__name__
        loss_fun = globals()[loss_fun] if isinstance(loss_fun, str) else loss_fun
        self.model = model
        self.params = params
        self.opt = opt
        self.loss_fun = loss_fun(model)
        if no_cache:
            self.solver = jaxopt.OptaxSolver(opt=opt, fun=self.loss_fun)
            self.solver_step = jax.jit(self.solver.update)
        else:
            if CACHED_SOLVERS.get(loss_fun_name) is None:
                CACHED_SOLVERS[loss_fun_name] = {}
                CACHED_SOLVERS[loss_fun_name]['solver'] = jaxopt.OptaxSolver(opt=opt, fun=self.loss_fun)
                CACHED_SOLVERS[loss_fun_name]['step'] = jax.jit(CACHED_SOLVERS[loss_fun_name]['solver'].update)
            self.solver = CACHED_SOLVERS[loss_fun_name]['solver']
            self.solver_step = CACHED_SOLVERS[loss_fun_name]['step']
        self.state = self.solver.init_state(params)
        self.rng = np.random.default_rng(seed)
        self.metrics = Metrics(model, metrics)
        self.params_tree_structure = jax.tree_util.tree_structure(self.params)

    def change_loss_fun(self, loss_fun):
        loss_fun_name = loss_fun.__name__
        self.loss_fun = loss_fun(self.model)
        if CACHED_SOLVERS.get(loss_fun_name) is None:
            CACHED_SOLVERS[loss_fun_name] = {}
            CACHED_SOLVERS[loss_fun_name]['solver'] = jaxopt.OptaxSolver(opt=self.opt, fun=self.loss_fun)
        self.solver = CACHED_SOLVERS[loss_fun_name]['solver']
        self.solver_step = CACHED_SOLVERS[loss_fun_name]['step']
        self.state = self.solver.init_state(self.params)

    def set_parameters(self, params_leaves):
        self.params = jax.tree_util.tree_unflatten(self.params_tree_structure, params_leaves)

    def get_parameters(self):
        return jax.tree_util.tree_leaves(self.params)

    def step(self, X, Y, epochs, steps_per_epoch=None, batch_size=32, verbose=0):
        for e in range(epochs):
            loss = 0.0
            idx = np.array_split(self.rng.permutation(len(Y)), math.ceil(len(Y) / batch_size))
            if steps_per_epoch:
                idx = idx[:steps_per_epoch]
            if verbose:
                idx = tqdm(idx)
            for ix in idx:
                self.params, self.state = self.solver_step(params=self.params, state=self.state, X=X[ix], Y=Y[ix])
                loss += self.state.value
                if verbose:
                    idx.set_postfix_str(f"LOSS: {self.state.value:.3f}, epoch: {e + 1}/{epochs}")
        return {"loss": loss.item() / math.ceil(len(Y) / batch_size)}

    def predict(self, X, batch_size=200):
        idxs = np.array_split(np.arange(len(X)), math.ceil(len(X) / batch_size))
        return np.concatenate([self.model.apply(self.params, X[idx]) for idx in idxs])

    def evaluate(self, X, Y, batch_size=32, verbose=0):
        indices = np.arange(len(Y))
        idx = indices[:len(indices) - (len(indices) % batch_size)].reshape((-1, batch_size))
        if verbose:
            idx = tqdm(idx)
        for ix in idx:
            self.metrics.add_batch(self.params, X[ix], Y[ix])
        if len(indices) % batch_size:
            ix = indices[-len(indices) % batch_size:]
            self.metrics.add_batch(self.params, X[ix], Y[ix])
        return self.metrics.compute()


class FMNISTNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return nn.softmax(x)


def _ridge_func(X, Y, alpha, W):
    residual = X.dot(W) - Y
    f = 0.5 * residual.dot(residual) + 0.5 * alpha * W.dot(W)
    grad = X.T @ residual + alpha * W
    return f, grad


class RidgeModel:
    def __init__(self, metrics: List[str | Callable], alpha: float = 1.0):
        self.alpha = alpha
        self.params = np.zeros((2, 115,))
        self.metrics = [getattr(skm, m) if isinstance(m, str) else m for m in metrics]
        self.metric_names = [m.__name__ for m in self.metrics]

    def change_loss_fun(self):
        pass

    def set_parameters(self, parameters):
        self.params = parameters[0]

    def get_parameters(self):
        return [self.params]

    def step(self, X, Y, epochs, **kwargs):
        X = einops.rearrange(X, 'b h s -> b (h s)')

        loss = 0
        for i in range(Y.shape[1]):
            results = sp_opt.minimize(
                partial(_ridge_func, X, Y[:, i], self.alpha),
                x0=self.params[i],
                method="L-BFGS-B",
                tol=1e-6,
                bounds=[(0, np.inf)] * X.shape[1],
                jac=True,
                options={"maxiter": epochs}
            )
            self.params[i] = results['x']
            loss += results['fun']

        return {"loss": loss / (Y.shape[1] * X.shape[1])}

    def evaluate(self, X, Y, **kwargs):
        X = einops.rearrange(X, 'b h s -> b (h s)')
        preds = []
        for i in range(self.params.shape[0]):
            preds.append(X.dot(self.params[i]))
        predictions = np.stack(preds, axis=-1)

        measurements = {}
        for metric_name, metric_fun in zip(self.metric_names, self.metrics):
            measurements[metric_name] = metric_fun(Y, predictions)
        return measurements


def regional_distribution(labels, network_arch, rng, alpha=0.5):
    nmiddleservers = len(network_arch['clients'])
    nclients = [count_clients(subnet) for subnet in network_arch['clients']]
    distribution = [[] for _ in range(sum(nclients))]
    nclasses = len(np.unique(labels))
    proportions = rng.dirichlet(np.repeat(alpha, sum(nclients)), size=nclasses)
    client_i = 0
    for i in range(nmiddleservers):
        rdist = rng.dirichlet(np.repeat(alpha, nclients[i]))
        proportions[-(i + 1)] = np.zeros_like(proportions[-(i + 1)])
        proportions[-(i + 1)][client_i:client_i + nclients[i]] = rdist
        client_i += nclients[i]

    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


def regional_test_distribution(labels, network_arch):
    nmiddleservers = len(network_arch['clients'])
    nclients = [count_clients(subnet) for subnet in network_arch['clients']]
    distribution = [[] for _ in range(sum(nclients))]
    nclasses = len(np.unique(labels))
    client_i = 0
    for i, middle_server_nclients in enumerate(nclients):
        c = nclasses - i - 1
        for j in range(middle_server_nclients):
            distribution[client_i] = distribution[client_i] + np.where(labels == c)[0].tolist()
            client_i += 1

    for i in range(len(distribution)):
        distribution[i] = distribution[i] + np.where(
                ~np.isin(labels, list(range(nclasses - 1, nclasses - nmiddleservers - 1, -1))))[0].tolist()
    return distribution


def lda(labels, nclients, rng, alpha=0.5):
    """
    Latent Dirichlet allocation defined in https://arxiv.org/abs/1909.06335
    default value from https://arxiv.org/abs/2002.06440
    Optional arguments:
    - alpha: the alpha parameter of the Dirichlet function,
    the distribution is more i.i.d. as alpha approaches infinity and less i.i.d. as alpha approaches 0
    """
    distribution = [[] for _ in range(nclients)]
    nclasses = len(np.unique(labels))
    proportions = rng.dirichlet(np.repeat(alpha, nclients), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


def create_fmnist_model(
    seed=None,
    lr=0.01,
    opt=partial(optax.sgd, momentum=0.9),
    loss="crossentropy_loss",
    metrics=["accuracy", "crossentropy_loss"]
):
    model = FMNISTNet()
    params = model.init(jax.random.PRNGKey(seed if seed else 42), jnp.zeros((1, 28, 28, 1)))
    return Model(
        model,
        params,
        opt(lr),
        loss,
        metrics=metrics,
        seed=seed
    )


def create_solar_home_model(metrics=["mean_absolute_error", "mean_squared_error"], **kwargs):
    return RidgeModel(metrics=metrics)


def get_experiment_config(all_exp_configs, exp_id):
    experiment_config = {k: v for k, v in all_exp_configs.items() if k != "experiments"}
    variables = all_exp_configs['experiments'][exp_id - 1]
    experiment_config.update(variables)
    return experiment_config
