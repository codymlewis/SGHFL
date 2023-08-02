from typing import List, Any
import datasets
import numpy as np
import einops
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jaxopt

import flagon
import ntmg


PyTree = Any


def load_mnist() -> ntmg.Dataset:
    """
    Load the Fashion MNIST dataset http://arxiv.org/abs/1708.07747

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("fashion_mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data = {t: {'X': ds[t]['X'], 'Y': ds[t]['Y']} for t in ['train', 'test']}
    dataset = ntmg.Dataset(data)
    return dataset


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return nn.softmax(x)


def loss(model):
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def accuracy(model):
    def _apply(params, X, Y):
        preds = jnp.argmax(model.apply(params, X), axis=-1)
        return jnp.mean(preds == Y)
    return _apply


class Metrics:
    def __init__(self, model, metrics):
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


class ModelState:
    def __init__(self, model, params, opt, rng=np.random.default_rng()):
        self.model = model
        self.params = params
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=loss(model), maxiter=3000)
        self.state = self.solver.init_state(params)
        self.solver_step = jax.jit(self.solver.update)
        self.rng = rng
        self.metrics = Metrics(model, [accuracy, loss])

    def set_parameters(self, params_leaves):
        self.params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(self.params), params_leaves)

    def get_parameters(self):
        return jax.tree_util.tree_leaves(self.params)
    
    def step(self, X, Y, epochs, steps_per_epoch=None, verbose=0):
        indices = np.arange(len(Y))
        self.rng.shuffle(indices)
        idx = indices[:len(indices) - (len(indices) % 32)].reshape((-1, 32))
        for ix in (pbar := tqdm(idx)):
            self.params, self.state = self.solver_step(params=self.params, state=self.state, X=X[ix], Y=Y[ix])
            pbar.set_postfix_str(f"LOSS: {self.state.value:.3f}")
        if len(indices) % 32:
            ix = indices[-len(indices) % 32:]
            self.params, self.state = self.solver_step(params=self.params, state=self.state, X=X[ix], Y=Y[ix])
        return {"loss": self.state.value}
    
    def evaluate(self, X, Y, verbose=0):
        indices = np.arange(len(Y))
        idx = indices[:len(indices) - (len(indices) % 32)].reshape((-1, 32))
        for ix in (pbar := tqdm(idx)):
            self.metrics.add_batch(self.params, X[ix], Y[ix])
        if len(indices) % 32:
            ix = indices[-len(indices) % 32:]
            self.metrics.add_batch(self.params, X[ix], Y[ix])
        return self.metrics.compute()


def create_model(rng=np.random.default_rng()):
    model = Net()
    params = model.init(jax.random.PRNGKey(42), jnp.zeros((1, 28, 28, 1)))
    return ModelState(model, params, optax.sgd(0.1), rng=rng)


class Client(flagon.Client):
    def __init__(self, data, create_model_fn, rng):
        self.data = data
        self.model = create_model_fn(rng)

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        history = self.model.step(self.data['train']['X'], self.data['train']['Y'], epochs=config['num_epochs'], steps_per_epoch=config.get("num_steps"), verbose=0)
        return self.model.get_parameters(), len(self.data['train']), history

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        return len(self.data['test']), self.model.evaluate(self.data['test']['X'], self.data['test']['Y'], verbose=0)


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


def create_clients(data, create_model_fn, network_arch, seed=None):
    Y = data['train']['Y']
    rng = np.random.default_rng(seed)
    idx = iter(lda(Y, flagon.common.count_clients(network_arch), rng, alpha=1000))

    def create_client(client_id: str) -> Client:
        return Client(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, rng)
    return create_client

if __name__ == "__main__":
    data = load_mnist()
    seed = 42
    config = {"num_rounds": 10, "num_episodes": 1, "num_epochs": 1, "num_clients": 10}
    server = flagon.Server(create_model().get_parameters(), config)
    network_arch = {"clients": config['num_clients']}
    history = flagon.start_simulation(
        server,
        create_clients(data, create_model, network_arch, seed=seed),
        network_arch
    )