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

import flax_lightning

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


def create_model(seed=None):
    model = Net()
    params = model.init(jax.random.PRNGKey(42), jnp.zeros((1, 28, 28, 1)))
    return flax_lightning.Model(
        model,
        params,
        optax.sgd(0.1),
        "crossentropy_loss",
        metrics=["accuracy", "crossentropy_loss"],
        seed=seed
    )


class Client(flagon.Client):
    def __init__(self, data, create_model_fn, seed=None):
        self.data = data
        self.model = create_model_fn(seed)

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.step(
            self.data['train']['X'],
            self.data['train']['Y'],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get("num_steps"),
            verbose=config.get("verbose")
        )
        return self.model.get_parameters(), len(self.data['train']), metrics

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        return len(self.data['test']), self.model.evaluate(self.data['test']['X'], self.data['test']['Y'], verbose=config.get("verbose"))


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
        return Client(data.select({"train": next(idx), "test": np.arange(len(data['test']))}), create_model_fn, seed)
    return create_client


if __name__ == "__main__":
    data = load_mnist()
    seed = 42
    config = {"num_rounds": 5, "num_episodes": 1, "num_epochs": 1, "num_clients": 10}
    server = flagon.Server(create_model().get_parameters(), config)
    network_arch = {"clients": config['num_clients']}
    history = flagon.start_simulation(
        server,
        create_clients(data, create_model, network_arch, seed=seed),
        network_arch
    )