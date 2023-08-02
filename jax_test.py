from typing import List
import time
import json
import itertools
import datasets
import numpy as np
import scipy as sp
import sklearn.metrics as skm
import sklearn.cluster as skc
import sklearn.decomposition as skd
import einops
import matplotlib.pyplot as plt
from tqdm.auto import trange
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jaxopt

import flagon
import ntmg


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


def loss_fun(model):
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


if __name__ == "__main__":
    data = load_mnist()
    model = Net()
    params = model.init(jax.random.PRNGKey(42), data['train']['X'][:1])
    solver = jaxopt.OptaxSolver(opt=optax.sgd(0.1), fun=loss_fun(model), maxiter=3000)
    state = solver.init_state(params)
    step = jax.jit(solver.update)
    for e in (pbar := trange(solver.maxiter)):
        idx = np.random.randint(0, len(data['train']), size=32)
        params, state = step(
            params=params, state=state, X=data['train']['X'][idx], Y=data['train']['Y'][idx]
        )
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
