from typing import Callable, Dict, Iterable, Self
from numpy.typing import NDArray
from itertools import product
from functools import partial
import datasets
import numpy as np
import einops
import pandas as pd
import pickle
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import flagon
import flax_lightning


def get_data():
    with open("data/solar_home_data.pkl", 'rb') as f:
        data = pickle.load(f)
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    for customer_data in data.values():
        idx = np.arange(24, len(customer_data))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        X_train.append(customer_data[expanded_idx][:300 * 24])
        Y_train.append(customer_data[idx, 0][:300 * 24])
        X_test.append(customer_data[expanded_idx][300 * 24:])
        Y_test.append(customer_data[idx, 0][300 * 24:])
    return np.concatenate(X_train), np.concatenate(Y_train), np.concatenate(X_test), np.concatenate(Y_test)


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3,))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3,))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3,))(x)
        x = nn.relu(x)
        x = einops.rearrange(x, "b t s -> b (t s)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)


def create_model(lr=0.001, opt=optax.sgd, loss="mean_absolute_error", seed=None):
    model = Net()
    params = model.init(jax.random.PRNGKey(seed if seed else 42), jnp.zeros((1, 23, 4)))
    return flax_lightning.Model(
        model,
        params,
        opt(lr),
        loss,
        metrics=["mean_absolute_error", "r2score"],
        seed=seed
    )


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()
    processor = skp.MinMaxScaler().fit(Y_train.reshape(-1, 1))
    Y_train = processor.transform(Y_train.reshape(-1, 1)).reshape(-1)
    Y_test = processor.transform(Y_test.reshape(-1, 1)).reshape(-1)
    # X normalization
    processor = skp.StandardScaler().fit(X_train.reshape(-1, 23 * 4))
    X_train = processor.transform(X_train.reshape(-1, 23 * 4)).reshape(-1, 23, 4)
    X_test = processor.transform(X_test.reshape(-1, 23 * 4)).reshape(-1, 23, 4)

    optimizers = {"sgd": optax.sgd, "momentum": partial(optax.sgd, momentum=0.9), "adam": optax.adam}
    df = pd.DataFrame()

    for lr, opt_name, loss in product((0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001), optimizers.keys(), ("mean_absolute_error", "l2_loss", "log_cosh_loss")):
        print(f"{lr=}, {opt_name=}, {loss=}")
        opt = optimizers[opt_name]
        model = create_model(lr=lr, opt=opt, loss=loss)
        model.step(X_train, Y_train, 1, verbose=1)
        results = model.evaluate(X_test, Y_test, verbose=0)
        print(results)
        df = pd.concat((df, pd.DataFrame({"lr": [lr], "optimizer": [opt_name], "loss": [loss], "r2 score": [results['r2score'].item()]})))

    df.to_csv("results/grid_search.csv", index=False)


"""
X norm
        lr optimizer                 loss  r2 score
33  0.0050      adam  mean_absolute_error  0.289264
30  0.0050  momentum  mean_absolute_error  0.294649
5   0.1000  momentum        log_cosh_loss  0.296722
21  0.0100  momentum  mean_absolute_error  0.298170
60  0.0001      adam  mean_absolute_error  0.306792
53  0.0005      adam        log_cosh_loss  0.306975
43  0.0010      adam              l2_loss  0.308029
4   0.1000  momentum              l2_loss  0.308470
62  0.0001      adam        log_cosh_loss  0.311509
52  0.0005      adam              l2_loss  0.315859

No X norm
        lr optimizer                 loss  r2 score
9   0.0500       sgd  mean_absolute_error  0.244854
53  0.0005      adam        log_cosh_loss  0.245883
3   0.1000  momentum  mean_absolute_error  0.246716
51  0.0005      adam  mean_absolute_error  0.267268
60  0.0001      adam  mean_absolute_error  0.274334
62  0.0001      adam        log_cosh_loss  0.275485
52  0.0005      adam              l2_loss  0.281419
44  0.0010      adam        log_cosh_loss  0.288516
61  0.0001      adam              l2_loss  0.289859
42  0.0010      adam  mean_absolute_error  0.294919
"""