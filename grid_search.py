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
from src import common


def get_data():
    with open("data/solar_home_data.pkl", 'rb') as f:
        data = pickle.load(f)
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    for customer_data in data.values():
        idx = np.arange(24, len(customer_data))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        X_train.append(customer_data[expanded_idx][:300 * 24])
        Y_train.append(customer_data[idx, :2][:300 * 24])
        X_test.append(customer_data[expanded_idx][300 * 24:])
        Y_test.append(customer_data[idx, :2][300 * 24:])
    return np.concatenate(X_train), np.concatenate(Y_train), np.concatenate(X_test), np.concatenate(Y_test)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()

    processor = skp.MinMaxScaler().fit(Y_train)
    Y_train = processor.transform(Y_train)
    Y_test = processor.transform(Y_test)
    # X normalization
    processor = skp.StandardScaler().fit(X_train.reshape(-1, 23 * 5))
    X_train = processor.transform(X_train.reshape(-1, 23 * 5)).reshape(-1, 23, 5)
    X_test = processor.transform(X_test.reshape(-1, 23 * 5)).reshape(-1, 23, 5)

    optimizers = {"sgd": optax.sgd, "momentum": partial(optax.sgd, momentum=0.9), "adam": optax.adam}
    df = pd.DataFrame()

    for lr, opt_name, loss in product((0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001), optimizers.keys(), ("mean_absolute_error", "l2_loss", "log_cosh_loss")):
        print(f"{lr=}, {opt_name=}, {loss=}")
        opt = optimizers[opt_name]
        model = common.create_solar_home_model(lr=lr, opt=opt, loss=loss)
        model.step(X_train, Y_train, 1, batch_size=128, verbose=1)
        results = model.evaluate(X_test, Y_test, verbose=0)
        print(results)
        df = pd.concat((df, pd.DataFrame({"lr": [lr], "optimizer": [opt_name], "loss": [loss], "r2 score": [results['r2score'].item()]})))

    df.to_csv("results/grid_search.csv", index=False)


"""
OLD X norm
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

OLD No X norm
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

New X norm
        lr optimizer                 loss  r2 score
21  0.0100  momentum  mean_absolute_error  0.367363
61  0.0001      adam              l2_loss  0.371487
42  0.0010      adam  mean_absolute_error  0.376294
52  0.0005      adam              l2_loss  0.386165
53  0.0005      adam        log_cosh_loss  0.387663
60  0.0001      adam  mean_absolute_error  0.394844
51  0.0005      adam  mean_absolute_error  0.397019
62  0.0001      adam        log_cosh_loss  0.398107
44  0.0010      adam        log_cosh_loss  0.399209
43  0.0010      adam              l2_loss  0.422508


NEW No X norm
        lr optimizer                 loss  r2 score
62  0.0001      adam        log_cosh_loss  0.243371
32  0.0050  momentum        log_cosh_loss  0.246114
51  0.0005      adam  mean_absolute_error  0.247786
34  0.0050      adam              l2_loss  0.254473
33  0.0050      adam  mean_absolute_error  0.261028
52  0.0005      adam              l2_loss  0.262483
53  0.0005      adam        log_cosh_loss  0.266770
24  0.0100      adam  mean_absolute_error  0.267581
42  0.0010      adam  mean_absolute_error  0.301085
44  0.0010      adam        log_cosh_loss  0.313766
"""