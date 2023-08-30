from functools import partial
import pickle
import numpy as np
import einops
import sklearn.metrics as skm
import scipy.optimize as sp_opt
from tqdm import trange
import json

import src.data_manager


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.parameters = np.zeros((2, 115))

    def fit(self, X, Y, epochs=1):
        loss = 0.0
        for i in range(Y.shape[1]):
            info = sp_opt.minimize(
                partial(RidgeModel.func, X, Y[:, i], self.alpha),
                x0=self.parameters[i],
                method="L-BFGS-B",
                tol=1e-6,
                bounds=[(0, np.inf)] * X.shape[1],
                jac=True,
                options={"maxiter": epochs}
            )
            self.parameters[i] = info['x']
            loss += info['fun']
        return loss / (X.shape[1] * Y.shape[1])

    def predict(self, X):
        preds = []
        for i in range(self.parameters.shape[0]):
            preds.append(X.dot(self.parameters[i]))
        return np.stack(preds, axis=-1)

    def func(X, Y, alpha, w):
        residual = X.dot(w) - Y
        f = 0.5 * residual.dot(residual) + 0.5 * alpha * w.dot(w)
        grad = X.T @ residual + alpha * w
        return f, grad


class Server:
    def __init__(self, clients, config):
        self.model = RidgeModel()
        self.clients = clients
        self.config = config

    def fit(self):
        for r in (pbar := trange(self.config['num_rounds'])):
            loss = self.step()
            pbar.set_postfix_str(f"loss: {loss:.3f}")
        return loss

    def step(self):
        all_params, all_losses, all_samples = [], [], []
        for c in self.clients:
            params, loss, num_samples = c.step(self.model.parameters.copy(), self.config)
            all_params.append(params)
            all_losses.append(loss)
            all_samples.append(num_samples)
        all_grads = [cp - self.model.parameters for cp in all_params]
        self.model.parameters += np.average(all_grads, weights=all_samples, axis=0)
        return np.average(all_losses, weights=all_samples)

    def analytics(self):
        all_preds, all_Y_test = [], []
        for c in self.clients:
            client_preds, client_Y_test = c.analytics(self.model.parameters.copy(), self.config)
            all_preds.append(client_preds)
            all_Y_test.append(client_Y_test)
        preds = np.concatenate(all_preds)
        Y_test = np.concatenate(all_Y_test)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "r2 score": skm.r2_score(Y_test, preds),
        }

    def evaluate(self, X_test, Y_test):
        preds = self.model.predict(X_test)
        return {
            "MAE": skm.mean_absolute_error(Y_test, preds),
            "RMSE": np.sqrt(skm.mean_squared_error(Y_test, preds)),
            "r2 score": skm.r2_score(Y_test, preds),
        }    


class MiddleServer:
    def __init__(self, clients):
        self.clients = clients

    def step(self, parameters, config):
        for e in range(config['num_episodes']):
            all_params, all_losses, all_samples = [], [], []
            for c in self.clients:
                params, loss, num_samples = c.step(parameters.copy(), config)
                all_params.append(params)
                all_losses.append(loss)
                all_samples.append(num_samples)
            all_grads = [cp - parameters for cp in all_params]
            parameters += np.average(all_grads, weights=all_samples, axis=0)
        return parameters, np.average(all_losses, weights=all_samples), sum(all_samples)

    def analytics(self, parameters, config):
        if config.get("num_finetune_episodes"):
            for e in range(config["num_finetune_episodes"]):
                parameters, _, _ = self.step(parameters, config)

        all_preds, all_Y_test = [], []
        for c in self.clients:
            preds, Y_test = c.analytics(parameters.copy(), config)
            all_preds.append(preds)
            all_Y_test.append(Y_test)
        return np.concatenate(all_preds), np.concatenate(all_Y_test)


class Client:
    def __init__(self, data):
        self.data = data

    def step(self, parameters, config):
        model = RidgeModel()
        model.parameters = parameters
        loss = model.fit(self.data['train']['X'], self.data['train']['Y'], epochs=config['num_epochs'])
        return model.parameters, loss, len(self.data['train'])

    def analytics(self, parameters, config):
        model = RidgeModel()
        model.parameters = parameters
        return model.predict(self.data['test']['X']), self.data['test']['Y']


def load_data():
    with open("data/solar_home_data.pkl", 'rb') as f:
        data = pickle.load(f)

    client_data = []
    X_test, Y_test = [], []
    for i in range(1, 301):
        idx = np.arange(24, len(data[i]))
        expanded_idx = np.array([np.arange(i - 24, i - 1) for i in idx])
        client_X, client_Y = data[i][expanded_idx], data[i][idx, :2]
        client_X = einops.rearrange(client_X, 'b h s -> b (h s)')
        client_data.append(src.data_manager.Dataset({
            "train": {"X": client_X[:300 * 24], "Y": client_Y[:300 * 24]},
            "test": {"X": client_X[300 * 24:], "Y": client_Y[300 * 24:]}
        }))
        X_test.append(client_data[-1]['test']['X'])
        Y_test.append(client_data[-1]['test']['Y'])
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    return client_data, X_test, Y_test


def load_customer_regions():
    with open("data/customer_regions.json", 'r') as f:
        customer_regions = json.load(f)
    regions = [[] for _ in np.unique(list(customer_regions.values()))]
    for customer, region_i in customer_regions.items():
        regions[region_i].append(int(customer) - 1)
    return regions


if __name__ == "__main__":
    client_data, X_test, Y_test = load_data()
    regions = load_customer_regions()
    # TODO: Momentum and logging

    server = Server(
        [MiddleServer([Client(client_data[r]) for r in region]) for region in regions],
        {"num_rounds": 5, "num_episodes": 1, "num_epochs": 1}
    )
    loss = server.fit()

    print("Client models:")
    print(server.analytics())

    print("Server model:")
    print(server.evaluate(X_test, Y_test))