from functools import partial
import flax.linen as nn
import numpy as np
import scipy.optimize as sp_opt
import einops


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


class CNN(nn.Module):
    "Neural network for MNIST prediction"

    @nn.compact
    def __call__(self, x):
        x = einops.reduce(x, 'b h w c -> b (h w c)')
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return nn.softmax(x)


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

    def __call__(self, X):
        preds = []
        for i in range(self.parameters.shape[0]):
            preds.append(X.dot(self.parameters[i]))
        return np.stack(preds, axis=-1)

    def func(X, Y, alpha, w):
        residual = X.dot(w) - Y
        f = 0.5 * residual.dot(residual) + 0.5 * alpha * w.dot(w)
        grad = X.T @ residual + alpha * w
        return f, grad
