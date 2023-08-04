import jax
import jax.numpy as jnp
import jaxopt
import numpy as np


def crossentropy_loss(model):
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
    def __init__(self, model, params, opt, loss_fun, metrics=[accuracy], seed=None):
        loss_fun = globals()[loss_fun] if isinstance(loss_fun, str) else loss_fun
        self.model = model
        self.params = params
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun(model))
        self.state = self.solver.init_state(params)
        self.solver_step = jax.jit(self.solver.update)
        self.rng = np.random.default_rng(seed)
        self.metrics = Metrics(model, metrics)
        self.params_tree_structure = jax.tree_util.tree_structure(self.params)

    def set_parameters(self, params_leaves):
        self.params = jax.tree_util.tree_unflatten(self.params_tree_structure, params_leaves)

    def get_parameters(self):
        return jax.tree_util.tree_leaves(self.params)
    
    def step(self, X, Y, epochs, steps_per_epoch=None, verbose=0):
        for e in range(epochs):
            indices = np.arange(len(Y))
            self.rng.shuffle(indices)
            idx = indices[:len(indices) - (len(indices) % 32)].reshape((-1, 32))
            if steps_per_epoch:
                idx = idx[:steps_per_epoch]
            if verbose:
                idx = tqdm(idx)
            for ix in idx:
                self.params, self.state = self.solver_step(params=self.params, state=self.state, X=X[ix], Y=Y[ix])
                if verbose:
                    idx.set_postfix_str(f"LOSS: {self.state.value:.3f}, epoch: {e + 1}/{epochs}")
            if len(indices) % 32:
                ix = indices[-len(indices) % 32:]
                self.params, self.state = self.solver_step(params=self.params, state=self.state, X=X[ix], Y=Y[ix])
        return {"loss": self.state.value.item()}
    
    def evaluate(self, X, Y, verbose=0):
        indices = np.arange(len(Y))
        idx = indices[:len(indices) - (len(indices) % 32)].reshape((-1, 32))
        if verbose:
            idx = tqdm(idx)
        for ix in idx:
            self.metrics.add_batch(self.params, X[ix], Y[ix])
        if len(indices) % 32:
            ix = indices[-len(indices) % 32:]
            self.metrics.add_batch(self.params, X[ix], Y[ix])
        return self.metrics.compute()