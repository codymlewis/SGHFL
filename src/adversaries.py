import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp

import fl


class EmptyUpdater(fl.Client):
    def step(self, global_params, batch_size=128, steps=1):
        self.state = self.state.replace(params=global_params)
        idx = self.rng.choice(len(self.data), batch_size, replace=False)
        loss, _ = fl.learner_step(self.state, self.data.X[idx], self.data.Y[idx])
        return loss, global_params


class Adversary(fl.Client):
    def __init__(self, client_id, model, info=None, data=None, seed=0, corroborator=None):
        super().__init__(client_id, model, info, data, seed)
        self.corroborator = corroborator
        self.corroborator.register(self)

    def honest_step(self, global_params, batch_size=128, steps=1):
        return super().step(global_params, batch_size, steps=steps)


class LIE(Adversary):
    def step(self, global_params, batch_size=128, steps=1):
        mu, sigma, loss = self.corroborator.calc_grad_stats(global_params, self.id, batch_size, steps)
        return loss, lie(mu, sigma, self.corroborator.z_max)


@jax.jit
def lie(mu, sigma, z_max):
    return jax.tree_util.tree_map(lambda m, s: m + z_max * s, mu, sigma)


class IPM(Adversary):
    def step(self, global_params, batch_size=128, steps=1):
        mu, sigma, loss = self.corroborator.calc_grad_stats(global_params, self.id, batch_size, steps)
        return loss, ipm(global_params, mu, self.corroborator.nadversaries)


@jax.jit
def ipm(params, mu, nadversaries):
    grads = jax.tree_util.tree_map(lambda p, m: p - m, params, mu)
    return jax.tree_util.tree_map(lambda p, g: p + (1 / nadversaries) * g, params, grads)


class Corroborator:
    def __init__(self, nclients, nadversaries):
        self.nclients = nclients
        self.adversaries = []
        self.nadversaries = nadversaries
        self.mu = None
        self.sigma = None
        self.loss = None
        self.parameters = None
        s = self.nclients // 2 + 1 - self.nadversaries
        self.z_max = jsp.stats.norm.ppf((self.nclients - s) / self.nclients)
        self.adv_ids = []
        self.updated_advs = []

    def register(self, adversary):
        self.adversaries.append(adversary)
        self.adv_ids.append(adversary.id)

    def calc_grad_stats(self, global_params, adv_id, batch_size, steps=1):
        if self.updated_advs:
            self.updated_advs.append(adv_id)
            if set(self.updated_advs) == set(self.adv_ids):  # if everyone has updated, stop using the cached value
                self.updated_advs = []
            return self.mu, self.sigma, self.loss

        honest_parameters = []
        honest_losses = []
        for a in self.adversaries:
            loss, parameters = a.honest_step(global_params, batch_size, steps)
            honest_parameters.append(parameters)
            honest_losses.append(loss)

        # Does some aggregation
        self.mu = fl.fedavg(honest_parameters)
        self.sigma = tree_std(honest_parameters, self.mu)
        self.loss = np.mean(honest_losses)
        self.updated_advs.append(adv_id)
        return self.mu, self.sigma, self.loss


def tree_std(trees, tree_mean):
    diffs = [jax.tree_util.tree_map(lambda p, m: (p - m)**2, tree, tree_mean) for tree in trees]
    var = jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *diffs)
    std = jax.tree_util.tree_map(lambda x: jnp.sqrt(x), var)
    return std
