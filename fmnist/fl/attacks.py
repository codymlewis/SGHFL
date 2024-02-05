import numpy as np
import scipy as sp

from . import client
from . import common
from .strategy import FedAVG


class EmptyUpdater(client.Client):
    def __init__(self, data, create_model_fn, corroborator, seed=None):
        super().__init__(data, create_model_fn, seed)

    def fit(self, parameters, config):
        _, samples, metrics = super().fit(parameters, config)
        return parameters, samples, metrics


class LIE(client.Client):
    def __init__(self, data, create_model_fn, corroborator, seed=None):
        super().__init__(data, create_model_fn, seed)
        self.corroborator = corroborator
        self.corroborator.register(self)

    def fit(self, parameters, config):
        z_max = self.corroborator.z_max
        history, mu, sigma = self.corroborator.calc_grad_stats(parameters, config)
        return [m + z_max * s for m, s in zip(mu, sigma)], len(self.data['train']), history

    def honest_fit(self, parameters, config):
        return super().fit(parameters, config)


class IPM(client.Client):
    def __init__(self, data, create_model_fn, corroborator, seed=None):
        super().__init__(data, create_model_fn, seed)
        self.corroborator = corroborator
        self.corroborator.register(self)

    def fit(self, parameters, config):
        history, mu, sigma = self.corroborator.calc_grad_stats(parameters, config)
        grads = [p - m for p, m in zip(parameters, mu)]
        return [p + (1.0 / self.corroborator.nadversaries) * g for p, g in zip(parameters, grads)], len(self.data['train']), history

    def honest_fit(self, parameters, config):
        return super().fit(parameters, config)


def backdoor_mapping(data, from_y, to_y):
    trigger = np.zeros((28, 28, 1))
    trigger[:5, :5, :] = 1
    def _apply(example):
        backdoor_idx = example['Y'] == from_y
        return {
            "X": np.array([bx if backdoor else tx for backdoor, bx, tx in zip(backdoor_idx, np.minimum(example['X'] + trigger, 1), example['X'])]),
            "true X": example['X'],
            "Y": np.where(backdoor_idx, to_y, example['Y']),
            "true Y": example["Y"],
        }
    return _apply

def sh_backdoor_mapping():
    def _apply(example):
        backdoor_idx = example['X'][:, -1, 3] > 0.9
        return {
            "X": example['X'],
            "true X": example['X'],
            "Y": np.where(backdoor_idx, 0, example['Y']),
            "true Y": example["Y"],
        }
    return _apply


class BackdoorClient(client.Client):
    def __init__(self, data, create_model_fn, backdoor_idx=None, seed=None):
        super().__init__(data, create_model_fn)
        self.backdoor_idx = backdoor_idx

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.step(
            self.data['train']['true X'],
            self.data['train']['true Y'],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get("num_steps"),
            verbose=0
        )
        return self.model.get_parameters(), len(self.data['train']), metrics

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.evaluate(
            self.data['test']['true X'],
            self.data['test']['true Y'],
            verbose=0
        )
        if self.backdoor_idx:
            backdoor_idx = self.backdoor_idx['test']
        else:
            backdoor_idx = self.data['test']['true Y'] == config['from_y']
        attacked_metrics = self.model.evaluate(
            self.data['test']['X'][backdoor_idx],
            self.data['test']['Y'][backdoor_idx],
            verbose=0
        )
        if attacked_metrics.get('accuracy'):
            metrics['asr'] = attacked_metrics['accuracy']
        else:
            metrics['amae'] = attacked_metrics['mean_absolute_error']
        return len(self.data['test']), metrics


class BackdoorLIE(BackdoorClient):
    def __init__(self, data, create_model_fn, corroborator, backdoor_idx=None, seed=None):
        super().__init__(data, create_model_fn, backdoor_idx, seed)
        self.corroborator = corroborator
        self.corroborator.register(self)

    def fit(self, parameters, config):
        update, history = self.corroborator.calc_backdoor_update(parameters, config)
        return update, len(self.data['train']), history

    def backdoor_fit(self, parameters, config):
        self.model.set_parameters(parameters)
        normal_state = self.model.state
        if self.backdoor_idx:
            backdoor_idx = self.backdoor_idx['train']
        else:
            backdoor_idx = self.data['train']['true Y'] == config['from_y']
        self.model.step(
            self.data['train']['X'][backdoor_idx],
            self.data['train']['Y'][backdoor_idx],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get("num_steps"),
            verbose=0
        )
        self.model.state = normal_state  # Ensure that adaptive gradients are not also corrupted
        return self.model.get_parameters()

    def honest_fit(self, parameters, config):
        return super().fit(parameters, config)


class Corroborator(FedAVG):
    def __init__(self, nclients):
        self.nclients = nclients
        self.adversaries = []
        self.round = 0
        self.mu = None
        self.sigma = None
        self.history = None

    @property
    def nadversaries(self):
        return len(self.adversaries)

    def register(self, adversary):
        self.adversaries.append(adversary)

    @property
    def z_max(self):
        s = self.nclients // 2 + 1 - self.nadversaries
        return sp.stats.norm.ppf((self.nclients - s) / self.nclients)

    def calc_grad_stats(self, parameters, config):
        honest_parameters = []
        honest_samples = []
        honest_metrics = []
        for a in self.adversaries:
            parameters, samples, metrics = a.honest_fit(parameters, config)
            honest_parameters.append(parameters)
            honest_samples.append(samples)
            honest_metrics.append(metrics)

        # Does some aggregation
        attr_honest_parameters = common.to_attribute_array(honest_parameters)
        mu = [np.average(layer, weights=honest_samples, axis=0) for layer in attr_honest_parameters]
        sigma = [np.sqrt(np.average((layer - m)**2, weights=honest_samples, axis=0)) for layer, m in zip(attr_honest_parameters, mu)]
        history = super().analytics(honest_metrics, honest_samples, config)
        return history, mu, sigma

    def calc_backdoor_update(self, parameters, config):
        if self.round == config['round']:
            return self.update, self.history

        self.history, self.mu, self.sigma = self.calc_grad_stats(parameters, config)
        backdoor_parameters = common.to_attribute_array([a.backdoor_fit(parameters, config) for a in self.adversaries])
        z_max = sp.stats.norm.ppf((self.nclients - (self.nclients // 2 + 1 - self.nadversaries)) / self.nclients)
        self.update = [
            np.clip(np.mean(p, axis=0), m - z_max * s, m + z_max * s)
            for p, m, s in zip(backdoor_parameters, self.mu, self.sigma)
        ]
        self.round = config['round']
        return self.update, self.history
