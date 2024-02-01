import numpy as np
import scipy as sp

import fl
import load_data
from logger import logger


class EmptyUpdater(fl.Client):
    def step(self, parameters, config):
        _, loss, samples = super().step(parameters, config)
        return parameters, loss, samples


class Adversary(fl.Client):
    def __init__(self, data, corroborator):
        super().__init__(data)
        self.corroborator = corroborator
        self.corroborator.register(self)

    def honest_step(self, parameters, config):
        return super().step(parameters, config)


class LIE(Adversary):
    def step(self, parameters, config):
        mu, sigma, loss = self.corroborator.calc_grad_stats(parameters, config)
        return mu + self.corroborator.z_max * sigma, loss, len(self.data['train'])


class IPM(Adversary):
    def step(self, parameters, config):
        mu, sigma, loss = self.corroborator.calc_grad_stats(parameters, config)
        grads = parameters - mu
        return parameters + (1 / self.corroborator.nadversaries) * grads, loss, len(self.data['train'])


class BackdoorLIE(Adversary):
    def __init__(self, data, corroborator):
        super().__init__(data, corroborator)
        self.backdoor_X, self.backdoor_Y = load_data.gen_backdoor_data(self.data['train']['X'], self.data['train']['Y'])

    def step(self, parameters, config):
        parameters, loss = self.corroborator.calc_backdoor_parameters(parameters, config)
        return parameters, loss, len(self.data['train'])

    def backdoor_step(self, parameters, config):
        model = fl.RidgeModel()
        model.parameters = parameters
        loss = model.fit(self.backdoor_X, self.backdoor_Y, epochs=config['num_epochs'])
        return model.parameters, loss, len(self.data['train'])


class Corroborator:
    def __init__(self, nclients):
        self.nclients = nclients
        self.adversaries = []
        self.nadversaries = 0
        self.round = -1
        self.mu = None
        self.sigma = None
        self.loss = None
        self.parameters = None
        self.z_max = 0

    def register(self, adversary):
        self.adversaries.append(adversary)
        self.nadversaries += 1
        s = self.nclients // 2 + 1 - self.nadversaries
        self.z_max = sp.stats.norm.ppf((self.nclients - s) / self.nclients)

    def calc_grad_stats(self, parameters, config):
        if self.round == config['round']:
            return self.mu, self.sigma, self.loss

        honest_parameters = []
        honest_samples = []
        honest_losses = []
        for a in self.adversaries:
            parameters, loss, samples = a.honest_step(parameters, config)
            honest_parameters.append(parameters)
            honest_samples.append(samples)
            honest_losses.append(loss)

        # Does some aggregation
        self.mu = np.average(honest_parameters, weights=honest_samples, axis=0)
        self.sigma = np.sqrt(np.average((honest_parameters - self.mu)**2, weights=honest_samples, axis=0))
        self.loss = np.average(honest_losses, weights=honest_samples)
        self.round = config['round']
        return self.mu, self.sigma, self.loss

    def calc_backdoor_parameters(self, parameters, config):
        if self.round == config['round']:
            return self.parameters, self.loss

        self.calc_grad_stats(parameters, config)

        backdoor_parameters = []
        backdoor_samples = []
        backdoor_losses = []
        for a in self.adversaries:
            parameters, loss, samples = a.backdoor_step(parameters, config)
            backdoor_parameters.append(parameters)
            backdoor_samples.append(samples)
            backdoor_losses.append(loss)
        logger.info(
            "The backdoor attack attained a loss of %f",
            np.average(backdoor_losses, weights=backdoor_samples)
        )

        self.parameters = np.clip(
            np.average(backdoor_parameters, weights=backdoor_samples, axis=0),
            self.mu - self.z_max * self.sigma,
            self.mu + self.z_max * self.sigma
        )
        return self.parameters, self.loss
