from functools import partial
import time
import flagon
import numpy as np
import jax
import jax.numpy as jnp


class IntermediateFineTuner(flagon.MiddleServer):
    def evaluate(self, parameters, config):
        flagon.common.logger.info("Starting finetuning on middle server")
        strategy = flagon.strategy.FedAVG()  # Use standard FedAVG for finetuning since it does not need to conform with the upper tier
        start_time = time.time()
        tuned_parameters = parameters
        for e in range(1, config['num_finetune_episodes'] + 1):
            client_parameters = []
            client_samples = []
            client_metrics = []
            clients = self.client_manager.sample()
            for c in clients:
                parameters, samples, metrics = c.fit(tuned_parameters, config)
                client_parameters.append(parameters)
                client_samples.append(samples)
                client_metrics.append(metrics)
            tuned_parameters = strategy.aggregate(
                client_parameters, client_samples, tuned_parameters, config
            )
        flagon.common.logger.info(f"Completed middle server finetuning in {time.time() - start_time}s")

        flagon.common.logger.info("Performing analytics on middle server")
        start_time = time.time()
        client_samples = []
        client_metrics = []
        clients = self.client_manager.sample()
        for c in clients:
            samples, metrics = c.evaluate(tuned_parameters, config)
            client_samples.append(samples)
            client_metrics.append(metrics)
        flagon.common.logger.info(f"Completed middle server analytics in {time.time() - start_time}s")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, config)
        flagon.common.logger.info(f"Aggregated final metrics {aggregated_metrics}")

        return sum(client_samples), aggregated_metrics


def mse(a_tree, b):
    a, _ = jax.flatten_util.ravel_pytree(a_tree)
    return jnp.mean((a - b)**2)


def cosine_distance(a_tree, b):
    a, _ = jax.flatten_util.ravel_pytree(a_tree)
    return 1 - jnp.abs(a.dot(b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def adaptive_loss(model, loss_fun, old_params, dist_fun=mse):
    def _apply(params, X, Y):
        return loss_fun(params, X, Y) + dist_fun(params, old_params)
    return _apply


class AdaptiveLoss(flagon.MiddleServer):
    def __init__(self, strategy, client_manager=None):
        super().__init__(strategy, client_manager)
        self.num_clients = 0

    def fit(self, parameters, config):
        if len(self.client_manager.clients) > self.num_clients:
            self.num_clients = len(self.client_manager.clients)
        elif len(self.client_manager.clients) < self.num_clients or config.get("adapt_loss"):
            flagon.common.logger.info("A middle server was lost, adapting loss")
            flattened_parameters = jnp.concatenate([jnp.array(p.reshape(-1)) for p in parameters])
            for c in self.client_manager.clients:
                c.model.change_loss_fun(
                    partial(
                        adaptive_loss,
                        loss_fun=c.model.loss_fun,
                        old_params=flattened_parameters,
                        dist_fun=mse if config['adaptive_loss'] == "mse" else cosine_distance
                    )
                )
            self.num_clients = len(self.client_manager.clients)
        return super().fit(parameters, config)


class AdaptiveLossIntermediateFineTuner(AdaptiveLoss):
    def evaluate(self, parameters, config):
        flagon.common.logger.info("Starting finetuning on middle server")
        strategy = flagon.strategy.FedAVG()  # Use standard FedAVG for finetuning since it does not need to conform with the upper tier
        start_time = time.time()
        tuned_parameters = parameters
        for e in range(1, config['num_finetune_episodes'] + 1):
            client_parameters = []
            client_samples = []
            client_metrics = []
            clients = self.client_manager.sample()
            for c in clients:
                parameters, samples, metrics = c.fit(tuned_parameters, config)
                client_parameters.append(parameters)
                client_samples.append(samples)
                client_metrics.append(metrics)
            tuned_parameters = strategy.aggregate(
                client_parameters, client_samples, tuned_parameters, config
            )
        flagon.common.logger.info(f"Completed middle server finetuning in {time.time() - start_time}s")

        flagon.common.logger.info("Performing analytics on middle server")
        start_time = time.time()
        client_samples = []
        client_metrics = []
        clients = self.client_manager.sample()
        for c in clients:
            samples, metrics = c.evaluate(tuned_parameters, config)
            client_samples.append(samples)
            client_metrics.append(metrics)
        flagon.common.logger.info(f"Completed middle server analytics in {time.time() - start_time}s")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, config)
        flagon.common.logger.info(f"Aggregated final metrics {aggregated_metrics}")

        return sum(client_samples), aggregated_metrics