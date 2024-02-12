"""
The server that is placed in intermediate levels of a hierarchical federated learning system.
"""

from typing import Optional, Self, List
import time

from .common import logger
from .client_manager import ClientManager
from .client import Client
from . import strategy


class MiddleServer(Client):
    "Server for intermediate levels of hierarchical federated learning"

    def __init__(self, strategy_name: Optional[str] = None, client_manager: Optional[ClientManager] = None):
        match strategy_name:
            case "mrcs":
                self.strategy = strategy.MRCS()
            case "topk_kickback":
                self.strategy = strategy.TopKKickbackMomentum()
            case "kickback":
                self.strategy = strategy.KickbackMomentum()
            case "topk":
                self.strategy = strategy.TopK()
            case "fedprox":
                self.strategy = strategy.FedProx()
            case _:
                self.strategy = strategy.FedAVG()

        if not client_manager:
            client_manager = ClientManager()
        self.client_manager = client_manager

    def register_client(self, client: Client) -> Self:
        logger.info("Registering client to middle server")
        self.client_manager.register(client)
        return self

    def register_clients(self, clients: List[Client]) -> Self:
        logger.info(f"Registering {len(clients)} clients to middle server")
        for client in clients:
            self.client_manager.register(client)

    def fit(self, parameters, config):
        self.parameters = parameters

        logger.info(f"Starting training on middle server for {config['num_episodes']} episodes")
        start_time = time.time()
        for e in range(1, config['num_episodes'] + 1):
            client_parameters = []
            client_samples = []
            client_metrics = []
            clients = self.client_manager.sample()
            for c in clients:
                parameters, samples, metrics = c.fit(self.parameters, config)
                client_parameters.append(parameters)
                client_samples.append(samples)
                client_metrics.append(metrics)
            self.parameters = self.strategy.aggregate(
                client_parameters, client_samples, self.parameters, config
            )
        logger.info(f"Completed middle server training in {time.time() - start_time}s")
        logger.info(f"Distributed final metrics: {client_metrics}")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, config)
        logger.info(f"Aggregated final metrics {aggregated_metrics}")

        return self.parameters, sum(client_samples), aggregated_metrics

    def evaluate(self, parameters, config):
        logger.info("Performing analytics on middle server")
        start_time = time.time()
        client_samples = []
        client_metrics = []
        clients = self.client_manager.test_sample()
        for c in clients:
            samples, metrics = c.evaluate(self.parameters, config)
            client_samples.append(samples)
            client_metrics.append(metrics)
        logger.info(f"Completed middle server analytics in {time.time() - start_time}s")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, config)
        logger.info(f"Aggregated final metrics {aggregated_metrics}")

        return sum(client_samples), aggregated_metrics


class IntermediateFineTuner(MiddleServer):
    def evaluate(self, parameters, config):
        logger.info("Starting finetuning on middle server")
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
            tuned_parameters = self.strategy.aggregate(
                client_parameters, client_samples, tuned_parameters, config
            )
        logger.info(f"Completed middle server finetuning in {time.time() - start_time}s")

        logger.info("Performing analytics on middle server")
        start_time = time.time()
        client_samples = []
        client_metrics = []
        clients = self.client_manager.sample()
        for c in clients:
            samples, metrics = c.evaluate(tuned_parameters, config)
            client_samples.append(samples)
            client_metrics.append(metrics)
        logger.info(f"Completed middle server analytics in {time.time() - start_time}s")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, config)
        logger.info(f"Aggregated final metrics {aggregated_metrics}")

        return sum(client_samples), aggregated_metrics
