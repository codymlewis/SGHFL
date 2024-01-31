"""
The server for federated learning, usually it just organizes the clients and aggregates their updates/metrics.
"""

from typing import Optional, Self, Iterable, Tuple, List
import time
import numpy as np

from .common import Parameters, History, Config, Metrics, logger
from .client import Client
from .client_manager import ClientManager
from . import strategy


class Server:
    "A server for federated learning"

    def __init__(
        self,
        initial_parameters: Parameters,
        config: Config,
        strategy_name: Optional[str] = None,
        client_manager: Optional[ClientManager] = None
    ):
        match strategy_name:
            case "median":
                self.strategy = strategy.Median()
            case "centre":
                self.strategy = strategy.Centre()
            case "krum":
                self.strategy = strategy.Krum()
            case "trimmed_mean":
                self.strategy = strategy.TrimmedMean()
            case _:
                self.strategy = strategy.FedAVG()
        if not client_manager:
            client_manager = ClientManager()
        self.client_manager = client_manager
        self.parameters = initial_parameters
        self.config = config
        self.config['round'] = 1

    def validate_config(self):
        "Check that the server config is valid, raise an error if it is not"
        if self.config.get("num_rounds") is None:
            raise AttributeError("The server config is missing the num_rounds attribute")
        if self.config["num_rounds"] < 0:
            raise AttributeError("The server config num_rounds attribute can not be negative")

        if self.config.get("num_epochs") is None:
            raise AttributeError("The server config is missing the num_epochs attribute")
        if self.config["num_epochs"] < 0:
            raise AttributeError("The server config num_epochs attribute can not be negative")

    def register_client(self, client: Client) -> Self:
        "Add a client to this server"
        logger.info("Registering client to the server")
        self.client_manager.register(client)
        return self

    def register_clients(self, clients: Iterable[Client]) -> Self:
        "Add a collections of clients"
        logger.info(f"Registering {len(clients)} clients to the server")
        for client in clients:
            self.client_manager.register(client)

    def fit(self) -> History:
        "Perform the configured rounds of training for federated learning."
        self.validate_config()
        history = History()

        logger.info(f"Starting training on the server for {self.config['num_rounds']} rounds")
        start_time = time.time()
        for r in range(1, self.config['num_rounds'] + 1):
            history, client_parameters, client_samples, client_metrics = self.round_fit(r, history)

        logger.info(f"Completed server training in {time.time() - start_time}s")
        logger.info(f"Distributed metrics: {history.history}")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, self.config)
        if self.config.get("metrics"):
            aggregated_metrics = calc_additional_metrics(
                aggregated_metrics, self.config, self.parameters, client_parameters, client_samples
            )
        history.add_aggregate(self.config['num_rounds'], aggregated_metrics)
        logger.info(f"Aggregated final metrics {aggregated_metrics}")
        logger.info("Finding test metrics")
        test_metrics = self.evaluate()
        history.add_test(r, test_metrics)

        return history

    def round_fit(self, r: int, history: History) -> Tuple[History, List[Parameters], List[int], List[Metrics]]:
        client_parameters = []
        client_samples = []
        client_metrics = []
        clients = self.client_manager.sample()
        for c in clients:
            parameters, samples, metrics = c.fit(self.parameters, self.config)
            client_parameters.append(parameters)
            client_samples.append(samples)
            client_metrics.append(metrics)
        self.parameters = self.strategy.aggregate(
            client_parameters, client_samples, self.parameters, self.config
        )
        history.add(r, client_metrics)
        p = self.config.get("eval_every") and (r % self.config.get("eval_every")) == 0
        q = self.config.get("eval_at") and self.config.get("eval_at") == r
        if p or q:
            aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, self.config)
            history.add_aggregate(r, aggregated_metrics)
            logger.info(f"Aggregated training metrics at round {r}: {aggregated_metrics}")
            logger.info("Finding test metrics")
            test_metrics = self.evaluate()
            history.add_test(r, test_metrics)
        self.config['round'] += 1
        return history, client_parameters, client_samples, client_metrics

    def evaluate(self) -> Metrics:
        "Perform federated analytics for the model performance"
        logger.info("Performing analytics on the server")
        start_time = time.time()
        client_samples = []
        client_metrics = []
        clients = self.client_manager.test_sample()
        for c in clients:
            samples, metrics = c.evaluate(self.parameters, self.config)
            client_samples.append(samples)
            client_metrics.append(metrics)
        logger.info(f"Completed server analytics in {time.time() - start_time}s")
        aggregated_metrics = self.strategy.analytics(client_metrics, client_samples, self.config)
        logger.info(f"Aggregated final metrics {aggregated_metrics}")

        return aggregated_metrics


def calc_additional_metrics(aggregated_metrics, config, global_parameters, client_parameters, client_samples):
    for metric in config['metrics']:
        aggregated_metrics[str(metric)] = metric(global_parameters, client_parameters, client_samples)
    return aggregated_metrics


class DroppingClientManager(ClientManager):
    def __init__(self, drop_round, seed=None):
        super().__init__()
        self.round = 0
        self.drop_round = drop_round
        self.test_clients = []
        self.rng = np.random.default_rng(seed)

    def register(self, client):
        super().register(client)
        self.test_clients.append(client)

    def sample(self):
        self.round += 1
        if self.round == self.drop_round:
            for _ in range(2):
                self.clients.pop()
        return super().sample()

    def test_sample(self):
        return self.test_clients


class FractionalClientManager(ClientManager):
    def __init__(self, k=1, seed=None):
        super().__init__()
        self.k = k
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.choice(self.clients, min(len(self.clients), self.k), replace=False)
