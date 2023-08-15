import numpy as np
import flagon


class DroppingClientManager(flagon.client_manager.ClientManager):
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
                # self.clients.pop(self.rng.integers(len(self.clients)))
        return super().sample()

    def test_sample(self):
        return self.test_clients


class Adaptive(flagon.Server):
    def __init__(self, initial_parameters, config, strategy=None, client_manager=None):
        super().__init__(initial_parameters, config, strategy, client_manager)
        self.num_clients = 0

    def round_fit(self, r, history):
        self.config["adapt_loss"] = False
        if len(self.client_manager.clients) > self.num_clients:
            self.num_clients = len(self.client_manager.clients)
        elif len(self.client_manager.clients) < self.num_clients:
            flagon.common.logger.info("Lost clients, adapting loss")
            self.config["adapt_loss"] = True
            self.num_clients = len(self.client_manager.clients)
        return super().round_fit(r, history)