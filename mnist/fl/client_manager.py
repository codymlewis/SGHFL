"""
Manager for client selection.
"""

from typing import Self, Iterable
from .client import Client


class ClientManager:
    "Manager for client selection made by a server."

    def __init__(self):
        self.clients = []

    def register(self, client: Client) -> Self:
        """
        Add a client to this manager

        Arguments:
        - client: Client to add
        """
        self.clients.append(client)
        return self

    def sample(self) -> Iterable[Client]:
        """
        Get a subset of the clients as an iterable
        """
        return self.clients

    def test_sample(self) -> Iterable[Client]:
        """
        Client sampling method for evaluating model performance.
        """
        return self.sample()
