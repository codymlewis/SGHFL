"""
Simulate federated learning.
"""

from typing import Callable, Dict, List, Optional

from .common import History
from .middle_server import MiddleServer
from .server import Server
from .client import Client


def start_simulation(
    server: Server,
    client_fn: Callable[[int], Client],
    network_arch: Dict[str, int | Dict | List[Dict]],
) -> History:
    """
    Construct and run a federated learning simulation.

    Arguments:
    - server: The top level server
    - client_fn: A function for creating a new client
    - network_arch: The architecture for the system, specified as a dictionary
    """
    clients = construct_network(client_fn, network_arch)
    server.register_clients(clients)
    history = server.fit()
    return history


def construct_network(
    client_fn: Callable[[int], Client],
    network_arch: Dict[str, int | Dict | List[Dict]],
    level_id: Optional[str] = "",
) -> List[Client]:
    "Construct the network as specified in the network_arch variable"
    if isinstance(network_arch['clients'], int):
        return [client_fn(f"{level_id}{i}") for i in range(network_arch['clients'])]
    elif isinstance(network_arch['clients'], dict):
        subnet = network_arch['clients']
        if subnet.get("middle_server_class"):
            middle_server_class = subnet['middle_server_class']
        else:
            middle_server_class = MiddleServer
        middle_server = middle_server_class(
            strategy=subnet.get("strategy"),
            client_manager=subnet.get("client_manager")
        )
        middle_server.register_clients([
            client_fn(f"{level_id}{i}") for i in range(subnet['clients'])
        ])
        return [middle_server]

    clients = []
    for i, subnet in enumerate(network_arch['clients']):
        if subnet.get("middle_server_class"):
            middle_server_class = subnet['middle_server_class']
        else:
            middle_server_class = MiddleServer
        middle_server = middle_server_class(
            strategy=subnet.get("strategy"),
            client_manager=subnet.get("client_manager")
        )
        middle_server.register_clients(construct_network(client_fn, subnet, level_id=f"{level_id}{i}-"))
        clients.append(middle_server)
    return clients
