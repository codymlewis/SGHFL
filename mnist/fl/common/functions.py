"""
Commonly used and flexible functions
"""


from typing import List, T, Dict, Iterable
import numpy as np
from numpy.typing import NDArray


def to_attribute_array(convertee: Iterable[Iterable[T]]) -> List[NDArray[T]]:
    """
    Convert a collection of per client elements to a list of numpy arrays aligned attribute-wise

    Arguments:
    - convertee: Collection of collections, where the first axis is client-wise and the second is attribute-wise
    """
    return [np.array(t) for t in zip(*convertee)]


def count_clients(network_arch: Dict[str, int | Dict | List[Dict]]) -> int:
    """
    Count the number of clients within a network architecture specification

    Arguments:
    - network_arch: A network architecture specification
    """
    if isinstance(network_arch['clients'], int):
        return network_arch['clients']
    if isinstance(network_arch['clients'], dict):
        return count_clients(network_arch['clients'])

    count = 0
    for subnet in network_arch['clients']:
        count += count_clients(subnet)
    return count
