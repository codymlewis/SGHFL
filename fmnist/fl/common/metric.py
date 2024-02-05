from abc import ABC, abstractmethod
from typing import Iterable
from .typing import Parameters


class Metric(ABC):
    def __str__(self):
        return self.__class__.__name__.lower()

    @abstractmethod
    def __call__(self, global_parameters: Parameters, client_parameters: Iterable[Parameters], client_samples: Iterable[int]) -> int | float:
        pass