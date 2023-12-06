"""
Commonly used typings.
"""


from typing import Dict, List, Self
from numpy.typing import NDArray


Config = Dict[str, str | int | float]
"Configuration object held by the server and passed throughout learning/analytics"

Parameters = List[NDArray]
"Model parameters to be trained"

Metrics = List[Dict[str, int | float]]
"Metrics recorded by model evaluation"


class History:
    "A per-round recording of the metrics during federated learning"

    def __init__(self):
        self.history: Dict[int, List[Metrics]] = {}
        self.aggregate_history: Dict[int, Metrics] = {}
        self.test_history: Dict[int, Metrics] = {}

    def add(self, round_num: int, metrics: List[Metrics]) -> Self:
        """
        Add the recorded metrics for this round

        Arguments:
        - round_num: The number of the round this is recorded at
        - metrics: Metrics to record
        """
        self.history[round_num] = metrics
        return self

    def add_aggregate(self, round_num, aggregated_metrics: Metrics) -> Self:
        """
        Add aggregated metrics for this round.

        Arguments:
        - round_num: The number of the round this is recorded at
        - aggregated_metrics: Metrics to record
        """
        self.aggregate_history[round_num] = aggregated_metrics
        return self

    def add_test(self, round_num, test_metrics: Metrics) -> Self:
        """
        Add test metrics for this round.

        Arguments:
        - round_num: The number of the round this is recorded at
        - test_metrics: Metrics to record
        """
        self.test_history[round_num] = test_metrics
