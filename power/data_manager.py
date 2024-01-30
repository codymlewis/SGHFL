"""
A fast and simple data management library for machine learning
"""

from typing import Dict, Callable, Iterable, Self
import numpy as np
from numpy.typing import NDArray


class DatasetDict:
    """
    Store and manage split datasets.
    """

    def __init__(self, data: Dict[str, NDArray]):
        """
        Input data is assumed to follow the format `X, Y key -> numpy array`.
        """
        self.__data = data
        length = 0
        for k, v in data.items():
            if length == 0:
                length = len(v)
            elif length != len(v):
                raise AttributeError(
                    f"Data should be composed of equal length arrays, column {k} has length {len(v)} should be {length}"
                )
        self.length = length

    def select(self, idx: int | Iterable[int | bool]):
        """
        Return a new DatasetDict that only contains the samples at the indices specified.

        Arguments:
        - idx: Index or indices of the samples to take from the data
        """
        return DatasetDict({k: v[idx] for k, v in self.__data.items()})

    def map(self, mapping_fn: Callable[[Dict[str, NDArray]], Dict[str, NDArray]]) -> Self:
        """
        Apply a mapping to the data.

        Arguments:
        - mapping_fn: A function that takes as input a dictionary of format `X, Y key -> numpy array` and returns the same
        """
        self.__data = mapping_fn(self.__data)
        return self

    def normalise(self, mean: float, std: float) -> Self:
        """
        Perform Guassian normalisation of features of the data according to the input statistics.

        Arguments:
        - mean: Mean value of the sample features
        - std: Standard deviation value of the sample features
        """
        for k, v in self.__data.items():
            if "X" in k:
                self.__data[k] = (v - mean) / std
        return self

    def normalize(self, mean: float, std: float) -> Self:
        """
        Perform Guassian normalization of features of the data according to the input statistics.

        Arguments:
        - mean: Mean value of the sample features
        - std: Standard deviation value of the sample features
        """
        return self.normalise(mean, std)

    def __getitem__(self, i: str) -> NDArray:
        return self.__data[i]

    def __len__(self) -> int:
        return len(self.__data['X'])

    def __str__(self) -> str:
        return str(self.__data)

    def short_details(self) -> str:
        "Give shortened details on the structure of the data."
        details = "{"
        for k, v in self.__data.items():
            details += f"{k}: type {v.dtype}, shape {v.shape}, range [{v.min()}, {v.max()}], "
        details = details[:-2] + "}"
        return details


class Dataset:
    """
    Store and manage a whole dataset.
    """

    def __init__(self, data: Dict[str, Dict[str, NDArray] | DatasetDict]):
        """
        Input data when creating a dataset is assumed to follow the format of
        `train/test/validation/etc. key -> X, Y keys -> numpy array`.
        Data is always assumed to have at least a train key with the corresponding structure underneath.
        """
        if np.all([isinstance(v, DatasetDict) for v in data.values()]):
            self.__data = data
        else:
            self.__data = {k: DatasetDict(v) for k, v in data.items()}

    def __getitem__(self, i: str) -> DatasetDict:
        return self.__data[i]

    def __str__(self) -> str:
        string = "{\n"
        for k, v in self.__data.items():
            string += f"\t{k}: {v.short_details()}\n"
        string += "}"
        return string

    def map(self, mapping_fn: Callable[[Dict[str, NDArray]], Dict[str, NDArray]]) -> Self:
        """
        Apply a mapping upon each of the split datasets.

        Arguments:
        - mapping_fn: A function that takes as input a split dataset of format `X, Y keys -> numpy array`
                      and returns one of the same format
        """
        for v in self.__data.values():
            v.map(mapping_fn)
        return self

    def keys(self) -> Iterable[str]:
        """
        Get the top level keys of the dataset, i.e., the split of the dataset.
        """
        return self.__data.keys()

    def select(self, idx_dict: Dict[str, int | Iterable[int | bool]]):
        """
        Return a subdataset which includes only the data at the specified indices.

        Arguments:
        - idx_dict: A dictionary with the format of `train/test/validation/etc. key -> numpy array of indices`
        """
        return Dataset({k: self.__data[k].select(idx) for k, idx in idx_dict.items()})

    def normalise(self) -> Self:
        """
        Normalise the data to the standard Guassian distribution on the basis of the training dataset.
        """
        mean = self.__data['train']['X'].mean()
        std = self.__data['train']['X'].std()
        self.__data = {k: v.normalise(mean, std) for k, v in self.__data.items()}
        return self

    def normalize(self) -> Self:
        """
        Normalize the data to the standard Guassian distribution on the basis of the training dataset.
        """
        return self.normalise()
