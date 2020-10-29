"""Module with base dataset class"""

from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base dataset"""

    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError
