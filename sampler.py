from typing import Optional, Sized, TypeVar, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler, DistributedSampler

from utils import classname

T_co = TypeVar('T_co', covariant=True)


class InfiniteRandomSampler(Sampler):

    def __init__(self, data_source, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        if len(self.data_source) > 0:
            while True:
                yield from self.__iter_once__()
        else:
            yield from iter([])

    def __iter_once__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.data_source)).tolist())
        return iter(torch.arange(start=0, end=len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


class LimitedIterationSampler(Sampler):
    """
    this is to give a limited size of batch sampler
    """

    def __init__(self, data_source: Optional[Sized], *, stop_iteration: int, shuffle: bool = True) -> None:
        super().__init__(data_source)
        self._data_source = data_source
        self._stop_iteration = stop_iteration
        self._shuffle = shuffle
        if self._shuffle is not True:
            raise NotImplementedError(self._shuffle)

    def __iter__(self):
        available_nums = np.arange(0, len(self._data_source))
        if self._shuffle:
            chosen_nums = np.random.choice(available_nums, size=self._stop_iteration, replace=True)
            return iter(chosen_nums)

    def __len__(self):
        return self._stop_iteration


class InfiniteDistributedSampler(DistributedSampler):
    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield from super(InfiniteDistributedSampler, self).__iter__()
            self._epoch += 1

    def set_epoch(self, epoch: int) -> None:
        raise RuntimeError(f"`set_epoch` not supported for {classname(self)}")
