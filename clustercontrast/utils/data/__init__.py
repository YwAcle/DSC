from __future__ import absolute_import

from .base_dataset import BaseDataset, BaseImageDataset
from .preprocessor import Preprocessor


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = iter(self.loader)

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)


    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
