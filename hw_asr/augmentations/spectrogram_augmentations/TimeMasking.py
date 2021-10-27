import random

from torch import Tensor
from torchaudio.transforms import TimeMasking as torchTimeMasking

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, prob=0.5, ratio=0.1):
        self.prob = prob
        self.ratio = ratio

    def __call__(self, data: Tensor):
        q = random.random()
        if q < self.prob:
            data = torchTimeMasking(int(self.ratio * data.size()[-1]))(data)
        return data
