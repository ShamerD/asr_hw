from torch import Tensor
import torch.distributions
from torchaudio.transforms import TimeStretch as torchTimeStretch

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, min_rate, max_rate, *args, **kwargs):
        assert 0 < min_rate <= max_rate
        self.sampler = torch.distributions.Uniform(min_rate, max_rate)
        self.aug = torchTimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        rate = self.sampler.sample().item()
        return self.aug(data, rate)
