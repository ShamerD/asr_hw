import torch.distributions
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, scale=0.02, *args, **kwargs):
        self.sampler = torch.distributions.Normal(0.0, scale=scale)

    def __call__(self, data: Tensor):
        noise = self.sampler.sample(data.size())
        return data + noise
