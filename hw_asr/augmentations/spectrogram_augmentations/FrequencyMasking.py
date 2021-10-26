from torch import Tensor
from torchaudio.transforms import FrequencyMasking as torchFrequencyMasking

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = torchFrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self.aug(data)
