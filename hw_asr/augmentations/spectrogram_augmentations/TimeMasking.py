from torch import Tensor
from torchaudio.transforms import TimeMasking as torchTimeMasking

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = torchTimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self.aug(data)
