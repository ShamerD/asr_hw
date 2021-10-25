from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

from hw_asr.base import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.net = Sequential(
            nn.Conv1d(input_size, output_size, kernel_size, stride=stride, dilation=dilation),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class TCSConv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, dilation=1, relu=True):
        super().__init__()
        layers = [
            nn.Conv1d(input_size, output_size, kernel_size,
                      padding=padding, stride=stride, dilation=dilation, groups=input_size),
            nn.Conv1d(output_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size)
        ]
        if relu:
            layers.append(nn.ReLU())
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QuartzNetBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride=1, repeats=5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [TCSConv(input_size, output_size, kernel_size, stride, padding=padding)]
        for i in range(1, repeats):
            layers.append(TCSConv(output_size, output_size, kernel_size, stride,
                                  padding=padding, relu=(i + 1 != repeats)))

        self.net = Sequential(*layers)
        self.residual = Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size)
        )

    def forward(self, x):
        F.relu(self.net(x) + self.residual(x))


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class):
        super().__init__(n_feats, n_class)
        self.c1 = ConvBlock(n_feats, 256, kernel_size=33, stride=2)
        self.blocks = nn.Sequential(
            QuartzNetBlock(256, 256, kernel_size=33),
            QuartzNetBlock(256, 256, kernel_size=39),
            QuartzNetBlock(256, 512, kernel_size=51),
            QuartzNetBlock(512, 512, kernel_size=63),
            QuartzNetBlock(512, 512, kernel_size=75)
        )
        self.c2 = ConvBlock(512, 512, kernel_size=87)
        self.c3 = ConvBlock(512, 1024, kernel_size=1)
        self.c4 = ConvBlock(1024, n_class, kernel_size=1, dilation=2)

    def forward(self, spectrogram, *args, **kwargs):
        x = self.c1(spectrogram)
        x = self.blocks(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return {'logits': x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
