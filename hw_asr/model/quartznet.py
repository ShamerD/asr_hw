import torch
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
    def __init__(self, input_size, output_size, kernel_size, padding, stride=1, dilation=1, relu=True):
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
        layers = [TCSConv(input_size, output_size, kernel_size, padding=padding, stride=stride)]
        for i in range(1, repeats):
            layers.append(TCSConv(output_size, output_size, kernel_size,
                                  padding=padding, stride=stride, relu=(i + 1 != repeats)))

        self.net = Sequential(*layers)
        self.residual = Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size)
        )

    def forward(self, x):
        return F.relu(self.net(x) + self.residual(x))


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class):
        super().__init__(n_feats, n_class)
        self.c1 = TCSConv(n_feats, 256, kernel_size=33, padding=16, stride=2)
        self.blocks = nn.Sequential(
            QuartzNetBlock(256, 256, kernel_size=33),
            QuartzNetBlock(256, 256, kernel_size=39),
            QuartzNetBlock(256, 512, kernel_size=51),
            QuartzNetBlock(512, 512, kernel_size=63),
            QuartzNetBlock(512, 512, kernel_size=75)
        )
        self.c2 = TCSConv(512, 512, kernel_size=87, padding=86, dilation=2)
        self.c3 = ConvBlock(512, 1024, kernel_size=1)
        self.c4 = ConvBlock(1024, n_class, kernel_size=1)

    def forward(self, spectrogram, *args, **kwargs):
        # spectrogram is [batch_size, time, n_feats] but conv expects [batch_size, n_feats, time]
        x = torch.transpose(spectrogram, dim0=-1, dim1=-2)
        x = self.c1(x)
        x = self.blocks(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return {'logits': torch.transpose(x, -1, -2)}

    def transform_input_lengths(self, input_lengths):
        return torch.ceil(input_lengths / 2).type(torch.int32)
