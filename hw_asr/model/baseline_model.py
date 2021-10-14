from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.net(spectrogram)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here


class BaselineLSTM(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=256, num_layers=3, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=n_feats,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            *args, **kwargs
        )
        self.fc = nn.Linear(hidden_size, n_class, *args, **kwargs)

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.fc(self.lstm(spectrogram))}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
