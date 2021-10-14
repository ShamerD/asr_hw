import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    if len(dataset_items) == 0:
        raise ValueError("Trying to collate empty list")

    # make a dict of lists from list of dicts
    # keys are ["audio", "spectrogram", "duration", "text", "text_encoded", "audio_path", "text_encoded_length",
    dataset_dict = {}
    for dict_key in dataset_items[0]:
        dataset_dict[dict_key] = [dataset_item[dict_key] for dataset_item in dataset_items]

    # keys that do not need additional processing
    for key in ['audio', 'duration', 'text', 'audio_path']:
        result_batch[key] = dataset_dict[key]

    # spectrogram is [1(channel), n_features, time] so we squeeze and transpose
    dataset_dict['spectrogram'] = [spec.squeeze().transpose(0, 1) for spec in dataset_dict['spectrogram']]
    result_batch['spectrogram'] = pad_sequence(dataset_dict['spectrogram'], batch_first=True)

    dataset_dict['text_encoded'] = [text.squeeze() for text in dataset_dict['text_encoded']]
    result_batch['text_encoded'] = pad_sequence(dataset_dict['text_encoded'], batch_first=True)

    result_batch['text_encoded_length'] = torch.Tensor([x.size()[0] for x in dataset_dict['text_encoded']])
    result_batch['spectrogram_length'] = torch.Tensor([x.size()[0] for x in dataset_dict['spectrogram']])

    return result_batch
