from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1)
        for log_prob_vec, target_text in zip(predictions, text):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCCharTextEncoder, use_lm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.lm_path = "./lm.arpa" if use_lm else None

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], *args, **kwargs):
        cers = []
        predictions = self.text_encoder.ctc_beam_search(log_probs, log_probs_length, lm_path=self.lm_path)
        for pred, target_text in zip(predictions, text):
            pred_text = pred[0][0]
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
