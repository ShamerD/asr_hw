from typing import List, Tuple, Union

import torch
from ctcdecode import CTCBeamDecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.alphabet = [self.EMPTY_TOK] + alphabet

    def ctc_decode(self, indices: Union[List[int], torch.Tensor]) -> str:
        result = ""
        last_ind = -1

        if type(indices) is torch.Tensor:
            indices = indices.tolist()

        for ind in indices:
            if ind == last_ind:
                continue
            if ind == self.char2ind[self.EMPTY_TOK]:
                last_ind = -1
                continue
            result += self.ind2char[ind]
            last_ind = ind

        return result

    def ctc_beam_search(self, log_probs: torch.Tensor, log_probs_length: torch.Tensor,
                        beam_size: int = 100) -> List[List[Tuple[str, float]]]:
        """
        Performs beam search and returns a list (batch) of lists of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 3
        batch_size, char_length, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)

        decoder = CTCBeamDecoder(
            self.alphabet,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=beam_size,
            num_processes=1,
            blank_id=self.char2ind[self.EMPTY_TOK],
            log_probs_input=True
        )

        result = []
        for log_probs_vec, log_probs_len in zip(log_probs, log_probs_length):
            hypos = []
            beam_results, beam_scores, _, out_lens = decoder.decode(log_probs_vec[:log_probs_len].unsqueeze(0))
            for j in range(beam_size):
                # beam_score is -log(prob) in ctcdecode
                hypos.append(
                    (self.ctc_decode(beam_results[0][j][:out_lens[0][j]]), -beam_scores[0][j])
                )
            # beams are already sorted
            result.append(hypos)
        return result
