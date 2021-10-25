import unittest

import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        probs = torch.zeros((1, 3, len(text_encoder.char2ind)))
        probs.fill_(1e-7)

        probs[0][0][0] = 0.1
        probs[0][0][1] = 0.5
        probs[0][0][2] = 0.4

        probs[0][1][0] = 0.2
        probs[0][1][1] = 0.2
        probs[0][1][2] = 0.6

        probs[0][2][0] = 0.7
        probs[0][2][1] = 0.1
        probs[0][2][2] = 0.2

        beam_search_results = text_encoder.ctc_beam_search(probs.log(), torch.tensor([3]), beam_size=3)
        self.assertEqual(len(beam_search_results), 1)
        self.assertEqual(len(beam_search_results[0]), 3)

        hypots = [x[0] for x in beam_search_results[0]]

        # these have highest probability in hand calculation
        self.assertIn('ab', hypots)
        self.assertIn('b', hypots)
        print("Beam search results: ", beam_search_results)
