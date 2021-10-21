# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    if len(target_text) == 0:
        return 1.0
    edit_dist = editdistance.eval(target_text, predicted_text)
    return edit_dist / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    edit_dist = editdistance.eval(target_words, predicted_words)
    return edit_dist / len(target_words)
