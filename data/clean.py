import re
from langdetect import detect
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def length_filter(src: str, tgt: str) -> bool:
    sw, tw = len(src.split()), len(tgt.split())
    if sw < 5 or sw > 100 or tw < 5 or tw > 100:
        return False
    if max(sw, tw) / min(sw, tw) > 2.5:
        return False
    return True


def copy_filter(src: str, tgt: str, lo: float = 0.25, hi: float = 0.80) -> bool:
    # BLEU lower bound raised from 0.10: a pair with BLEU < 0.25 is almost
    # always an unrelated sentence, not a paraphrase. Upper bound nudged
    # down to 0.80 to drop near-identical pairs.
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([src.split()], tgt.split(), smoothing_function=smooth)
    return lo <= bleu <= hi


def lang_filter(src: str, tgt: str) -> bool:
    try:
        return detect(src) == "en" and detect(tgt) == "en"
    except Exception:
        return False


def basic_clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_pair(src: str, tgt: str) -> bool:
    src, tgt = basic_clean(src), basic_clean(tgt)
    if not src or not tgt:
        return False
    return length_filter(src, tgt) and copy_filter(src, tgt) and lang_filter(src, tgt)
