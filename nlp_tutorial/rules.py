"""A text-only OCR baseline."""

import re
import unicodedata

REPETITION = re.compile(r"(.{1,12}?)\1{3,}")


def score_ocr(text):
    unusual = sum(
        character == "\ufffd"
        or (character not in "\n\r\t" and unicodedata.category(character) in {"Cc", "Co"})
        for character in text
    )
    repeats = sum(1 for _ in REPETITION.finditer(text))
    return {"score": unusual + repeats, "unusual": unusual, "repeated_fragments": repeats}
