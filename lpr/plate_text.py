"""Egyptian licence-plate text: normalisation, validation and formatting.

Egyptian plates carry up to 3 Arabic letters and up to 4 Arabic-Indic digits
under a blue "مصر EGYPT" band. Letters are read right-to-left, digits
left-to-right. Everything here is pure Python so it can be unit-tested
without the vision models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Letters used on Egyptian plates, in their canonical form.
PLATE_LETTERS = "ابجدرسصطعفقلمنهوى"

# OCR / typing variants mapped onto the canonical letter.
_LETTER_VARIANTS = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ي": "ى",
    "ئ": "ى",
    "ة": "ه",
    "ہ": "ه",
    "ھ": "ه",
    "ؤ": "و",
    "ک": "ق",  # rare OCR confusion
}

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = {str(i): ARABIC_DIGITS[i] for i in range(10)}
_DIGIT_MAP.update({"۰۱۲۳۴۵۶۷۸۹"[i]: ARABIC_DIGITS[i] for i in range(10)})  # Persian digits

_TATWEEL_AND_MARKS = re.compile("[\u0640\u064B-\u065F\u0670\u200C-\u200F\u202A-\u202E]")


@dataclass(frozen=True)
class PlateRules:
    min_letters: int = 1
    max_letters: int = 3
    min_digits: int = 1
    max_digits: int = 4
    min_total: int = 4


@dataclass(frozen=True)
class Plate:
    letters: str  # canonical letters in reading order, no spaces, e.g. "بطص"
    digits: str  # Arabic-Indic digits in reading order, e.g. "١٢٣٤"

    @property
    def key(self) -> str:
        """Compact key used for storage, search and voting."""
        return f"{self.letters}{self.digits}"

    @property
    def display(self) -> str:
        """Human form: spaced letters then digits, e.g. "ب ط ص ١٢٣٤"."""
        return f"{' '.join(self.letters)} {self.digits}".strip()


def normalize_char(ch: str) -> str:
    ch = _LETTER_VARIANTS.get(ch, ch)
    return _DIGIT_MAP.get(ch, ch)


def normalize_text(text: str) -> str:
    """Map OCR output to canonical letters/digits and drop everything else."""
    text = _TATWEEL_AND_MARKS.sub("", text or "")
    out = []
    for ch in text:
        ch = normalize_char(ch)
        if ch in PLATE_LETTERS or ch in ARABIC_DIGITS:
            out.append(ch)
    return "".join(out)


def split_letters_digits(text: str) -> tuple[str, str]:
    norm = normalize_text(text)
    letters = "".join(c for c in norm if c in PLATE_LETTERS)
    digits = "".join(c for c in norm if c in ARABIC_DIGITS)
    return letters, digits


def validate(letters: str, digits: str, rules: PlateRules = PlateRules()) -> Plate | None:
    if not (rules.min_letters <= len(letters) <= rules.max_letters):
        return None
    if not (rules.min_digits <= len(digits) <= rules.max_digits):
        return None
    if len(letters) + len(digits) < rules.min_total:
        return None
    if digits.startswith("٠"):
        return None  # Egyptian plate numbers never start with zero
    return Plate(letters=letters, digits=digits)


def parse(text: str, rules: PlateRules = PlateRules()) -> Plate | None:
    """Parse free text (user search, single OCR token) into a plate."""
    letters, digits = split_letters_digits(text)
    return validate(letters, digits, rules)


def search_key(text: str) -> str:
    """Normalise a user search query to the compact storage form."""
    return normalize_text(text)
