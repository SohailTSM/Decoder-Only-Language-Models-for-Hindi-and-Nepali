"""Document-level cleaning: Unicode normalization, rule-based correction, line-level
script-purity filtering, quality heuristics, repetitive-document detection, and a
keyword-based safety filter. Each function operates on one document's text at a time;
build_corpus.py applies them per document across the whole corpus."""

import re
import unicodedata
from pathlib import Path

# Devanagari block covers both Hindi and Nepali.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_HTML_ENTITY_RE = re.compile(r"&(nbsp|amp|lt|gt|quot|#39);")
_HTML_ENTITY_MAP = {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">", "quot": '"', "#39": "'"}

WORDLISTS_DIR = Path(__file__).parent / "wordlists"


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def rule_based_correction(text: str) -> str:
    """Strip control characters, leftover HTML entities, and collapse excess whitespace."""
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HTML_ENTITY_RE.sub(lambda m: _HTML_ENTITY_MAP[m.group(1)], text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def script_purity_filter(text: str, threshold: float = 0.75) -> str:
    """Drop individual lines below `threshold` fraction native-script (Devanagari)
    characters — same rule IndicCorpV2 itself uses, applied at line granularity so a
    stray English heading is dropped without discarding the rest of the document."""
    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue
        total = len(stripped)
        native = len(_DEVANAGARI_RE.findall(stripped))
        if total == 0 or native / total >= threshold:
            kept_lines.append(line)
    return "\n".join(kept_lines)


def passes_text_metrics(text: str, min_words: int = 15, max_symbol_ratio: float = 0.3) -> bool:
    """Gopher/C4-style heuristics: minimum length, and a cap on the fraction of
    "words" that are just punctuation/symbols (junk/spam signal)."""
    words = text.split()
    if len(words) < min_words:
        return False
    symbol_words = sum(1 for w in words if not any(c.isalnum() for c in w))
    if symbol_words / len(words) > max_symbol_ratio:
        return False
    return True


def is_repetitive(text: str, max_duplicate_line_ratio: float = 0.3) -> bool:
    """Flags documents dominated by repeated lines (nav menus, boilerplate spam)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return False
    duplicate_count = len(lines) - len(set(lines))
    return duplicate_count / len(lines) > max_duplicate_line_ratio


_BLOCKLIST_FILES = {"hindi": "offensive_hi.txt", "nepali": "offensive_ne.txt"}


def load_blocklist(lang: str) -> set[str]:
    """Hindi and Nepali share a lot of vocabulary (including profanity), so Nepali's
    blocklist is unioned with Hindi's rather than used alone — see
    report/phase1_data_report.md for why the Nepali-specific list is short (most
    open Nepali profanity lists found were either romanized or contaminated with
    political/ethnic terms, so this is a small, hand-curated, high-precision list)."""
    with (WORDLISTS_DIR / _BLOCKLIST_FILES["hindi"]).open(encoding="utf-8") as f:
        words = {line.strip().lower() for line in f if line.strip()}
    if lang == "nepali":
        with (WORDLISTS_DIR / _BLOCKLIST_FILES["nepali"]).open(encoding="utf-8") as f:
            words |= {line.strip().lower() for line in f if line.strip()}
    return words


def passes_safety_filter(text: str, blocklist: set[str]) -> bool:
    lowered = text.lower()
    return not any(word in lowered for word in blocklist)


def clean_document(text: str, blocklist: set[str]) -> str | None:
    """Full pipeline for one document. Returns cleaned text, or None if the document
    should be dropped entirely."""
    text = normalize_unicode(text)
    text = rule_based_correction(text)
    text = script_purity_filter(text)
    text = rule_based_correction(text)  # re-collapse blank lines left by the filter above
    if not passes_text_metrics(text):
        return None
    if is_repetitive(text):
        return None
    if not passes_safety_filter(text, blocklist):
        return None
    return text
