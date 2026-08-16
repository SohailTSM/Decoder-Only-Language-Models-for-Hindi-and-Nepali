"""Corpus-level deduplication: exact (document-level), paragraph-level, and MinHash
near-duplicate detection — applied in that order across ALL sources combined (manual +
public), since manual sources were previously only deduped against themselves and never
checked against each other or against the public corpus."""

import hashlib
import sys

from datasketch import MinHash, MinHashLSH


def _hash_text(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exact_dedup(documents: list[str]) -> list[str]:
    seen = set()
    out = []
    for doc in documents:
        h = _hash_text(doc)
        if h not in seen:
            seen.add(h)
            out.append(doc)
    return out


def paragraph_dedup(documents: list[str], min_paragraph_words: int = 8) -> list[str]:
    """Drops individual duplicate paragraphs (e.g. shared boilerplate/disclaimers
    repeated across many articles) rather than whole documents. Short paragraphs are
    left alone (a min_paragraph_words floor) since e.g. a byline or a one-line quote
    legitimately repeating isn't the "boilerplate" problem this targets."""
    seen = set()
    out = []
    for doc in documents:
        kept_paragraphs = []
        for para in doc.split("\n\n"):
            stripped = para.strip()
            if not stripped or len(stripped.split()) < min_paragraph_words:
                kept_paragraphs.append(para)
                continue
            h = _hash_text(stripped)
            if h not in seen:
                seen.add(h)
                kept_paragraphs.append(para)
        cleaned = "\n\n".join(p for p in kept_paragraphs if p.strip())
        if cleaned.strip():
            out.append(cleaned)
    return out


def _shingles(text: str, k: int = 5) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_dedup(
    documents: list[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    log_every: int = 50_000,
) -> list[str]:
    """LSH-based near-duplicate removal — catches the same underlying story reworded
    slightly across outlets, which exact/paragraph hashing misses."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    out = []
    for i, doc in enumerate(documents):
        mh = MinHash(num_perm=num_perm)
        for shingle in _shingles(doc):
            mh.update(shingle.encode("utf-8"))
        if not lsh.query(mh):
            lsh.insert(f"doc_{i}", mh)
            out.append(doc)
        if log_every and (i + 1) % log_every == 0:
            print(f"  [minhash_dedup] {i + 1:,}/{len(documents):,} scanned, "
                  f"{len(out):,} kept", file=sys.stderr)
    return out
