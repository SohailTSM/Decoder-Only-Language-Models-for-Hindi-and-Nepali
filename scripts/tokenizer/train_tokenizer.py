"""Train a SentencePiece Unigram tokenizer from scratch for one language, on that
language's own train split only (never the other language's data, never a shared
vocab — each model gets an independent tokenizer per the assignment's constraints).

Usage: python -m scripts.tokenizer.train_tokenizer --lang hindi
"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_SIZE = 32_000
# SentencePiece samples this many lines (shuffled) from train.txt rather than reading
# the whole multi-GB file — standard practice; a large representative sample trains a
# vocabulary just as well as the full corpus at a fraction of the time/memory.
# Kept small (rather than the more common 1-5M) because this machine has only ~7.6GB
# RAM and the unigram trainer's EM phase needs far more memory than the raw sampled
# text (suffix-array/substring statistics over the corpus). 5M sentences OOM-killed the
# process outright; even 1M (292MB of text) hit std::bad_alloc under a 3GB cap right
# after preprocessing, before the seed vocabulary/EM step could even start.
INPUT_SENTENCE_SIZE = 150_000


def train(lang: str, vocab_size: int = VOCAB_SIZE,
          input_sentence_size: int = INPUT_SENTENCE_SIZE) -> Path:
    train_path = REPO_ROOT / lang / "data" / "processed" / "train.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found — run the preprocessing pipeline "
            f"(scripts.preprocessing.build_corpus) first")

    out_dir = REPO_ROOT / lang / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out_dir / f"{lang}_unigram_{vocab_size // 1000}k")

    print(f"[{lang}] training SentencePiece Unigram tokenizer "
          f"(vocab_size={vocab_size:,}, sampling up to {input_sentence_size:,} lines "
          f"from {train_path})...", file=sys.stderr)

    spm.SentencePieceTrainer.train(
        input=str(train_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,  # standard default; excludes only extremely rare/noise chars
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        num_threads=6,
    )
    print(f"[{lang}] wrote {model_prefix}.model / {model_prefix}.vocab", file=sys.stderr)
    return Path(f"{model_prefix}.model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--input-sentence-size", type=int, default=INPUT_SENTENCE_SIZE)
    args = ap.parse_args()
    train(args.lang, args.vocab_size, args.input_sentence_size)


if __name__ == "__main__":
    main()
