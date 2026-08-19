"""Vocab-size selection sweep: train several candidate vocab sizes per language on the
same fixed sample and measure fertility/UNK rate on held-out val.txt, per the
assignment's guidance ("choose using fertility / unknown-token rate on held-out text").

Uses the 150,000-sentence sample size (not the final 1M-sample models) because this is
an exploratory sweep across many training runs, and 150k is the value already proven to
train reliably within this machine's memory limits.

Usage: python -m scripts.tokenizer.vocab_size_sweep --lang hindi
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_SENTENCE_SIZE = 150_000
EVAL_SAMPLE_LINES = 200_000
VOCAB_SIZES = [4_000, 8_000, 16_000, 24_000, 32_000, 40_000, 48_000]


def train_and_eval(lang: str, vocab_size: int, out_dir: Path) -> dict:
    train_path = REPO_ROOT / lang / "data" / "processed" / "train.txt"
    val_path = REPO_ROOT / lang / "data" / "processed" / "val.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out_dir / f"{lang}_unigram_{vocab_size // 1000}k")

    print(f"[{lang}] training vocab_size={vocab_size:,} "
          f"(input_sentence_size={INPUT_SENTENCE_SIZE:,})...", file=sys.stderr)
    spm.SentencePieceTrainer.train(
        input=str(train_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        num_threads=6,
    )

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    unk_id = sp.unk_id()

    lines = []
    with val_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= EVAL_SAMPLE_LINES:
                break

    total_words = total_tokens = unk_count = total_chars = 0
    for line in lines:
        ids = sp.encode(line, out_type=int)
        total_words += len(line.split())
        total_tokens += len(ids)
        total_chars += len(line)
        unk_count += sum(1 for i in ids if i == unk_id)

    result = {
        "language": lang,
        "vocab_size": vocab_size,
        "input_sentence_size": INPUT_SENTENCE_SIZE,
        "eval_lines_sampled": len(lines),
        "eval_words": total_words,
        "eval_tokens": total_tokens,
        "fertility_tokens_per_word": round(total_tokens / total_words, 4) if total_words else 0.0,
        "unk_rate": round(unk_count / total_tokens, 6) if total_tokens else 0.0,
        "avg_chars_per_token": round(total_chars / total_tokens, 4) if total_tokens else 0.0,
    }
    print(f"[{lang}] vocab_size={vocab_size:,}: "
          f"fertility={result['fertility_tokens_per_word']}, "
          f"unk_rate={result['unk_rate']}, "
          f"avg_chars_per_token={result['avg_chars_per_token']}", file=sys.stderr)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=VOCAB_SIZES)
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.lang / "tokenizer" / "vocab_sweep"
    results = [train_and_eval(args.lang, v, out_dir) for v in args.vocab_sizes]

    results_path = REPO_ROOT / "report" / f"{args.lang}_vocab_sweep.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[{args.lang}] sweep results written to {results_path}", file=sys.stderr)

    print(f"\n| Vocab size | Fertility | UNK rate | Avg chars/token |")
    print(f"|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['vocab_size']:,} | {r['fertility_tokens_per_word']} | "
              f"{r['unk_rate']*100:.3f}% | {r['avg_chars_per_token']} |")


if __name__ == "__main__":
    main()
