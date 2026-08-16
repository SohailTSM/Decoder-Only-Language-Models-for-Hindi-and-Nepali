"""Evaluate a trained tokenizer on the held-out validation split (never seen during
training): fertility (tokens/word), UNK rate, average characters/token, most frequent
tokens, and example tokenizations — the exact per-language stats the assignment asks
for. Also supports comparing multiple candidate vocab sizes if trained.

Usage: python -m scripts.tokenizer.evaluate_tokenizer --lang hindi
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_LINES = 200_000


def evaluate(lang: str, model_path: Path, sample_lines: int = DEFAULT_SAMPLE_LINES) -> dict:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = sp.unk_id()

    val_path = REPO_ROOT / lang / "data" / "processed" / "val.txt"
    if not val_path.exists():
        raise FileNotFoundError(f"{val_path} not found — run preprocessing first")

    lines = []
    with val_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= sample_lines:
                break

    total_words = total_tokens = total_chars = unk_count = 0
    token_freq: Counter = Counter()

    for line in lines:
        ids = sp.encode(line, out_type=int)
        total_words += len(line.split())
        total_tokens += len(ids)
        total_chars += len(line)
        unk_count += sum(1 for i in ids if i == unk_id)
        token_freq.update(ids)

    fertility = total_tokens / total_words if total_words else 0.0
    unk_rate = unk_count / total_tokens if total_tokens else 0.0
    avg_chars_per_token = total_chars / total_tokens if total_tokens else 0.0

    examples = []
    for line in lines[:5]:
        pieces = sp.encode(line, out_type=str)
        examples.append({"text": line[:150], "num_tokens": len(pieces), "tokens": pieces[:40]})

    stats = {
        "language": lang,
        "model_file": model_path.name,
        "vocab_size": sp.vocab_size(),
        "eval_split": "val (held out, never used in training)",
        "eval_lines_sampled": len(lines),
        "eval_words": total_words,
        "eval_tokens": total_tokens,
        "fertility_tokens_per_word": round(fertility, 4),
        "unk_rate": round(unk_rate, 6),
        "avg_chars_per_token": round(avg_chars_per_token, 4),
        "top_20_most_frequent_tokens": [
            {"token": sp.id_to_piece(tid), "count": c}
            for tid, c in token_freq.most_common(20)
        ],
        "examples": examples,
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--model", type=Path, default=None,
                     help="Path to a specific .model file; defaults to the newest "
                          "<lang>_unigram_*.model under <lang>/tokenizer/.")
    ap.add_argument("--sample-lines", type=int, default=DEFAULT_SAMPLE_LINES)
    args = ap.parse_args()

    model_path = args.model
    if model_path is None:
        candidates = sorted((REPO_ROOT / args.lang / "tokenizer").glob(
            f"{args.lang}_unigram_*.model"))
        if not candidates:
            raise FileNotFoundError(
                f"no trained tokenizer found under {args.lang}/tokenizer/ — run "
                f"scripts.tokenizer.train_tokenizer first")
        model_path = candidates[-1]

    stats = evaluate(args.lang, model_path, args.sample_lines)

    out_path = REPO_ROOT / "report" / f"{args.lang}_tokenizer_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[{args.lang}] stats written to {out_path}", file=sys.stderr)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
