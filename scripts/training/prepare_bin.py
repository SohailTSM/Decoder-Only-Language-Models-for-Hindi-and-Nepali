"""Tokenize train/val/test.txt into flat uint16 token-id binaries for pretraining.

Each line of a split's .txt is treated as one document. Documents are tokenized in
batches (fast path through sentencepiece) and joined with the tokenizer's <eos> id
(id=3) into a single continuous token stream per split, written straight to disk as
we go — never held fully in memory, so this works the same whether train.txt is
500MB or 5GB. No <bos> is inserted: <eos> alone is enough to mark document
boundaries, matching standard GPT-style pretraining corpora (e.g. GPT-2/nanoGPT).

The training script (scripts/training/train.py) memory-maps this flat stream and
slices random T+1-length windows from it on the fly, rather than this script
pre-cutting fixed-length examples — see report/phase2_model_report.md discussion.

Usage:
    python -m scripts.training.prepare_bin --lang hindi
    python -m scripts.training.prepare_bin --lang nepali \
        --data-dir /kaggle/input/lma-nepali/nepali/data/processed \
        --tokenizer /kaggle/input/lma-nepali/nepali/tokenizer/nepali_unigram_16k.model \
        --out-dir /kaggle/working/nepali/data/tokenized
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNK_LINES = 20_000  # documents encoded per sentencepiece batch call


def tokenize_split(sp: spm.SentencePieceProcessor, txt_path: Path, bin_path: Path) -> int:
    eos_id = sp.eos_id()
    total_tokens = 0
    buf: list[str] = []

    with open(txt_path, "r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        def flush() -> None:
            nonlocal total_tokens
            if not buf:
                return
            flat: list[int] = []
            for ids in sp.encode(buf):
                flat.extend(ids)
                flat.append(eos_id)
            arr = np.array(flat, dtype=np.uint16)
            arr.tofile(fout)
            total_tokens += arr.size
            buf.clear()

        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            buf.append(line)
            if len(buf) >= CHUNK_LINES:
                flush()
        flush()

    return total_tokens


def prepare(lang: str, data_dir: Path, tokenizer_path: Path, out_dir: Path) -> None:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "lang": lang,
        "tokenizer": str(tokenizer_path),
        "vocab_size": sp.vocab_size(),
        "eos_id": sp.eos_id(),
        "bos_id": sp.bos_id(),
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
        "dtype": "uint16",
        "tokens": {},
    }

    for split in ("train", "val", "test"):
        txt_path = data_dir / f"{split}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"{txt_path} not found")
        bin_path = out_dir / f"{split}.bin"
        print(f"[{lang}] tokenizing {txt_path} -> {bin_path}", file=sys.stderr)
        n = tokenize_split(sp, txt_path, bin_path)
        meta["tokens"][split] = n
        print(f"[{lang}] {split}: {n:,} tokens", file=sys.stderr)

    meta_path = out_dir / "tokenized_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{lang}] wrote {meta_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Directory containing train/val/test.txt "
                          "(default: <lang>/data/processed)")
    ap.add_argument("--tokenizer", type=Path, default=None,
                     help="Path to the .model file "
                          "(default: <lang>/tokenizer/<lang>_unigram_16k.model)")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Where to write {split}.bin + tokenized_meta.json "
                          "(default: <lang>/data/tokenized)")
    args = ap.parse_args()

    data_dir = args.data_dir or (REPO_ROOT / args.lang / "data" / "processed")
    tokenizer_path = args.tokenizer or (
        REPO_ROOT / args.lang / "tokenizer" / f"{args.lang}_unigram_16k.model")
    out_dir = args.out_dir or (REPO_ROOT / args.lang / "data" / "tokenized")

    prepare(args.lang, data_dir, tokenizer_path, out_dir)


if __name__ == "__main__":
    main()
