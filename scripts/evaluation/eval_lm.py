"""Test-set intrinsic language-modeling metrics: cross-entropy, perplexity, and
bits-per-byte (BPB), per assignment §2.3.

Evaluated deterministically over every non-overlapping block_size window of test.bin —
unlike train.py's periodic validation (a random sample every eval_interval, for a cheap
running signal during training), a final reported number should cover the whole held-out
set, not an estimate.

BPB makes the two languages comparable despite having different tokenizers/fertility:
    BPB = (mean_loss_nats * tokens_evaluated / ln(2)) / bytes(test.txt)

Usage:
    python -m scripts.evaluation.eval_lm --lang hindi \
        --checkpoint /path/to/best.pt \
        --data-dir <dir with test.bin> \
        --test-text <dir>/test.txt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.evaluation.checkpoint_utils import load_model
from scripts.training.dataset import load_memmap

REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def evaluate_test_loss(model, data: np.memmap, block_size: int, batch_size: int,
                        device: str) -> tuple[float, int]:
    n_windows = (len(data) - 1) // block_size
    total_loss = 0.0
    total_seqs = 0

    for batch_start in range(0, n_windows, batch_size):
        n_seq = min(batch_size, n_windows - batch_start)
        xs, ys = [], []
        for i in range(batch_start, batch_start + n_seq):
            s = i * block_size
            xs.append(data[s:s + block_size].astype(np.int64))
            ys.append(data[s + 1:s + 1 + block_size].astype(np.int64))
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * n_seq
        total_seqs += n_seq

    mean_loss = total_loss / total_seqs
    tokens_evaluated = total_seqs * block_size
    return mean_loss, tokens_evaluated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Directory with test.bin (default: <lang>/data/tokenized)")
    ap.add_argument("--test-text", type=Path, default=None,
                     help="Path to test.txt, for the byte count BPB needs "
                          "(default: <lang>/data/processed/test.txt)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="Write results JSON here")
    args = ap.parse_args()

    data_dir = args.data_dir or (REPO_ROOT / args.lang / "data" / "tokenized")
    test_text_path = args.test_text or (REPO_ROOT / args.lang / "data" / "processed" / "test.txt")

    model, ckpt = load_model(args.checkpoint, args.device)
    block_size = model.cfg.max_seq_len
    print(f"[{args.lang}] loaded checkpoint from iter {ckpt.get('iter_num')}, "
          f"block_size={block_size}", file=sys.stderr)

    test_data = load_memmap(data_dir / "test.bin")
    mean_loss, tokens_evaluated = evaluate_test_loss(
        model, test_data, block_size, args.batch_size, args.device)

    total_bytes = test_text_path.stat().st_size  # raw UTF-8 byte count of the held-out text
    total_bits = mean_loss * tokens_evaluated / math.log(2)
    bpb = total_bits / total_bytes
    perplexity = math.exp(mean_loss)

    results = {
        "lang": args.lang,
        "checkpoint": str(args.checkpoint),
        "checkpoint_iter": ckpt.get("iter_num"),
        "test_loss_nats": mean_loss,
        "perplexity": perplexity,
        "bpb": bpb,
        "tokens_evaluated": tokens_evaluated,
        "test_bytes": total_bytes,
    }
    print(json.dumps(results, indent=2), file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"[{args.lang}] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
