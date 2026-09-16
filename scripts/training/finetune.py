"""Reasoning-finetune one pretrained monolingual GPT (Model H or Model L) on its own
Phase 3 synthetic reasoning dataset (assignment PDF §3.1 "Protocol"): start from that
language's own Phase 2 pretrained checkpoint, keep the tokenizer/vocab fixed, finetune
independently per language (no shared data/weights), save in the same resume-capable
checkpoint format pretraining uses.

Unlike pretrain.py's memmap random-window sampling over a ~930M-token stream, the
reasoning dataset is small and discrete (a few thousand short examples) - this script
uses an ordinary shuffled `DataLoader` (scripts/training/finetune_dataset.py) and trains
for a fixed number of epochs over it, with a full pass over the (small) val split at
the end of every epoch rather than a fixed-sample estimate.

Usage (first run, from a Phase 2 checkpoint):
    python -m scripts.training.finetune --lang hindi \
        --init-checkpoint /path/to/hindi/best.pt \
        --out-dir /kaggle/working/hindi/reasoning_checkpoints

Resume an interrupted finetune run:
    python -m scripts.training.finetune --lang hindi \
        --resume /kaggle/working/hindi/reasoning_checkpoints/last.pt \
        --out-dir /kaggle/working/hindi/reasoning_checkpoints

Optional logging/backup, both off by default (same flags as train.py):
    --wandb-project <proj>   log train/val loss + lr to Weights & Biases
    --hf-repo-id <user/repo> push best.pt/last.pt/config.json to a HF Hub model repo
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from scripts.training.finetune_dataset import ReasoningDataset, collate_fn
from scripts.training.model import GPT, GPTConfig
from scripts.training.train import configure_optimizer, get_lr, hf_ensure_repo, hf_upload

REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def evaluate_full(model, loader, device, ctx) -> float:
    model.eval()
    total_loss, total_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def init_wandb(args, cfg: GPTConfig):
    import wandb
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or f"{args.lang}-gpt-reasoning",
        config={**asdict(cfg), **{k: str(v) for k, v in vars(args).items()}},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Directory with train.jsonl/val.jsonl (default: <lang>/reasoning/data)")
    ap.add_argument("--tokenizer", type=Path, default=None,
                     help="Default: <lang>/tokenizer/<lang>_unigram_16k.model")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Checkpoint directory (default: <lang>/reasoning/checkpoints)")
    ap.add_argument("--init-checkpoint", type=Path, default=None,
                     help="Phase 2 pretrained checkpoint to start finetuning from "
                          "(required unless --resume is given)")
    ap.add_argument("--resume", type=Path, default=None,
                     help="Finetune checkpoint to resume training from")

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup-iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=None,
                     help="Run a full val pass + checkpoint check every this many "
                          "iters, instead of only at epoch boundaries. Default: once "
                          "per epoch (same as before) - set this explicitly when "
                          "epochs are large relative to how often you want to check, "
                          "so 'best.pt' isn't tied to how many examples happen to fit "
                          "in one epoch.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)

    ap.add_argument("--hf-repo-id", type=str, default=None,
                     help="Enable pushing checkpoints to this HF Hub model repo (user/repo)")
    ap.add_argument("--hf-public", action="store_true")
    args = ap.parse_args()

    if args.init_checkpoint is None and args.resume is None:
        ap.error("one of --init-checkpoint or --resume is required")

    data_dir = args.data_dir or (REPO_ROOT / args.lang / "reasoning" / "data")
    out_dir = args.out_dir or (REPO_ROOT / args.lang / "reasoning" / "checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer or (
        REPO_ROOT / args.lang / "tokenizer" / f"{args.lang}_unigram_16k.model")

    torch.manual_seed(args.seed)
    device = args.device

    if args.dtype is not None:
        dtype = args.dtype
    elif device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    elif device == "cuda":
        dtype = "float16"
    else:
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if dtype == "float32" else torch.amp.autocast(device_type=device, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    print(f"[{args.lang}] device={device} dtype={dtype}", file=sys.stderr)

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    pad_id = sp.pad_id()
    train_ds = ReasoningDataset(data_dir / "train.jsonl", sp)
    val_ds = ReasoningDataset(data_dir / "val.jsonl", sp)
    collate = partial(collate_fn, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"[{args.lang}] train={len(train_ds)} val={len(val_ds)} examples", file=sys.stderr)

    iters_per_epoch = len(train_loader)
    max_iters = iters_per_epoch * args.epochs

    iter_num = 0
    best_val_loss = float("inf")
    cfg = None
    model = None
    optimizer = None

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = configure_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2))
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        iter_num = ckpt["iter_num"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"[{args.lang}] resumed from {args.resume} at iter {iter_num}", file=sys.stderr)
    else:
        base_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        cfg = GPTConfig(**base_ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(base_ckpt["model"])
        optimizer = configure_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2))
        print(f"[{args.lang}] initialized from {args.init_checkpoint} "
              f"(pretrained iter {base_ckpt.get('iter_num')})", file=sys.stderr)

    print(f"[{args.lang}] {model.num_params():,} parameters, "
          f"{max_iters} max_iters ({args.epochs} epochs x {iters_per_epoch} iters/epoch)",
          file=sys.stderr)

    wandb_run = init_wandb(args, cfg) if args.wandb_project else None
    if args.hf_repo_id:
        hf_ensure_repo(args.hf_repo_id, private=not args.hf_public)
        config_path = out_dir / "config.json"
        config_path.write_text(json.dumps(asdict(cfg), indent=2))
        hf_upload(args.hf_repo_id, args.lang, config_path, "config.json")

    def save_checkpoint(path: Path, val_loss: float) -> None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if dtype == "float16" else None,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "config": asdict(cfg),
            "args": vars(args) | {"data_dir": str(data_dir), "out_dir": str(out_dir)},
        }, path)
        if args.hf_repo_id:
            hf_upload(args.hf_repo_id, args.lang, path, path.name)

    eval_interval = args.eval_interval or iters_per_epoch

    def check_and_save(epoch: int) -> float:
        val_loss = evaluate_full(model, val_loader, device, ctx)
        print(f"[{args.lang}] iter {iter_num} (epoch {epoch}): val_loss={val_loss:.4f}",
              file=sys.stderr)
        if wandb_run is not None:
            wandb_run.log({"val/loss": val_loss}, step=iter_num)
        return val_loss

    t0 = time.time()
    running_loss, running_count = 0.0, 0
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            lr = get_lr(iter_num, args.warmup_iters, max_iters, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_count += 1
            if iter_num % args.log_interval == 0 and iter_num > 0:
                dt = time.time() - t0
                avg_loss = running_loss / running_count
                print(f"[{args.lang}] iter {iter_num}/{max_iters} (epoch {epoch}): "
                      f"loss={avg_loss:.4f} lr={lr:.2e} ({dt:.1f}s)", file=sys.stderr)
                if wandb_run is not None:
                    wandb_run.log({"train/loss_smoothed": avg_loss, "lr": lr}, step=iter_num)
                running_loss, running_count = 0.0, 0
                t0 = time.time()
            iter_num += 1

            if iter_num % eval_interval == 0:
                val_loss = check_and_save(epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(out_dir / "best.pt", val_loss)
                save_checkpoint(out_dir / "last.pt", val_loss)

    print(f"[{args.lang}] finetuning complete: {iter_num} iters, "
          f"best_val_loss={best_val_loss:.4f}", file=sys.stderr)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
