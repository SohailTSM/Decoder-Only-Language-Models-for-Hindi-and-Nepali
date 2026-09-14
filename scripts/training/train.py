"""Pretrain one monolingual GPT (Model H or Model L) on its .bin token stream.

Run independently per language, on Kaggle GPU. Designed to be interruption-safe:
checkpoints save optimizer + iteration state at a fixed interval so a killed/expired
Kaggle session can resume with --resume instead of restarting from scratch.

Usage:
    python -m scripts.training.train --lang hindi \
        --data-dir /kaggle/input/lma-hindi-bin/hindi/data/processed \
        --out-dir /kaggle/working/hindi/checkpoints

Resume:
    python -m scripts.training.train --lang hindi \
        --data-dir /kaggle/input/lma-hindi-bin/hindi/data/processed \
        --out-dir /kaggle/working/hindi/checkpoints \
        --resume /kaggle/working/hindi/checkpoints/last.pt

Optional logging/backup, both off by default:
    --wandb-project <proj>   log train/val loss + lr to Weights & Biases (needs
                              WANDB_API_KEY in the environment)
    --hf-repo-id <user/repo> push best.pt/last.pt/config.json to a HF Hub model repo
                              on every checkpoint save (needs HF_TOKEN in the environment)

Bonus ablation:
    --no-rope                disable RoPE entirely (report/bonus_ablation_report.md) -
                              no positional information anywhere in the model
"""

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.training.dataset import get_batch, load_memmap
from scripts.training.model import GPT, GPTConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer(model: torch.nn.Module, weight_decay: float, lr: float,
                         betas: tuple[float, float]) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def estimate_loss(model, data_by_split, block_size, batch_size, device, eval_iters, ctx):
    model.eval()
    out = {}
    for split, data in data_by_split.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def init_wandb(args, cfg: GPTConfig):
    import wandb
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or f"{args.lang}-gpt",
        config={**asdict(cfg), **{k: str(v) for k, v in vars(args).items()}},
    )


def hf_ensure_repo(repo_id: str, private: bool) -> None:
    from huggingface_hub import create_repo
    create_repo(repo_id, exist_ok=True, private=private, repo_type="model")


def hf_upload(repo_id: str, lang: str, path: Path, path_in_repo: str) -> None:
    try:
        from huggingface_hub import upload_file
        upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo,
                    repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"[{lang}] [warn] HF upload of {path_in_repo} failed: {e}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Directory with train.bin/val.bin (default: <lang>/data/processed)")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Checkpoint directory (default: <lang>/checkpoints)")
    ap.add_argument("--resume", type=Path, default=None,
                     help="Checkpoint file to resume training from")

    ap.add_argument("--vocab-size", type=int, default=16_000)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=11)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-rope", action="store_true",
                     help="Bonus ablation: disable RoPE entirely (no positional "
                          "information anywhere in the model)")

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                     help="Micro-batches accumulated per optimizer step. Effective batch "
                          "size = --batch-size * --grad-accum-steps, at --batch-size's "
                          "peak memory instead of the effective batch's.")
    ap.add_argument("--max-iters", type=int, default=20_000)
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--lr-decay-iters", type=int, default=None,
                     help="Default: same as --max-iters")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=100)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--checkpoint-interval", type=int, default=1000)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "bfloat16", "float16", "float32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--wandb-project", type=str, default=None,
                     help="Enable W&B logging under this project name")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)

    ap.add_argument("--hf-repo-id", type=str, default=None,
                     help="Enable pushing checkpoints to this HF Hub model repo (user/repo)")
    ap.add_argument("--hf-public", action="store_true",
                     help="Make the HF repo public (default: private)")
    args = ap.parse_args()

    data_dir = args.data_dir or (REPO_ROOT / args.lang / "data" / "processed")
    out_dir = args.out_dir or (REPO_ROOT / args.lang / "checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    lr_decay_iters = args.lr_decay_iters or args.max_iters

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
    if dtype == "float32":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    print(f"[{args.lang}] device={device} dtype={dtype}", file=sys.stderr)

    train_data = load_memmap(data_dir / "train.bin")
    val_data = load_memmap(data_dir / "val.bin")
    data_by_split = {"train": train_data, "val": val_data}

    cfg = GPTConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_seq_len=args.block_size,
        dropout=args.dropout, use_rope=not args.no_rope,
    )
    model = GPT(cfg).to(device)
    print(f"[{args.lang}] {model.num_params():,} parameters", file=sys.stderr)
    if args.compile:
        model = torch.compile(model)

    optimizer = configure_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2))

    wandb_run = init_wandb(args, cfg) if args.wandb_project else None

    if args.hf_repo_id:
        hf_ensure_repo(args.hf_repo_id, private=not args.hf_public)
        config_path = out_dir / "config.json"
        config_path.write_text(json.dumps(asdict(cfg), indent=2))
        hf_upload(args.hf_repo_id, args.lang, config_path, "config.json")

    iter_num = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        iter_num = ckpt["iter_num"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"[{args.lang}] resumed from {args.resume} at iter {iter_num}", file=sys.stderr)

    def save_checkpoint(path: Path, val_loss: float) -> None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if dtype == "float16" else None,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "config": asdict(cfg),
            "args": vars(args) | {"data_dir": str(data_dir), "out_dir": str(out_dir),
                                   "resume": str(args.resume) if args.resume else None},
        }, path)
        if args.hf_repo_id:
            hf_upload(args.hf_repo_id, args.lang, path, path.name)

    x, y = get_batch(train_data, args.block_size, args.batch_size, device)
    t0 = time.time()
    running_loss = 0.0

    while iter_num < args.max_iters:
        lr = get_lr(iter_num, args.warmup_iters, lr_decay_iters, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, data_by_split, args.block_size, args.batch_size,
                                    device, args.eval_iters, ctx)
            print(f"[{args.lang}] iter {iter_num}: train_loss={losses['train']:.4f} "
                  f"val_loss={losses['val']:.4f} lr={lr:.2e}", file=sys.stderr)
            if wandb_run is not None:
                wandb_run.log({"train/loss": losses["train"], "val/loss": losses["val"],
                                "lr": lr}, step=iter_num)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(out_dir / "best.pt", losses["val"])

        step_loss = 0.0
        for _ in range(args.grad_accum_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            scaler.scale(loss).backward()
            step_loss += loss.item()

        if args.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += step_loss
        if iter_num % args.log_interval == 0 and iter_num > 0:
            dt = time.time() - t0
            avg_loss = running_loss / args.log_interval
            ms_per_iter = dt / args.log_interval * 1000
            print(f"[{args.lang}] iter {iter_num}: loss={avg_loss:.4f} "
                  f"({ms_per_iter:.1f} ms/iter)", file=sys.stderr)
            if wandb_run is not None:
                wandb_run.log({"train/loss_smoothed": avg_loss, "perf/ms_per_iter": ms_per_iter},
                               step=iter_num)
            running_loss = 0.0
            t0 = time.time()

        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint(out_dir / "last.pt", best_val_loss)

        iter_num += 1

    save_checkpoint(out_dir / "last.pt", best_val_loss)
    print(f"[{args.lang}] training complete: {iter_num} iters, best_val_loss={best_val_loss:.4f}",
          file=sys.stderr)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
