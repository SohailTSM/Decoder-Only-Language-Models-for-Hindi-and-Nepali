"""Shared helper for loading a trained GPT from a train.py-style checkpoint dict."""

from pathlib import Path

import torch

from scripts.training.model import GPT, GPTConfig


def load_model(checkpoint_path: Path, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
