"""Prompt-loss-masked supervised finetuning dataset for the Phase 3 reasoning task.

Small, discrete dataset (thousands of short examples) - unlike pretraining's ~930M-token
flat memmap stream (dataset.py), every example is tokenized once at construction and
kept fully in memory; no separate prepare/.bin step is needed at this scale.

Each example becomes one packed sequence `prompt_ids + answer_ids + [eos_id]`. The
collate function right-pads to the batch's max length and sets the shifted target to
-100 (the default `F.cross_entropy` ignore_index GPT.forward already uses, model.py:164)
at every prompt-token and pad position, so only answer+eos positions contribute to the
loss. No attention/padding mask is needed: GPT's causal self-attention (`is_causal=True`,
model.py) already guarantees a real position never attends to anything after it, so
trailing pad tokens can never influence a real prediction.
"""

import json
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class ReasoningDataset(Dataset):
    def __init__(self, jsonl_path: Path, sp: spm.SentencePieceProcessor):
        self.examples: list[tuple[list[int], list[bool]]] = []
        eos_id = sp.eos_id()
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompt_ids = sp.encode(rec["prompt"])
                answer_ids = sp.encode(rec["answer"])
                input_ids = prompt_ids + answer_ids + [eos_id]
                loss_mask = [False] * len(prompt_ids) + [True] * (len(answer_ids) + 1)
                self.examples.append((input_ids, loss_mask))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[bool]]:
        return self.examples[idx]


def collate_fn(batch: list[tuple[list[int], list[bool]]], pad_id: int = 0
               ) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [(ids[:-1], ids[1:], mask[1:]) for ids, mask in batch]
    max_len = max(len(x) for x, _, _ in seqs)
    batch_size = len(seqs)

    x = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    y = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, (xi, yi, mi) in enumerate(seqs):
        length = len(xi)
        x[i, :length] = torch.tensor(xi, dtype=torch.long)
        yi_t = torch.tensor(yi, dtype=torch.long)
        mi_t = torch.tensor(mi, dtype=torch.bool)
        y[i, :length][mi_t] = yi_t[mi_t]
    return x, y
