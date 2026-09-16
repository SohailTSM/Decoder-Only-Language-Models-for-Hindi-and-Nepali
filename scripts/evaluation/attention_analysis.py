"""Attention analysis, per assignment §2.3 "Attention analysis":

- Heatmaps (query vs. key position), all heads, for 2 early + 2 mid + 2 late layers
  (exceeds the assignment's "at least one early layer, one late layer" minimum), on an
  example sentence in the model's own language.
- Attention entropy per head/layer (higher = more diffuse).
- Mean attention distance per head (how far, on average, each query attends).

Uses CausalSelfAttention's `return_attn=True` path (model.py), which computes attention
manually (softmax(QK^T/sqrt(d_k) + M)V) instead of the fused SDPA kernel training/inference
otherwise use, since the fused kernel never materializes the weights themselves.

Usage:
    python -m scripts.evaluation.attention_analysis --lang hindi \
        --checkpoint /path/to/best.pt --out-dir results/hindi_attention
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch
from matplotlib import font_manager

matplotlib.use("Agg")

from scripts.evaluation.checkpoint_utils import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


def find_devanagari_font() -> font_manager.FontProperties | None:
    """Hindi and Nepali are both written in Devanagari; the default matplotlib font
    (DejaVu Sans) has no Devanagari glyphs, so token-label axes render as empty boxes
    without this. Used only for the token tick-labels themselves (via `fontproperties=`)
    — the rest of each figure (titles, "head N", colorbar) is plain Latin text and stays
    on the default font, since Devanagari-only fonts don't cover Latin glyphs either.
    Returns None (with a warning) if no such font is installed — install one (e.g.
    `apt-get install -y fonts-noto`) and re-run if that happens."""
    candidates = ["Noto Sans Devanagari", "Noto Sans Devanagari UI", "Lohit Devanagari"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return font_manager.FontProperties(family=name)

    # Not in matplotlib's cached font list — that cache can be stale relative to a font
    # installed moments earlier in the same session (e.g. via `apt-get install
    # fonts-noto` right before this runs). Search common system font directories
    # directly and register any match with the live FontManager, which works
    # regardless of the cache's state.
    search_globs = [
        "/usr/share/fonts/**/*NotoSansDevanagari*.ttf",
        "/usr/share/fonts/**/*Devanagari*.ttf",
        "/usr/local/share/fonts/**/*Devanagari*.ttf",
    ]
    for pattern in search_globs:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            font_manager.fontManager.addfont(matches[0])
            return font_manager.FontProperties(fname=matches[0])

    print("[warn] no Devanagari-capable font found — token labels on heatmaps will "
          "render as missing-glyph boxes. Install one (e.g. `apt-get install -y "
          "fonts-noto`) and re-run.", file=sys.stderr)
    return None


def get_example_sentences(test_text_path: Path, n: int, max_tokens_len: int,
                           sp: spm.SentencePieceProcessor) -> list[str]:
    sentences = []
    with open(test_text_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids = sp.encode(line)
            if 6 <= len(ids) <= max_tokens_len:
                sentences.append(line)
            if len(sentences) >= n:
                break
    return sentences


def get_sentences_from_file(prompts_file: Path, n: int) -> list[str]:
    """Fixed hand-picked prompts (e.g. comparative-reasoning sentences), used as-is
    (no length filtering) so the same prompts can be fed to a pretrained and a
    finetuned checkpoint for a directly comparable pre-vs-post-finetune heatmap."""
    sentences = []
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sentences.append(line)
            if len(sentences) >= n:
                break
    return sentences


@torch.no_grad()
def get_attn_for_sentence(model, sp: spm.SentencePieceProcessor, sentence: str, device: str):
    ids = sp.encode(sentence)
    idx = torch.tensor([ids], device=device)
    _, _, attn_by_layer = model(idx, return_attn=True)
    # attn_by_layer[l]: (1, n_heads, T, T)
    # sentencepiece's "▁" word-start marker isn't covered by Devanagari-only fonts
    # (nor is a middle dot); drop it so labels render cleanly without missing glyphs.
    tokens = [sp.id_to_piece(i).replace("▁", "") for i in ids]
    return attn_by_layer, tokens


def plot_heatmaps(attn_by_layer, tokens: list[str], layer_idx: int, lang: str,
                   label: str, out_dir: Path, devanagari_font=None) -> list[Path]:
    """One full-size heatmap PER HEAD (not a packed multi-head grid) - a grid crams
    each head's token labels down to single-digit font sizes, which made them
    unreadable; a dedicated figure per head keeps every tick label legible."""
    attn = attn_by_layer[layer_idx][0]  # (n_heads, T, T)
    n_heads = attn.shape[0]
    n_tokens = len(tokens)
    side = max(5.5, 0.42 * n_tokens + 2.5)  # scale with sentence length so labels stay readable

    written = []
    for h in range(n_heads):
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(attn[h].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        ax.set_xticklabels(tokens, rotation=90, fontsize=11, fontproperties=devanagari_font)
        ax.set_yticklabels(tokens, fontsize=11, fontproperties=devanagari_font)
        ax.set_xlabel("key position", fontsize=10, color=INK_SECONDARY)
        ax.set_ylabel("query position", fontsize=10, color=INK_SECONDARY)
        ax.set_title(f"{lang.capitalize()} — layer {layer_idx} ({label}) — head {h}",
                     fontsize=12, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.85, label="attention weight")

        out_path = out_dir / f"{lang}_attn_layer{layer_idx}_{label}_head{h}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def select_layers(n_layers: int) -> dict[str, list[int]]:
    """2 early + 2 mid + 2 late layer indices (deduplicated, for small n_layers)."""
    early = sorted({0, min(1, n_layers - 1)})
    late = sorted({max(0, n_layers - 2), n_layers - 1})
    mid_center = n_layers // 2
    mid = sorted({max(0, mid_center - 1), min(n_layers - 1, mid_center)})
    return {"early": early, "mid": mid, "late": late}


def entropy_and_distance(attn_by_layer) -> dict:
    """Per (layer, head): mean entropy (nats) and mean attention distance, averaged
    over query positions >= 1 (position 0 trivially attends only to itself)."""
    summary = {}
    for layer_idx, attn in enumerate(attn_by_layer):
        attn = attn[0]  # (n_heads, T, T)
        n_heads, T, _ = attn.shape
        positions = torch.arange(T, dtype=torch.float32, device=attn.device)
        dist = (positions.view(1, T, 1) - positions.view(1, 1, T)).abs()  # (1,T,T)

        for h in range(n_heads):
            row_entropies, row_distances = [], []
            for i in range(1, T):
                p = attn[h, i, :i + 1]
                row_entropies.append(-(p * (p + 1e-12).log()).sum().item())
                row_distances.append((p * dist[0, i, :i + 1]).sum().item())
            summary[f"layer{layer_idx}_head{h}"] = {
                "mean_entropy_nats": float(np.mean(row_entropies)),
                "mean_attention_distance": float(np.mean(row_distances)),
            }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--test-text", type=Path, default=None)
    ap.add_argument("--prompts-file", type=Path, default=None,
                     help="Use these fixed sentences (one per line, no length filtering) "
                          "instead of sampling from --test-text - e.g. comparative-reasoning "
                          "prompts, so a pretrained and a finetuned checkpoint can be "
                          "compared on identical input.")
    ap.add_argument("--n-sentences", type=int, default=5,
                     help="Example sentences used for entropy/distance averaging")
    ap.add_argument("--max-sentence-tokens", type=int, default=20,
                     help="Keep heatmaps readable — short sentences only")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    devanagari_font = find_devanagari_font()

    tokenizer_path = args.tokenizer or (
        REPO_ROOT / args.lang / "tokenizer" / f"{args.lang}_unigram_16k.model")
    test_text_path = args.test_text or (REPO_ROOT / args.lang / "data" / "processed" / "test.txt")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model, ckpt = load_model(args.checkpoint, args.device)
    n_layers = model.cfg.n_layers
    print(f"[{args.lang}] loaded checkpoint from iter {ckpt.get('iter_num')}, "
          f"n_layers={n_layers}", file=sys.stderr)

    if args.prompts_file:
        sentences = get_sentences_from_file(args.prompts_file, args.n_sentences)
        print(f"[{args.lang}] using {len(sentences)} fixed prompts from {args.prompts_file}",
              file=sys.stderr)
    else:
        sentences = get_example_sentences(test_text_path, args.n_sentences,
                                           args.max_sentence_tokens, sp)
        print(f"[{args.lang}] using {len(sentences)} example sentences", file=sys.stderr)

    # Heatmaps: 2 early + 2 mid + 2 late layers, all heads, on the first example sentence.
    attn0, tokens0 = get_attn_for_sentence(model, sp, sentences[0], args.device)
    layer_groups = select_layers(n_layers)
    print(f"[{args.lang}] heatmap layers: {layer_groups}", file=sys.stderr)
    n_heatmap_files = 0
    for group, layer_idxs in layer_groups.items():
        for layer_idx in layer_idxs:
            written = plot_heatmaps(attn0, tokens0, layer_idx, args.lang, group,
                                     args.out_dir, devanagari_font)
            n_heatmap_files += len(written)
    print(f"[{args.lang}] wrote {n_heatmap_files} per-head heatmap images", file=sys.stderr)

    # Entropy / mean distance, averaged over all example sentences.
    per_sentence_summaries = []
    for sentence in sentences:
        attn_by_layer, _ = get_attn_for_sentence(model, sp, sentence, args.device)
        per_sentence_summaries.append(entropy_and_distance(attn_by_layer))

    keys = per_sentence_summaries[0].keys()
    averaged = {
        k: {
            "mean_entropy_nats": float(np.mean([s[k]["mean_entropy_nats"] for s in per_sentence_summaries])),
            "mean_attention_distance": float(np.mean([s[k]["mean_attention_distance"] for s in per_sentence_summaries])),
        }
        for k in keys
    }

    results = {
        "lang": args.lang,
        "checkpoint": str(args.checkpoint),
        "n_layers": n_layers,
        "n_heads": model.cfg.n_heads,
        "heatmap_layers": layer_groups,
        "n_sentences_averaged": len(sentences),
        "example_sentence_for_heatmaps": sentences[0],
        "per_head_summary": averaged,
    }
    out_json = args.out_dir / f"{args.lang}_attention_summary.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.lang}] wrote {out_json} and heatmaps to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
