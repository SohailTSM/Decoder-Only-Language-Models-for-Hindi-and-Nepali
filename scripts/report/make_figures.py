"""Generate the static figures embedded in report/phase1_data_report.md and
report/phase2_model_report.md.

Uses a fixed, colorblind-validated categorical palette (blue=Hindi, orange=Nepali,
kept consistent across every figure so color always identifies the same language).

Usage: python -m scripts.report.make_figures
"""

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures"

# Palette (validated categorical slots 1/2, plus light-mode chart chrome).
HINDI = "#2a78d6"
HINDI_LIGHT = "#6da7ec"
NEPALI = "#eb6834"
NEPALI_LIGHT = "#f6bba4"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _clean_axes(ax, hide_left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


def fig_manual_sources():
    """4.4: per-source contribution, one panel per language (different metric
    available per language: articles for Hindi, characters for Nepali)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Hindi: article counts (chars weren't logged per-source).
    hindi_sources = ["abplive_archive", "abplive", "bhaskar", "jagran",
                      "navbharattimes", "amarujala", "prabhatkhabar"]
    hindi_vals = [178839, 12386, 3520, 2614, 1041, 820, 768]
    ax = axes[0]
    y = np.arange(len(hindi_sources))
    ax.barh(y, hindi_vals, color=HINDI, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(hindi_sources, fontsize=10)
    ax.invert_yaxis()
    for yi, v in zip(y, hindi_vals):
        ax.text(v + max(hindi_vals) * 0.015, yi, f"{v:,}", va="center",
                 fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlim(0, max(hindi_vals) * 1.22)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.set_xlabel("Articles")
    ax.set_title("Hindi manual sources", fontsize=12, color=INK, loc="left", pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)

    # Nepali: characters (the metric actually available per source).
    nepali_sources = ["khabarhub", "nepalpress", "annapurnapost", "deshsanchar",
                       "nagariknews"]
    nepali_vals = [228_577_365, 198_119_929, 81_980_046, 43_300_650, 0]
    ax = axes[1]
    y = np.arange(len(nepali_sources))
    ax.barh(y, nepali_vals, color=NEPALI, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(nepali_sources, fontsize=10)
    ax.invert_yaxis()
    for yi, v in zip(y, nepali_vals):
        label = f"{v / 1e6:.1f}M" if v else "0"
        ax.text(v + max(nepali_vals) * 0.015, yi, label, va="center",
                 fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlim(0, max(nepali_vals) * 1.22)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.0f}M")
    ax.set_xlabel("Characters")
    ax.set_title("Nepali manual sources", fontsize=12, color=INK, loc="left", pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)

    fig.suptitle("", fontsize=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_manual_sources.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_near_dup_rate():
    """5.7: final v3 MinHash near-duplicate rate, Hindi vs Nepali."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    langs = ["Hindi", "Nepali"]
    vals = [0.55, 4.58]
    colors = [HINDI, NEPALI]
    x = np.arange(2)
    bars = ax.bar(x, vals, color=colors, width=0.52)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.12, f"{v:.2f}%", ha="center", fontsize=12,
                 color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11.5)
    ax.set_ylabel("Near-duplicate rate (stage 3)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("MinHash near-duplicate rate by language", fontsize=12.5,
                 color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_near_dup_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_vocab_fertility():
    """6.1: fertility vs. vocab size sweep, both languages, 16k marked."""
    vocab = [4000, 8000, 16000, 24000, 32000, 40000, 48000]
    hindi = [1.4516, 1.3017, 1.2099, 1.1766, 1.1598, 1.1500, 1.1438]
    nepali = [1.7126, 1.4937, 1.3480, 1.2879, 1.2537, 1.2312, 1.2155]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(vocab, hindi, color=HINDI, marker="o", markersize=5.5, linewidth=2,
            solid_capstyle="round")
    ax.plot(vocab, nepali, color=NEPALI, marker="o", markersize=5.5, linewidth=2,
            solid_capstyle="round")

    ax.set_ylim(1.08, 1.78)
    ax.axvline(16000, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(16000, 1.10, "chosen: 16k", ha="center", fontsize=9.5,
            color=INK_SECONDARY)

    ax.text(vocab[-1] + 900, hindi[-1], "Hindi", color=HINDI, fontsize=11,
            fontweight="bold", va="center")
    ax.text(vocab[-1] + 900, nepali[-1], "Nepali", color=NEPALI, fontsize=11,
            fontweight="bold", va="center")

    ax.set_xticks(vocab)
    ax.set_xticklabels([f"{v // 1000}k" for v in vocab], fontsize=10)
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("Fertility (tokens / word)")
    ax.set_xlim(2500, 54500)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Tokenizer fertility vs. vocab size", fontsize=13, color=INK,
                 loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_vocab_fertility.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_manual_ratio_scenarios():
    """7: manual-token % under the actual (520M public target) vs. a
    counterfactual (-50M smaller public target) scenario, dumbbell style,
    with the 20% requirement marked."""
    langs = ["Hindi", "Nepali"]
    actual = [20.00, 19.08]
    counterfactual = [21.66, 20.70]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(langs))

    dark = [HINDI, NEPALI]
    light = [HINDI_LIGHT, NEPALI_LIGHT]

    for i in range(len(langs)):
        ax.plot([actual[i], counterfactual[i]], [y[i], y[i]], color=INK_MUTED,
                linewidth=1.5, zorder=1)
        ax.scatter([actual[i]], [y[i]], color=dark[i], s=110, zorder=3)
        ax.scatter([counterfactual[i]], [y[i]], color=light[i], s=110,
                   zorder=3, edgecolor=dark[i], linewidth=1.3)
        ax.text(actual[i], y[i] - 0.18, f"{actual[i]:.2f}%", ha="center",
                fontsize=9.5, color=INK_SECONDARY)
        ax.text(counterfactual[i], y[i] + 0.24, f"{counterfactual[i]:.2f}%",
                ha="center", fontsize=9.5, color=INK_SECONDARY)

    # Neutral legend proxies: the legend communicates marker STYLE
    # (solid vs. ringed), not hue — hue is reserved for language identity.
    ax.scatter([], [], color=INK_MUTED, s=110, label="Actual (520M public target)")
    ax.scatter([], [], color=SURFACE, edgecolor=INK_MUTED, linewidth=1.3, s=110,
               label="Counterfactual (− 50M public)")

    ax.axvline(20.0, color="#d03b3b", linewidth=1.3, linestyle=(0, (4, 3)))
    ax.text(20.0, len(langs) - 0.42, "20% requirement", color="#d03b3b",
            fontsize=9.5, ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(langs, fontsize=12)
    ax.set_xlabel("Manual share of total tokens")
    ax.set_xlim(17.5, 23.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(langs) - 0.4)
    _clean_axes(ax, hide_left=True)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5,
              bbox_to_anchor=(1.0, -0.38), ncol=2)
    ax.set_title("Manual-token share: actual vs. a smaller public budget",
                 fontsize=12.5, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_manual_ratio_scenarios.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


EVAL_RE = re.compile(r"iter (\d+): train_loss=([\d.]+) val_loss=([\d.]+)")
RESUME_ITER = 19_000  # iter both languages' Kaggle sessions resumed from after the 12h cap


def _parse_pretrain_log(lang: str) -> dict[int, tuple[float, float]]:
    """Reads pre_train/<lang>_1.log + _2.log (session 1, session 2), keyed by iter —
    later files overwrite earlier ones so the post-resume re-print of iter 19000
    doesn't double up."""
    events: dict[int, tuple[float, float]] = {}
    for part in ("1", "2"):
        path = REPO_ROOT / lang / "pre_train" / f"{lang}_{part}.log"
        for m in EVAL_RE.finditer(path.read_text()):
            events[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return events


def fig_pretrain_loss():
    """§6: validation loss over the full pretraining run, both languages, with the
    Kaggle 12-hour session boundary (auto-resumed from HF Hub) marked."""
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for lang, color in [("hindi", HINDI), ("nepali", NEPALI)]:
        events = _parse_pretrain_log(lang)
        iters = sorted(events)
        val_loss = [events[i][1] for i in iters]
        ax.plot(iters, val_loss, color=color, linewidth=2, solid_capstyle="round",
                 label=lang.capitalize())

    ax.axvline(RESUME_ITER, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(RESUME_ITER, 9.2, "session resumed\n(12h Kaggle cap)", ha="center",
            fontsize=9, color=INK_SECONDARY)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("val/loss (cross-entropy, nats)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.set_title("Pretraining validation loss", fontsize=13, color=INK, loc="left",
                 pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_pretrain_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_manual_sources()
    fig_near_dup_rate()
    fig_vocab_fertility()
    fig_manual_ratio_scenarios()
    fig_pretrain_loss()
    print(f"wrote 5 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
