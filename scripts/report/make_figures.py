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


def fig_test_ppl_bpb():
    """§7.1: test-set perplexity and BPB, Hindi vs Nepali, side by side — the pair
    of panels makes the "PPL says Nepali is much worse, BPB says they're tied"
    finding visible at a glance rather than only readable from the table."""
    langs = ["Hindi", "Nepali"]
    colors = [HINDI, NEPALI]
    ppl = [27.32, 45.21]
    bpb = [0.4435, 0.4333]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))

    ax = axes[0]
    x = np.arange(2)
    ax.bar(x, ppl, color=colors, width=0.52)
    for xi, v in zip(x, ppl):
        ax.text(xi, v + max(ppl) * 0.02, f"{v:.2f}", ha="center", fontsize=11.5,
                 color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11.5)
    ax.set_ylabel("Perplexity (test set)")
    ax.set_ylim(0, max(ppl) * 1.28)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Perplexity", fontsize=12, color=INK, loc="left", pad=10)

    ax = axes[1]
    ax.bar(x, bpb, color=colors, width=0.52)
    for xi, v in zip(x, bpb):
        ax.text(xi, v + max(bpb) * 0.02, f"{v:.4f}", ha="center", fontsize=11.5,
                 color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11.5)
    ax.set_ylabel("Bits-per-byte (test set)")
    ax.set_ylim(0, max(bpb) * 1.28)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("BPB", fontsize=12, color=INK, loc="left", pad=10)

    fig.suptitle("Test-set perplexity vs. bits-per-byte: PPL vs. BPB tell different "
                  "stories", fontsize=12.5, color=INK, x=0.02, ha="left", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_test_ppl_bpb.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


DECODING_SETTINGS = ["Greedy", "Temp 0.5", "Temp 1.0", "Temp 1.5"]

GENERATION_METRICS = {
    "hindi": {
        "bleu4": [2.75, 2.25, 0.83, 0.16],
        "chrf": [16.87, 19.43, 17.99, 15.17],
        "rougeL": [0.010, 0.000, 0.007, 0.000],
    },
    "nepali": {
        "bleu4": [0.63, 0.61, 0.29, 0.09],
        "chrf": [13.67, 16.29, 16.36, 15.58],
        "rougeL": [0.000, 0.000, 0.0003, 0.000],
    },
}

DIVERSITY_METRICS = {
    "hindi": {
        "repetition": [0.330, 0.093, 0.004, 0.000],
        "distinct1": [0.159, 0.253, 0.422, 0.680],
        "distinct2": [0.347, 0.617, 0.874, 0.984],
    },
    "nepali": {
        "repetition": [0.431, 0.057, 0.0003, 0.000],
        "distinct1": [0.108, 0.318, 0.531, 0.721],
        "distinct2": [0.185, 0.651, 0.924, 0.993],
    },
}


def fig_generation_quality():
    """§7.2: BLEU-4 / chrF++ / ROUGE-L across the 4 decoding settings, one panel
    per metric so each keeps its own natural scale (BLEU/chrF are 0-20ish,
    ROUGE-L is near-zero throughout)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    x = np.arange(len(DECODING_SETTINGS))

    panels = [("bleu4", "BLEU-4"), ("chrf", "chrF++"), ("rougeL", "ROUGE-L (F1)")]
    for ax, (key, title) in zip(axes, panels):
        for lang, color in [("hindi", HINDI), ("nepali", NEPALI)]:
            vals = GENERATION_METRICS[lang][key]
            ax.plot(x, vals, color=color, marker="o", markersize=6, linewidth=2,
                     solid_capstyle="round", label=lang.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(DECODING_SETTINGS, fontsize=9.5, rotation=12)
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _clean_axes(ax)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10)

    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    fig.suptitle("Generation quality vs. decoding setting", fontsize=13, color=INK,
                 x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_generation_quality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_generation_diversity():
    """§7.2: repetition rate and Distinct-1/2 across decoding settings — the
    coherence/diversity tradeoff that motivates using temperature at all."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    x = np.arange(len(DECODING_SETTINGS))

    ax = axes[0]
    for lang, color in [("hindi", HINDI), ("nepali", NEPALI)]:
        vals = DIVERSITY_METRICS[lang]["repetition"]
        ax.plot(x, vals, color=color, marker="o", markersize=6, linewidth=2,
                 solid_capstyle="round", label=lang.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(DECODING_SETTINGS, fontsize=9.5, rotation=12)
    ax.set_ylabel("Repetition rate (4-gram)")
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Repetition rate", fontsize=11.5, color=INK, loc="left", pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    ax = axes[1]
    for lang, color, light in [("hindi", HINDI, HINDI_LIGHT),
                                ("nepali", NEPALI, NEPALI_LIGHT)]:
        ax.plot(x, DIVERSITY_METRICS[lang]["distinct1"], color=color, marker="o",
                 markersize=6, linewidth=2, solid_capstyle="round",
                 label=f"{lang.capitalize()} distinct-1")
        ax.plot(x, DIVERSITY_METRICS[lang]["distinct2"], color=color, marker="o",
                 markersize=6, linewidth=2, linestyle=(0, (4, 2)),
                 solid_capstyle="round", label=f"{lang.capitalize()} distinct-2")
    ax.set_xticks(x)
    ax.set_xticklabels(DECODING_SETTINGS, fontsize=9.5, rotation=12)
    ax.set_ylabel("Distinct-n ratio")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Distinct-1 / Distinct-2", fontsize=11.5, color=INK, loc="left",
                 pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    fig.suptitle("Diversity/repetition diagnostics vs. decoding setting", fontsize=13,
                 color=INK, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_generation_diversity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_attention_entropy_distance():
    """§7.3: mean entropy and mean attention distance, early vs. late layer,
    dumbbell style (mirrors fig_manual_ratio_scenarios' visual language) so the
    early->late shift is the thing the eye follows, not just two more bars."""
    langs = ["Hindi", "Nepali"]
    entropy_early = [1.236, 1.743]
    entropy_late = [0.993, 1.073]
    dist_early = [1.683, 3.330]
    dist_late = [2.184, 4.697]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    dark = [HINDI, NEPALI]
    light = [HINDI_LIGHT, NEPALI_LIGHT]
    y = np.arange(len(langs))

    def _dumbbell(ax, early, late, xlabel, fmt):
        for i in range(len(langs)):
            ax.plot([early[i], late[i]], [y[i], y[i]], color=INK_MUTED,
                     linewidth=1.5, zorder=1)
            ax.scatter([early[i]], [y[i]], color=light[i], s=110, zorder=3,
                        edgecolor=dark[i], linewidth=1.3)
            ax.scatter([late[i]], [y[i]], color=dark[i], s=110, zorder=3)
            ax.text(early[i], y[i] - 0.22, fmt.format(early[i]), ha="center",
                     fontsize=9, color=INK_SECONDARY)
            ax.text(late[i], y[i] + 0.26, fmt.format(late[i]), ha="center",
                     fontsize=9, color=INK_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels(langs, fontsize=11.5)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.6, len(langs) - 0.4)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _clean_axes(ax, hide_left=True)

    _dumbbell(axes[0], entropy_early, entropy_late,
               "Mean entropy (nats)", "{:.3f}")
    axes[0].set_title("Entropy: early -> late layer", fontsize=11.5, color=INK,
                       loc="left", pad=10)
    _dumbbell(axes[1], dist_early, dist_late,
               "Mean attention distance", "{:.2f}")
    axes[1].set_title("Distance: early -> late layer", fontsize=11.5, color=INK,
                       loc="left", pad=10)

    axes[0].scatter([], [], color=SURFACE, edgecolor=INK_MUTED, linewidth=1.3,
                     s=110, label="Early layer (0)")
    axes[0].scatter([], [], color=INK_MUTED, s=110, label="Late layer (10)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9,
                    bbox_to_anchor=(1.65, -0.42), ncol=2)

    fig.suptitle("Attention entropy and distance, early vs. late layer",
                 fontsize=13, color=INK, x=0.01, ha="left", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_attention_entropy_distance.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_manual_sources()
    fig_near_dup_rate()
    fig_vocab_fertility()
    fig_manual_ratio_scenarios()
    fig_pretrain_loss()
    fig_test_ppl_bpb()
    fig_generation_quality()
    fig_generation_diversity()
    fig_attention_entropy_distance()
    print(f"wrote 9 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
