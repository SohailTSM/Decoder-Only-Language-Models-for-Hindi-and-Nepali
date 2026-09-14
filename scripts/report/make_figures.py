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


REASONING_ACCURACY = {
    "hindi": {"pretrained": 0.0000, "finetuned": 0.3629,
              "finetuned_by_form": {"which": 0.2197, "relation": 0.5641}},
    "nepali": {"pretrained": 0.0000, "finetuned": 0.3479,
               "finetuned_by_form": {"which": 0.2318, "relation": 0.5005}},
}

REASONING_ACCURACY_BY_CATEGORY = {
    "hindi": {"direct_numeric": 0.322, "chain_transitive": 0.425, "chain_multihop": 0.341},
    "nepali": {"direct_numeric": 0.322, "chain_transitive": 0.407, "chain_multihop": 0.315},
}


def fig_reasoning_accuracy():
    """Phase 3 §3: exact-match accuracy, pretrained (zero-shot) vs. finetuned, both
    languages (panel 1); finetuned accuracy by category (panel 2); and finetuned
    accuracy by answer form (panel 3) - the most informative split, since "relation"
    (closed 3-word vocabulary) scores much higher than "which" (must reproduce a
    held-out entity name) for both languages."""
    langs = ["Hindi", "Nepali"]
    dark = [HINDI, NEPALI]
    light = [HINDI_LIGHT, NEPALI_LIGHT]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))

    ax = axes[0]
    x = np.arange(len(langs))
    width = 0.32
    pretrained = [REASONING_ACCURACY[l.lower()]["pretrained"] for l in langs]
    finetuned = [REASONING_ACCURACY[l.lower()]["finetuned"] for l in langs]
    ax.bar(x - width / 2, pretrained, width, color=light, label="Pretrained (zero-shot)")
    ax.bar(x + width / 2, finetuned, width, color=dark, label="Finetuned")
    for xi, v in zip(x - width / 2, pretrained):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5, color=INK_SECONDARY)
    for xi, v in zip(x + width / 2, finetuned):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11.5)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 0.54)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("Overall accuracy", fontsize=11.5, color=INK, loc="left", pad=10)

    ax = axes[1]
    cats = ["direct_numeric", "chain_transitive", "chain_multihop"]
    x = np.arange(len(cats))
    for i, (lang, color) in enumerate(zip(langs, dark)):
        vals = [REASONING_ACCURACY_BY_CATEGORY[lang.lower()][c] for c in cats]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, color=color, label=lang)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(["direct\nnumeric", "chain\ntransitive", "chain\nmultihop"], fontsize=9.5)
    ax.set_ylabel("Finetuned accuracy")
    ax.set_ylim(0, 0.52)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2)
    ax.set_title("Finetuned accuracy by category", fontsize=11.5, color=INK,
                 loc="left", pad=10)

    ax = axes[2]
    forms = ["which", "relation"]
    x = np.arange(len(forms))
    for i, (lang, color) in enumerate(zip(langs, dark)):
        vals = [REASONING_ACCURACY[lang.lower()]["finetuned_by_form"][f] for f in forms]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, color=color, label=lang)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(["\"which\" (entity name)", "\"relation\" (3-word)"], fontsize=9.5)
    ax.set_ylabel("Finetuned accuracy")
    ax.set_ylim(0, 0.68)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2)
    ax.set_title("Finetuned accuracy by answer form", fontsize=11.5, color=INK,
                 loc="left", pad=10)

    fig.suptitle("Reasoning test-set exact-match accuracy", fontsize=13, color=INK,
                 x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "10_reasoning_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


REASONING_ATTENTION = {
    # (lang, layer) -> {pretrained, finetuned} mean entropy (nats) / mean distance
    "hindi_early": {"entropy": (2.285, 2.259), "distance": (8.151, 8.098)},
    "hindi_late": {"entropy": (1.665, 1.555), "distance": (9.749, 10.598)},
    "nepali_early": {"entropy": (2.429, 2.409), "distance": (7.896, 7.861)},
    "nepali_late": {"entropy": (1.655, 1.829), "distance": (8.424, 7.266)},
}


def fig_reasoning_attention_comparison():
    """Phase 3 §4: mean entropy and mean attention distance, pretrained vs. finetuned,
    early and late layer, both languages - dumbbell style (mirrors
    fig_manual_ratio_scenarios' light-circle=before/dark-circle=after visual language),
    on the same fixed comparative-reasoning prompts for both checkpoints."""
    rows = ["Hindi - early", "Hindi - late", "Nepali - early", "Nepali - late"]
    keys = ["hindi_early", "hindi_late", "nepali_early", "nepali_late"]
    dark = [HINDI, HINDI, NEPALI, NEPALI]
    light = [HINDI_LIGHT, HINDI_LIGHT, NEPALI_LIGHT, NEPALI_LIGHT]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))

    def _dumbbell(ax, metric, xlabel, fmt):
        for i, k in enumerate(keys):
            pre, fine = REASONING_ATTENTION[k][metric]
            ax.plot([pre, fine], [y[i], y[i]], color=INK_MUTED, linewidth=1.5, zorder=1)
            ax.scatter([pre], [y[i]], color=light[i], s=110, zorder=3,
                       edgecolor=dark[i], linewidth=1.3)
            ax.scatter([fine], [y[i]], color=dark[i], s=110, zorder=3)
            lo, hi = (pre, fine) if pre <= fine else (fine, pre)
            lo_label, hi_label = (fmt.format(pre), fmt.format(fine)) if pre <= fine \
                else (fmt.format(fine), fmt.format(pre))
            ax.text(lo, y[i] - 0.24, lo_label, ha="right", fontsize=8.5,
                    color=INK_SECONDARY)
            ax.text(hi, y[i] + 0.28, hi_label, ha="left", fontsize=8.5,
                    color=INK_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels(rows, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _clean_axes(ax, hide_left=True)

    _dumbbell(axes[0], "entropy", "Mean entropy (nats)", "{:.3f}")
    axes[0].set_title("Entropy: pretrained -> finetuned", fontsize=11, color=INK,
                       loc="left", pad=10)
    _dumbbell(axes[1], "distance", "Mean attention distance", "{:.2f}")
    axes[1].set_title("Distance: pretrained -> finetuned", fontsize=11, color=INK,
                       loc="left", pad=10)

    axes[0].scatter([], [], color=SURFACE, edgecolor=INK_MUTED, linewidth=1.3, s=110,
                     label="Pretrained")
    axes[0].scatter([], [], color=INK_MUTED, s=110, label="Finetuned")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9,
                   bbox_to_anchor=(1.75, -0.32), ncol=2)

    fig.suptitle("Attention entropy/distance on reasoning prompts: pretrained vs. finetuned",
                 fontsize=12.5, color=INK, x=0.01, ha="left", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "11_reasoning_attention_comparison.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


REASONING_DATASET_STATS = {
    # train-split counts only (val/test are proportionally similar, §1.4) - both
    # languages' entity pools also include val/test sizes, shown as stacked segments.
    "entity_pools": {
        "hindi_persons": {"train": 56, "val": 12, "test": 12},
        "hindi_objects": {"train": 35, "val": 8, "test": 7},
        "nepali_persons": {"train": 49, "val": 10, "test": 11},
        "nepali_objects": {"train": 31, "val": 7, "test": 7},
    },
    "category": {
        "hindi": {"direct_numeric": 5309, "chain_transitive": 5271, "chain_multihop": 5420},
        "nepali": {"direct_numeric": 5457, "chain_transitive": 5272, "chain_multihop": 5271},
    },
    "form": {
        "hindi": {"which": 9128, "relation": 6872},
        "nepali": {"which": 9146, "relation": 6854},
    },
    "attribute": {
        "hindi": {"age": 3994, "height": 3984, "price": 4004, "weight": 4018},
        "nepali": {"age": 4029, "height": 3965, "price": 4054, "weight": 3952},
    },
}


def fig_reasoning_dataset_composition():
    """Phase 3 §1.4: one figure covering all four dataset-composition tables - entity
    pool sizes (stacked train/val/test), and the train-split category/form/attribute
    distributions - so every table in §1.4 has a companion visualization."""
    d = REASONING_DATASET_STATS
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))

    # Panel 1: entity pools, stacked train/val/test.
    ax = axes[0]
    pool_keys = ["hindi_persons", "hindi_objects", "nepali_persons", "nepali_objects"]
    pool_labels = ["Hindi\npersons", "Hindi\nobjects", "Nepali\npersons", "Nepali\nobjects"]
    pool_colors = [HINDI, HINDI_LIGHT, NEPALI, NEPALI_LIGHT]
    x = np.arange(len(pool_keys))
    bottoms = np.zeros(len(pool_keys))
    for split, alpha in (("train", 1.0), ("val", 0.65), ("test", 0.4)):
        vals = [d["entity_pools"][k][split] for k in pool_keys]
        ax.bar(x, vals, bottom=bottoms, color=pool_colors, alpha=alpha, width=0.6,
               edgecolor=SURFACE, linewidth=0.6)
        bottoms += np.array(vals)
    for xi, total in zip(x, bottoms):
        ax.text(xi, total + 1.5, f"{int(total)}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(pool_labels, fontsize=8.5)
    ax.set_ylabel("Pool size (names/nouns)")
    ax.set_ylim(0, max(bottoms) * 1.2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Entity pools\n(dark=train, mid=val, light=test)", fontsize=10,
                 color=INK, loc="left", pad=10)

    # Panel 2: category distribution (train).
    ax = axes[1]
    cats = ["direct_numeric", "chain_transitive", "chain_multihop"]
    x = np.arange(len(cats))
    width = 0.32
    for i, lang in enumerate(["hindi", "nepali"]):
        vals = [d["category"][lang][c] for c in cats]
        color = HINDI if lang == "hindi" else NEPALI
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=lang.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(["direct\nnumeric", "chain\ntransitive", "chain\nmultihop"], fontsize=8.5)
    ax.set_ylabel("Train examples")
    ax.set_ylim(0, 6200)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Category distribution\n(train)", fontsize=10, color=INK, loc="left", pad=10)

    # Panel 3: answer-form distribution (train).
    ax = axes[2]
    forms = ["which", "relation"]
    x = np.arange(len(forms))
    for i, lang in enumerate(["hindi", "nepali"]):
        vals = [d["form"][lang][f] for f in forms]
        color = HINDI if lang == "hindi" else NEPALI
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=lang.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(["which", "relation"], fontsize=9.5)
    ax.set_ylabel("Train examples")
    ax.set_ylim(0, 10500)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Answer-form distribution\n(train)", fontsize=10, color=INK, loc="left", pad=10)

    # Panel 4: attribute distribution (train).
    ax = axes[3]
    attrs = ["age", "height", "price", "weight"]
    x = np.arange(len(attrs))
    for i, lang in enumerate(["hindi", "nepali"]):
        vals = [d["attribute"][lang][a] for a in attrs]
        color = HINDI if lang == "hindi" else NEPALI
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=lang.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, fontsize=9.5)
    ax.set_ylabel("Train examples")
    ax.set_ylim(0, 4700)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Attribute distribution\n(train)", fontsize=10, color=INK, loc="left", pad=10)

    fig.suptitle("Reasoning dataset composition", fontsize=13, color=INK, x=0.01,
                 ha="left", y=1.06)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "12_reasoning_dataset_composition.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


TUNING_ROUNDS = {
    "hindi": {
        "round1": {"overall": 0.327, "which": 0.221, "relation": 0.472},
        "round2": {"overall": 0.363, "which": 0.220, "relation": 0.564},
    },
    "nepali": {
        "round1": {"overall": 0.315, "which": 0.176, "relation": 0.512},
        "round2": {"overall": 0.348, "which": 0.232, "relation": 0.500},
    },
}


def fig_tuning_rounds():
    """Phase 3 §2.3: round 1 (4k examples, checked once/epoch) vs. round 2 (16k
    examples, checked every 50 iters) exact-match accuracy - overall and by form,
    both languages - the evidence behind the two tuning changes made."""
    langs = ["hindi", "nepali"]
    metrics = ["overall", "which", "relation"]
    metric_labels = ["Overall", "\"which\"", "\"relation\""]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, lang in zip(axes, langs):
        x = np.arange(len(metrics))
        width = 0.32
        r1 = [TUNING_ROUNDS[lang]["round1"][m] for m in metrics]
        r2 = [TUNING_ROUNDS[lang]["round2"][m] for m in metrics]
        color = HINDI if lang == "hindi" else NEPALI
        light = HINDI_LIGHT if lang == "hindi" else NEPALI_LIGHT
        ax.bar(x - width / 2, r1, width, color=light, label="Round 1 (4k, per-epoch check)")
        ax.bar(x + width / 2, r2, width, color=color, label="Round 2 (16k, every-50-iter check)")
        for xi, v in zip(x - width / 2, r1):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=INK_SECONDARY)
        for xi, v in zip(x + width / 2, r2):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9.5)
        ax.set_ylabel("Exact-match accuracy")
        ax.set_ylim(0, 0.66)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _clean_axes(ax)
        ax.legend(loc="upper center", frameon=False, fontsize=8, ncol=1)
        ax.set_title(lang.capitalize(), fontsize=12, color=INK, loc="left", pad=10)

    fig.suptitle("Finetuning round 1 vs. round 2: dataset size + checkpoint-check granularity",
                 fontsize=12.5, color=INK, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "13_tuning_rounds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_cross_phase_summary():
    """Phase 3 §5.2: the final consolidated cross-phase comparison - test perplexity
    and BPB (Phase 2) alongside reasoning accuracy pretrained/finetuned (Phase 3),
    Hindi vs. Nepali - the numbers behind §5.2's "task-level numbers converge, raw
    perplexity doesn't" argument, in one place."""
    langs = ["Hindi", "Nepali"]
    colors = [HINDI, NEPALI]
    ppl = [27.32, 45.21]
    bpb = [0.4435, 0.4333]
    reasoning = [0.363, 0.348]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    x = np.arange(2)

    ax = axes[0]
    ax.bar(x, ppl, color=colors, width=0.52)
    for xi, v in zip(x, ppl):
        ax.text(xi, v + max(ppl) * 0.02, f"{v:.2f}", ha="center", fontsize=11, color=INK,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Test perplexity")
    ax.set_ylim(0, max(ppl) * 1.28)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Perplexity (Phase 2)", fontsize=11, color=INK, loc="left", pad=10)

    ax = axes[1]
    ax.bar(x, bpb, color=colors, width=0.52)
    for xi, v in zip(x, bpb):
        ax.text(xi, v + max(bpb) * 0.02, f"{v:.4f}", ha="center", fontsize=11, color=INK,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Test BPB")
    ax.set_ylim(0, max(bpb) * 1.28)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Bits-per-byte (Phase 2)", fontsize=11, color=INK, loc="left", pad=10)

    ax = axes[2]
    ax.bar(x, reasoning, color=colors, width=0.52)
    for xi, v in zip(x, reasoning):
        ax.text(xi, v + max(reasoning) * 0.02, f"{v:.2f}", ha="center", fontsize=11,
                color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Finetuned reasoning accuracy")
    ax.set_ylim(0, max(reasoning) * 1.35)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Reasoning accuracy (Phase 3)", fontsize=11, color=INK, loc="left", pad=10)

    fig.suptitle("Cross-phase summary: raw perplexity diverges, task-level metrics converge",
                 fontsize=12.5, color=INK, x=0.01, ha="left", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "14_cross_phase_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


STANDARD = HINDI
ABLATED = "#7d5ba6"
ABLATED_LIGHT = "#c3aede"


def fig_bonus_lm_generation():
    """Bonus §: standard (RoPE) vs. ablated (no positional embeddings) Hindi -
    test PPL/BPB (panel 1-2) and BLEU-4 across decoding settings (panel 3), the
    quantitative core of "what breaks without position information"."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    labels = ["Standard\n(RoPE)", "Ablated\n(no position)"]
    colors = [STANDARD, ABLATED]
    x = np.arange(2)

    ax = axes[0]
    ppl = [27.32, 31.23]
    ax.bar(x, ppl, color=colors, width=0.52)
    for xi, v in zip(x, ppl):
        ax.text(xi, v + max(ppl) * 0.02, f"{v:.2f}", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test perplexity")
    ax.set_ylim(0, max(ppl) * 1.28)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Perplexity", fontsize=11.5, color=INK, loc="left", pad=10)

    ax = axes[1]
    bpb = [0.4435, 0.4614]
    ax.bar(x, bpb, color=colors, width=0.52)
    for xi, v in zip(x, bpb):
        ax.text(xi, v + max(bpb) * 0.02, f"{v:.4f}", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test BPB")
    ax.set_ylim(0, max(bpb) * 1.28)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_title("Bits-per-byte", fontsize=11.5, color=INK, loc="left", pad=10)

    ax = axes[2]
    settings = ["Greedy", "Temp 0.5", "Temp 1.0", "Temp 1.5"]
    xs = np.arange(len(settings))
    standard_bleu = [2.75, 2.25, 0.83, 0.16]
    ablated_bleu = [1.93, 1.96, 1.43, 0.18]
    ax.plot(xs, standard_bleu, color=STANDARD, marker="o", markersize=6, linewidth=2,
            solid_capstyle="round", label="Standard")
    ax.plot(xs, ablated_bleu, color=ABLATED, marker="o", markersize=6, linewidth=2,
            solid_capstyle="round", label="Ablated")
    ax.set_xticks(xs)
    ax.set_xticklabels(settings, fontsize=9, rotation=12)
    ax.set_ylabel("BLEU-4")
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_title("Generation quality (BLEU-4)", fontsize=11.5, color=INK, loc="left", pad=10)

    fig.suptitle("Bonus ablation: standard vs. no-positional-embeddings Hindi",
                 fontsize=13, color=INK, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "15_bonus_lm_generation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


BONUS_ATTENTION = {
    0: {"entropy": (1.236, 1.051), "distance": (1.683, 1.095)},
    1: {"entropy": (1.201, 1.144), "distance": (1.725, 1.602)},
    4: {"entropy": (1.020, 1.309), "distance": (1.805, 2.272)},
    5: {"entropy": (1.111, 1.193), "distance": (2.098, 1.924)},
    9: {"entropy": (1.034, 1.111), "distance": (1.955, 2.151)},
    10: {"entropy": (0.993, 1.027), "distance": (2.184, 2.352)},
}


def fig_bonus_attention_comparison():
    """Bonus §: mean entropy and mean attention distance by layer, standard vs.
    ablated Hindi, on identical held-out sentences - shows the standard model's
    clean early->late (diffuse->focused, near->far) progression becomes markedly
    less monotonic once positional information is removed."""
    layers = [0, 1, 4, 5, 9, 10]
    y = np.arange(len(layers))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))

    def _dumbbell(ax, metric, xlabel, fmt):
        for i, layer in enumerate(layers):
            std_v, abl_v = BONUS_ATTENTION[layer][metric]
            ax.plot([std_v, abl_v], [y[i], y[i]], color=INK_MUTED, linewidth=1.5, zorder=1)
            ax.scatter([std_v], [y[i]], color=STANDARD, s=110, zorder=3)
            ax.scatter([abl_v], [y[i]], color=ABLATED, s=110, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"layer {l}" for l in layers], fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.6, len(layers) - 0.4)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _clean_axes(ax, hide_left=True)

    _dumbbell(axes[0], "entropy", "Mean entropy (nats)", "{:.3f}")
    axes[0].set_title("Entropy by layer", fontsize=11.5, color=INK, loc="left", pad=10)
    _dumbbell(axes[1], "distance", "Mean attention distance", "{:.2f}")
    axes[1].set_title("Distance by layer", fontsize=11.5, color=INK, loc="left", pad=10)

    axes[0].scatter([], [], color=STANDARD, s=110, label="Standard (RoPE)")
    axes[0].scatter([], [], color=ABLATED, s=110, label="Ablated (no position)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9,
                   bbox_to_anchor=(1.85, -0.32), ncol=2)

    fig.suptitle("Bonus ablation: attention entropy/distance by layer, standard vs. ablated",
                 fontsize=12.5, color=INK, x=0.01, ha="left", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "16_bonus_attention_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


BONUS_LAG_RATIO = {
    500: 1.077, 1000: 1.319, 1500: 1.309, 2000: 1.282, 2500: 1.288, 3000: 1.308,
    3500: 1.332, 4000: 1.390, 4500: 1.408, 5000: 1.388, 5500: 1.368, 6000: 1.389,
    6500: 1.442, 7000: 1.434, 7500: 1.402, 8000: 1.486, 8500: 1.411, 9000: 1.455,
    9500: 1.492, 10000: 1.520, 10500: 1.571, 11000: 1.458, 11500: 1.491, 12000: 1.521,
    12500: 1.505, 13000: 1.450, 13500: 1.486, 14000: 1.502, 14500: 1.575, 15000: 1.525,
    15500: 1.651, 16000: 1.591, 16500: 1.595, 17000: 1.509, 17500: 1.563, 18000: 1.731,
    18500: 1.529, 19000: 1.655, 19500: 1.608, 20000: 1.617,
}


def fig_bonus_lag_ratio():
    """Bonus §: at each ablated-model checkpoint, find the (interpolated) iteration
    at which the standard model reached the same val loss, and plot
    ablated_iter / equivalent_standard_iter over training. A flat line would mean
    "the ablated model is simply the standard model run some fixed number of extra
    iterations behind" (i.e. purely a training-duration artifact); a rising line
    means the per-iteration learning penalty from removing position is growing, not
    constant - directly testable evidence against the "it's just undertraining"
    explanation for the perplexity gap."""
    iters = sorted(BONUS_LAG_RATIO)
    ratios = [BONUS_LAG_RATIO[i] for i in iters]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(iters, ratios, color=ABLATED, linewidth=1.3, alpha=0.55, zorder=1)
    ax.scatter(iters, ratios, color=ABLATED, s=22, zorder=2)

    # simple linear trend line to make the direction unambiguous
    n = len(iters)
    mean_x = sum(iters) / n
    mean_y = sum(ratios) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(iters, ratios)) / \
        sum((x - mean_x) ** 2 for x in iters)
    intercept = mean_y - slope * mean_x
    trend = [slope * x + intercept for x in iters]
    ax.plot(iters, trend, color=INK, linewidth=1.8, linestyle=(0, (5, 3)), zorder=3,
            label="linear trend")

    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=(0, (2, 2)))
    ax.text(iters[-1], 1.02, "1.0 = no penalty", ha="right", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xlabel("Ablated model's training iteration")
    ax.set_ylabel("Iterations needed ÷ standard model's\niterations to reach the same loss")
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("Bonus ablation: sample-efficiency penalty grows over training, not constant",
                 fontsize=12, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "17_bonus_lag_ratio.png", dpi=180, bbox_inches="tight")
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
    fig_reasoning_accuracy()
    fig_reasoning_attention_comparison()
    fig_reasoning_dataset_composition()
    fig_tuning_rounds()
    fig_cross_phase_summary()
    fig_bonus_lm_generation()
    fig_bonus_attention_comparison()
    fig_bonus_lag_ratio()
    print(f"wrote 17 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
