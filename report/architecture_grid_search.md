# Architecture grid search — targeting ~25M parameters

Sweeps `vocab_size × d_model`, solving for the number of layers that lands closest to
25,000,000 total trainable parameters at each combination. `n_heads` and `d_ff` are
derived, not swept independently.

## Fixed assumptions (carried over from the rest of this project)

- **Tied input/output embedding** — the embedding table is counted once, not twice.
- **Non-learned positional encoding** — this grid was computed under sinusoidal
  encoding; the project's final choice is RoPE (see `phase2_model_report.md` §2.4), but
  both are fixed (not learned) schemes that contribute **0** parameters regardless of
  context length, so every number below is unaffected by that later change. This is why
  `T` (context length) doesn't appear as a variable in the parameter formula below at
  all — it's a free choice, not something to solve for. `T = 512` is used uniformly
  across every row for consistency with the rest of the project; any other value would
  give an identical parameter count.
- **`d_k = 64`** (the classic Transformer head dimension) fixed → `n_heads = d_model / 64`.
  All three `d_model` values divide evenly (384→6, 512→8, 768→12).
- **`d_ff = 4 × d_model`**, as specified.
- **`nn.Linear` biases included** in every projection (Wq/Wk/Wv/Wo, both FFN layers) and
  every LayerNorm's `(γ, β)` — this project doesn't turn biases off.
- Pre-norm placement: 2 LayerNorms per block (`LN1`, `LN2`) + 1 final LayerNorm.

## Parameter formula

```
per_block(d)  = 4·(d² + d)                    # Wq, Wk, Wv, Wo
              + (d·d_ff + d_ff) + (d_ff·d + d) # FFN up + down, d_ff = 4d
              + 2·(2·d)                        # LN1 + LN2

total(V, d, N) = V·d              # tied embedding
               + N · per_block(d)  # N transformer blocks
               + 2·d               # final LayerNorm
```

`per_block(d)` simplifies to `12d² + 13d` when `d_ff = 4d`.

## Results

| Vocab (V) | d_model | n_heads | **n_layers (solved)** |  d_ff |   T | Total params | Δ vs 25M | Embedding share | d_model/n_layers |
| --------: | ------: | ------: | --------------------: | ----: | --: | -----------: | -------: | --------------: | ---------------: |
|     8,000 |     384 |       6 |                **12** | 1,536 | 512 |   24,366,336 |   −2.53% |          12.61% |             32.0 |
|    16,000 |     384 |       6 |                **11** | 1,536 | 512 |   25,663,872 |   +2.66% |          23.94% |             34.9 |
|    24,000 |     384 |       6 |                 **9** | 1,536 | 512 |   25,186,944 |   +0.75% |          36.59% |             42.7 |
|     8,000 |     512 |       8 |                 **7** | 2,048 | 512 |   26,163,712 |   +4.65% |          15.66% |             73.1 |
|    16,000 |     512 |       8 |                 **5** | 2,048 | 512 |   23,954,944 |   −4.18% |          34.20% |            102.4 |
|    24,000 |     512 |       8 |                 **4** | 2,048 | 512 |   24,898,560 |   −0.41% |          49.35% |        **128.0** |
|     8,000 |     768 |      12 |                 **3** | 3,072 | 512 |   27,409,152 |   +9.64% |          22.42% |            256.0 |
|    16,000 |     768 |      12 |                 **2** | 3,072 | 512 |   26,465,280 |   +5.86% |          46.43% |            384.0 |
|    24,000 |     768 |      12 |                 **1** | 3,072 | 512 |   25,521,408 |   +2.09% |          72.22% |            768.0 |

(`n_layers` is the integer value whose total parameter count is closest to 25M for that
`(V, d_model)` pair — both the chosen value and its neighbor were checked; the neighbor
was always farther from target.)

## Literature: is there a "right" depth/width ratio?

**Short answer: no rigid law, but a converged practitioner norm exists, and it's `d_model /
n_layers ≈ 64–128`.**

**Kaplan et al. 2020, "Scaling Laws for Neural Language Models"
([arXiv:2001.08361](https://arxiv.org/abs/2001.08361))** — the foundational scaling-laws
paper — states this directly: _"Transformer performance depends very weakly on the shape
parameters n_layer, n_heads, and d_ff when we hold the total non-embedding parameter count
N fixed."_ Their own formula for non-embedding params is `N ≈ 12 · n_layer · d_model²` —
the same shape as this document's `per_block(d) ≈ 12d²` for large `d`. They tested aspect
ratios (`d_model/n_layer`) from **33 to 715** — a >20x spread — and found losses within 3%
of each other. At fixed parameter count, shape is a genuinely low-stakes choice.

That said, real published models don't scatter randomly across that wide band — they
cluster tightly:

| Model             |    Params | n_layers |    d_model |     d_model/n_layers | head_dim |
| ----------------- | --------: | -------: | ---------: | -------------------: | -------: |
| GPT-2 Small       |      124M |       12 |        768 |                 64.0 |       64 |
| GPT-2 XL          |      1.5B |       48 |       1600 |                 33.3 |       64 |
| GPT-3 Small       |      125M |       12 |        768 |                 64.0 |       64 |
| GPT-3 6.7B – 175B | 6.7B–175B |    32–96 | 4096–12288 | **128.0** (constant) |      128 |
| LLaMA-65B         |     65.2B |       80 |       8192 |                102.4 |      128 |
| Chinchilla-70B    |       70B |       80 |       8192 |                102.4 |      128 |
| PaLM-540B         |      540B |      118 |      18432 |                156.2 |      256 |

(GPT-2: Radford et al. 2019. GPT-3 Table 2.1: Brown et al. 2020,
[arXiv:2005.14165](https://arxiv.org/abs/2005.14165). LLaMA Table 2: Touvron et al. 2023,
[arXiv:2302.13971](https://arxiv.org/abs/2302.13971). Chinchilla: Hoffmann et al. 2022,
[arXiv:2203.15556](https://arxiv.org/abs/2203.15556). PaLM Table 1: Chowdhery et al. 2022,
[arXiv:2204.02311](https://arxiv.org/abs/2204.02311).)

The ratio **converges to exactly 128** for every GPT-3 model from 6.7B params up, and
LLaMA-65B / Chinchilla-70B — trained years later, by a different lab — land on the
_identical_ 102.4 ratio with the _identical_ head_dim=128. This convergence is documented
as the informal "standard recipe" in kipp.ly's widely-cited
["Transformer Taxonomy"](https://kipp.ly/transformer-taxonomy/): pick `head_dim` (64 or
128), then scale `n_layer` and `d_model` together so `d_model/n_layer` lands in that band.

**Two caveats from papers that studied depth/width directly, not just as a side note:**

- **Levine et al. 2020, "The Depth-to-Width Interplay in Self-Attention"
  ([arXiv:2006.12467](https://arxiv.org/abs/2006.12467))** derive a theoretical
  depth-efficiency threshold and argue GPT-3 models up to ~13B were actually **too
  shallow** for their size — their computed optimum for a 125M-param model is **23
  layers × 555 width** (ratio ≈ 24), notably deeper/narrower than GPT-3-Small's actual
  12×768. They also argue GPT-3-175B (96×12288) was, conversely, too deep relative to its
  width.
- **Tay et al. 2021, "Scale Efficiently" ([arXiv:2109.10686](https://arxiv.org/abs/2109.10686))**
  — a 200+ configuration sweep — recommend increasing depth _before_ other knobs (it has
  the best Pareto-efficiency return), but find that return **flattens out past 32-36
  layers** regardless of total model size.

**Applied to this grid:** row `24,000 / 512 / 4 layers` hits the modern 128.0 ratio
exactly, but at only 4 layers it's far shallower than even Levine et al.'s _already
too-shallow_ verdict on 12-layer GPT-3-Small — a 4-layer network has very little room for
the compositional depth a "decoder-only Transformer" assignment is meant to exercise. The
`d_model=384` rows (ratios 32.0-42.7) sit closer to GPT-2-XL's own empirically-shipped
33.3 than to the modern large-model convention, but that's expected — the large-model
128-ratio convention emerged at 1B+ scale; Kaplan's own 3%-loss-difference finding means
this project's 25M scale has considerable freedom regardless of which convention it
follows. The practical takeaway is depth, not ratio-matching: prefer rows with more
layers (9-12, the `d_model=384` column) over rows that satisfy a ratio target by
collapsing to 1-4 layers.

## Observations

- **Depth collapses fast as `d_model` grows.** Because `per_block` scales
  quadratically in `d_model` (`12d²`) while the embedding only scales linearly
  (`V·d`), hitting ~25M at `d_model=768` leaves room for only 1-3 layers. `24,000 /
768 / 1 layer` is numerically the _tightest_ fit in isolation (+2.09%) but isn't a
  meaningful "deep" Transformer — a single block has no real depth to speak of, and
  most of its capacity (72.2%) is just the embedding table, not the attention/FFN
  stack the assignment is actually about.
- **Embedding share climbs with vocab, falls with `d_model`** — the same
  relationship established in the report's own vocab-size analysis (§6.1). At
  `d_model=384`, embedding share ranges a healthy 12.6%→36.6% across the vocab
  sweep; at `d_model=768` it ranges 22.4%→72.2%, mainly because so few layers fit
  in the budget once `d_model` is that large.
- **`d_model=384` gives the best balance of reasonable depth (9-12 layers) and
  healthy embedding share (12.6-36.6%)** across all three vocab sizes — every row
  in that block is architecturally sane, unlike several rows at 512/768 that trade
  away almost all depth to hit the parameter target.
- **This project's actual Phase 1/2 choice** — `vocab=16,000, d_model=384,
n_layers=11` (25,663,872 params, 23.94% embedding share) — sits in that healthy
  region: +2.66% over target, comparable depth to the other `d_model=384` rows, and
  an embedding share close to the ~30-35% range seen in reference small
  decoder-only LMs (GPT-2-small, GPT-Neo-125M, etc., per the earlier research pass).
