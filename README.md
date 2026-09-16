[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Language Models and Agents — Individual Project

Model H = **Hindi** (higher-resource). Model L = **Nepali** (lower-resource, from the assignment's
allowed list). Two fully independent monolingual pipelines — no shared data, tokenizer, vocabulary,
or weights.

## Repository layout

```
hindi/           # Model H
  data/          # manual/, raw_public/, processed/ (train/val/test.txt, stats.json)
                 # tokenized/ ({train,val,test}.bin + tokenized_meta.json, from prepare_bin.py)
  tokenizer/     # hindi_unigram_16k.{model,vocab} + vocab_sweep/ (vocab-size selection experiment)
  pre_train/     # pretraining logs (hindi_1.log = session 1, hindi_2.log = post-resume session 2)
  eval/          # hindi_lm_metrics.json, hindi_generation.json, attention/ (heatmaps + summary)
                 # attention_reasoning/{pretrained,finetuned}/ (Phase 3 attention comparison)
  reasoning/     # data/{train,val,test}.jsonl + stats.json (synthetic reasoning finetune set)
                 # attention_prompts.txt, eval/hindi_reasoning_metrics.json
                 # checkpoints/ (git-ignored — Phase 3 finetuned best.pt/last.pt, see HF link below)
  eval_ablation/ # hindi_lm_metrics.json, hindi_generation.json, attention_{standard,ablated}/
                 # (Bonus: standard-vs-no-position comparison, Hindi only)
nepali/          # Model L — same structure (no eval_ablation/ — bonus is Hindi-only)
scripts/
  scraping/      # manual data collection (sitemap-based news scraping)
  public_data/   # IndicCorpV2 download
  preprocessing/ # cleaning, dedup, splitting pipeline (build_corpus.py)
  tokenizer/     # tokenizer training, evaluation, vocab-size sweep, Colab notebook
  training/      # prepare_bin.py, model.py (GPT + RoPE + use_rope ablation flag), dataset.py,
                 # train.py (incl. --no-rope), kaggle_pretrain.ipynb
                 # finetune.py, finetune_dataset.py, kaggle_finetune.ipynb (Phase 3 reasoning finetune)
                 # kaggle_ablation_pretrain.ipynb (Bonus: no-positional-embeddings pretrain)
  evaluation/    # eval_lm.py (PPL/BPB), generate_eval.py (BLEU/chrF/ROUGE-L + diversity),
                 # attention_analysis.py (heatmaps/entropy/distance; --prompts-file for Phase 3),
                 # reasoning_eval.py (Phase 3 exact-match accuracy), kaggle_evaluate.ipynb
  reasoning/     # generate_reasoning_data.py — synthetic comparative/transitive reasoning dataset
  report/        # make_figures.py — generates report/figures/*.png
report/          # phase1_data_report.md, phase2_model_report.md, phase3_reasoning_report.md,
                 # bonus_ablation_report.md, architecture_grid_search.md,
                 # supporting stats/sweep JSON files, figures/
```

## Phase 1 — Google Drive links

Large artifacts (the full train/val/test corpora — several GB each) are on Google Drive rather
than committed to git. Small artifacts (tokenizer `.model`/`.vocab` files, stats JSON) are
committed directly in this repo under `<lang>/tokenizer/` and `report/`.

**All Phase 1 data** (both languages, all stages — manual, raw_public, processed, sources.json):
https://drive.google.com/drive/folders/1TuMHT3i1SvDRvKxCBMmP7KsVOxxTSq31?usp=sharing

Direct links to the final `train.txt` / `val.txt` / `test.txt` splits used for tokenizer training:

- **Hindi** (`processed/`): https://drive.google.com/drive/folders/1ZyB2Y7VSXhfWWKFQyTkbiL5agF7rN3Pm
- **Nepali** (`processed/`): https://drive.google.com/drive/folders/1tXaHj3OkzxtjkMJh1lRTHfkeuf5deAYs

Permissions are set to "Anyone with the link can view" — no access request needed.

## Phase 2 — pretrained checkpoints (Hugging Face Hub)

Model checkpoints (`best.pt`, `last.pt` — full training state: model weights, optimizer
state, iteration step, config) are hosted on the Hugging Face Hub rather than Google
Drive, pushed automatically during pretraining (`report/phase2_model_report.md` §5.3):

- **Model H (Hindi)**: https://huggingface.co/sohailtsm/lma-hindi-gpt
- **Model L (Nepali)**: https://huggingface.co/sohailtsm/lma-nepali-gpt

## Phase 3 — finetuned reasoning checkpoints (Hugging Face Hub)

Each language's own Phase 2 checkpoint, finetuned independently on that language's
synthetic reasoning dataset (`report/phase3_reasoning_report.md` §2), same
`best.pt`/`last.pt` full-state format:

- **Model H (Hindi)**: https://huggingface.co/sohailtsm/lma-hindi-gpt-reasoning
- **Model L (Nepali)**: https://huggingface.co/sohailtsm/lma-nepali-gpt-reasoning

## Bonus — no-positional-embeddings ablation (Hugging Face Hub)

Model H (Hindi) retrained fully from scratch with RoPE disabled (`GPTConfig.use_rope=False`,
`train.py --no-rope`) — this project's only positional scheme, so this is a genuine
no-position ablation. Same data/tokenizer/architecture/token budget as the standard
Hindi model otherwise; full Phase 2 evaluation suite compared in
`report/bonus_ablation_report.md`.

- **Model H, no positional embeddings**: https://huggingface.co/sohailtsm/lma-hindi-gpt-noposition
  (`best.pt`/`last.pt` at iteration 20,000 — see the report for why)

## Report

- Full Phase 1 report (data collection, non-overlap methodology, preprocessing, tokenizer
  construction and vocab-size selection, dataset statistics): [`report/phase1_data_report.md`](report/phase1_data_report.md)
- Phase 2 report (model architecture, pretraining, and evaluation — test PPL/BPB,
  generation quality, attention analysis): [`report/phase2_model_report.md`](report/phase2_model_report.md),
  with the full `d_model`/`n_layers` grid search and depth-vs-width literature review in
  [`report/architecture_grid_search.md`](report/architecture_grid_search.md)
- Phase 3 report (synthetic reasoning dataset design, finetuning protocol, pretrained-
  vs-finetuned reasoning accuracy, pretrained-vs-finetuned attention comparison, and
  the final consolidated report): [`report/phase3_reasoning_report.md`](report/phase3_reasoning_report.md)
- Bonus report (no-positional-embeddings ablation setup, full Phase 2 eval suite
  standard-vs-ablated, and what breaks/doesn't without position information):
  [`report/bonus_ablation_report.md`](report/bonus_ablation_report.md)

## Reproduction

1. Data collection: `scripts/scraping/launch_all.sh` (manual), `scripts/public_data/download_indiccorpv2.py` (public).
2. Preprocessing: `python -m scripts.preprocessing.build_corpus --lang <hindi|nepali>`.
3. Tokenizer training: `python -m scripts.tokenizer.train_tokenizer --lang <hindi|nepali>` (local,
   memory-constrained sample — see report §6.2), or `scripts/tokenizer/train_tokenizer_colab.ipynb`
   for the full 1M-sentence run used for the final models.
4. Tokenizer evaluation: `python -m scripts.tokenizer.evaluate_tokenizer --lang <hindi|nepali>`.
5. Vocab-size sweep: `python -m scripts.tokenizer.vocab_size_sweep --lang <hindi|nepali>`.
6. Report figures: `python -m scripts.report.make_figures` (writes `report/figures/*.png`).
7. Tokenize for pretraining: `python -m scripts.training.prepare_bin --lang <hindi|nepali>`
   (writes `{train,val,test}.bin` + `tokenized_meta.json` to `<lang>/data/tokenized/`; paths
   are overridable via `--data-dir`/`--tokenizer`/`--out-dir` for running on Kaggle/Colab).
8. Pretrain: `python -m scripts.training.train --lang <hindi|nepali> --data-dir <dir with
   .bin files> --out-dir <checkpoint dir>` (see `scripts/training/train.py` module docstring for
   Kaggle usage and resuming an interrupted run with `--resume`).
9. End-to-end on Kaggle (GPU T4 x2): [`scripts/training/kaggle_pretrain.ipynb`](scripts/training/kaggle_pretrain.ipynb)
   runs steps 7-8 for both languages in parallel (one per GPU) against the `lma-individual`
   (processed data) and `lma-individual-files` (tokenizer + scripts) Kaggle datasets, logging
   train/val loss to Weights & Biases and pushing checkpoints (including the latest, for
   resuming) to the Hugging Face Hub as it trains — see the notebook's own first cell for
   the exact inputs and Kaggle secrets (`WANDB_API_KEY`, `HF_TOKEN`) it expects.
10. Evaluation (test-set PPL/BPB, generation quality, attention analysis — assignment §2.3):
    [`scripts/evaluation/kaggle_evaluate.ipynb`](scripts/evaluation/kaggle_evaluate.ipynb) runs
    `eval_lm.py`, `generate_eval.py`, and `attention_analysis.py` for both languages against
    the checkpoints on the Hugging Face Hub; results are written to `<lang>/eval/` and
    summarized in `report/phase2_model_report.md` §7.
11. Reasoning dataset generation (assignment §3.1): `python -m
    scripts.reasoning.generate_reasoning_data` (add `--lang <hindi|nepali>` for one
    language only) writes `<lang>/reasoning/data/{train,val,test}.jsonl` +
    `stats.json` — a synthetic, programmatically-generated comparative/transitive
    reasoning dataset per language, with held-out entity names and held-out phrasing
    across splits. Deterministic (fixed seed) — no external inputs needed. Statistics
    summarized in `report/phase3_reasoning_report.md` §1.
12. Reasoning finetuning: `python -m scripts.training.finetune --lang <hindi|nepali>
    --init-checkpoint <path to that language's Phase 2 best.pt> --eval-interval 50`
    (see `scripts/training/finetune.py`'s module docstring for `--resume` usage;
    `--eval-interval 50` matters — the default (once per epoch) finds a visibly worse
    checkpoint at this dataset size, §2.3's "round 1 vs. round 2" finding). Saves
    `{best,last}.pt` in the same resume-capable format as pretraining to
    `<lang>/reasoning/checkpoints/` (git-ignored — see the Phase 3 HF links above), and
    a full training log to `<lang>/reasoning/finetune.log`.
    Protocol and hyperparameters in `report/phase3_reasoning_report.md` §2.
13. Reasoning evaluation: `python -m scripts.evaluation.reasoning_eval --lang
    <hindi|nepali> --label <pretrained|finetuned> --checkpoint <path> --out
    <lang>/reasoning/eval/<lang>_reasoning_metrics.json` — run once per checkpoint
    (results accumulate into the same output file by label). Exact-match accuracy by
    category/answer-form in `report/phase3_reasoning_report.md` §3.
14. Attention comparison, pretrained vs. finetuned (assignment §3.2): `python -m
    scripts.evaluation.attention_analysis --lang <hindi|nepali> --checkpoint <path>
    --prompts-file <lang>/reasoning/attention_prompts.txt --out-dir <dir>` — run once
    per checkpoint against the same fixed prompt file so the two runs are directly
    comparable. Results in `<lang>/eval/attention_reasoning/{pretrained,finetuned}/`,
    discussed in `report/phase3_reasoning_report.md` §4.
15. Kaggle notebook (alternative to steps 12-14 above, if not running locally):
    [`scripts/training/kaggle_finetune.ipynb`](scripts/training/kaggle_finetune.ipynb)
    runs the full finetune -> reasoning-eval -> attention-comparison pipeline for both
    languages on GPU T4 x2 (see the notebook's own first cell for required input
    datasets and secrets).
16. Bonus ablation (assignment "Bonus" section): `python -m scripts.training.train
    --lang hindi --no-rope --data-dir <dir with .bin files> --out-dir <checkpoint dir>`
    (same as pretraining, plus `--no-rope` to disable RoPE entirely — see
    `report/bonus_ablation_report.md` §1 for the exact hyperparameters used). On
    Kaggle: [`scripts/training/kaggle_ablation_pretrain.ipynb`](scripts/training/kaggle_ablation_pretrain.ipynb).
    Evaluation reuses `eval_lm.py`/`generate_eval.py`/`attention_analysis.py`
    unmodified against the resulting checkpoint — no new eval code needed. Results in
    `report/bonus_ablation_report.md`.
