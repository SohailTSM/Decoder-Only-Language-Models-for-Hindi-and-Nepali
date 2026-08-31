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
nepali/          # Model L — same structure
scripts/
  scraping/      # manual data collection (sitemap-based news scraping)
  public_data/   # IndicCorpV2 download
  preprocessing/ # cleaning, dedup, splitting pipeline (build_corpus.py)
  tokenizer/     # tokenizer training, evaluation, vocab-size sweep, Colab notebook
  training/      # prepare_bin.py, model.py (GPT + RoPE), dataset.py, train.py, kaggle_pretrain.ipynb
  evaluation/    # eval_lm.py (PPL/BPB), generate_eval.py (BLEU/chrF/ROUGE-L + diversity),
                 # attention_analysis.py (heatmaps/entropy/distance), kaggle_evaluate.ipynb
  report/        # make_figures.py — generates report/figures/*.png
report/          # phase1_data_report.md, phase2_model_report.md, architecture_grid_search.md,
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

**⚠️ these repos are currently private** — set to Public (repo Settings on huggingface.co)
before grading, or graders won't be able to open them without requesting access.

## Report

- Full Phase 1 report (data collection, non-overlap methodology, preprocessing, tokenizer
  construction and vocab-size selection, dataset statistics): [`report/phase1_data_report.md`](report/phase1_data_report.md)
- Phase 2 report (model architecture configuration and rationale — pretraining/evaluation
  to follow): [`report/phase2_model_report.md`](report/phase2_model_report.md), with the
  full `d_model`/`n_layers` grid search and depth-vs-width literature review in
  [`report/architecture_grid_search.md`](report/architecture_grid_search.md)

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
