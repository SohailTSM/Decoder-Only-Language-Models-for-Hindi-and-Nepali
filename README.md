[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Language Models and Agents — Individual Project

Model H = **Hindi** (higher-resource). Model L = **Nepali** (lower-resource, from the assignment's
allowed list). Two fully independent monolingual pipelines — no shared data, tokenizer, vocabulary,
or weights.

## Repository layout

```
hindi/           # Model H
  data/          # manual/, raw_public/, processed/ (train/val/test.txt, stats.json)
  tokenizer/     # hindi_unigram_32k.{model,vocab} + vocab_sweep/ (vocab-size selection experiment)
nepali/          # Model L — same structure
scripts/
  scraping/      # manual data collection (sitemap-based news scraping)
  public_data/   # IndicCorpV2 download
  preprocessing/ # cleaning, dedup, splitting pipeline (build_corpus.py)
  tokenizer/     # tokenizer training, evaluation, vocab-size sweep, Colab notebook
report/          # phase1_data_report.md + supporting stats/sweep JSON files
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

## Report

Full Phase 1 report (data collection, non-overlap methodology, preprocessing, tokenizer
construction and vocab-size selection, dataset statistics): [`report/phase1_data_report.md`](report/phase1_data_report.md)

## Reproduction

1. Data collection: `scripts/scraping/launch_all.sh` (manual), `scripts/public_data/download_indiccorpv2.py` (public).
2. Preprocessing: `python -m scripts.preprocessing.build_corpus --lang <hindi|nepali>`.
3. Tokenizer training: `python -m scripts.tokenizer.train_tokenizer --lang <hindi|nepali>` (local,
   memory-constrained sample — see report §6.2), or `scripts/tokenizer/train_tokenizer_colab.ipynb`
   for the full 1M-sentence run used for the final models.
4. Tokenizer evaluation: `python -m scripts.tokenizer.evaluate_tokenizer --lang <hindi|nepali>`.
5. Vocab-size sweep: `python -m scripts.tokenizer.vocab_size_sweep --lang <hindi|nepali>`.
