"""Generation quality evaluation, per assignment §2.3 "Generation quality":

- Greedy decoding + temperature sampling at {0.5, 1.0, 1.5} from held-out prefixes.
- Corpus-level BLEU-4, chrF++, ROUGE-L against reference continuations from the same
  held-out documents.
- Diversity/fluency diagnostics: repetition rate, Distinct-1, Distinct-2.
- A handful of qualitative examples saved for the report.

Held-out prefixes/references come directly from test.txt: each line is one document
(per prepare_bin.py's convention); the first PREFIX_TOKENS tokens of a document become
the prompt, the next REF_TOKENS tokens become the reference continuation. Documents too
short to supply both are skipped.

Usage:
    python -m scripts.evaluation.generate_eval --lang hindi \
        --checkpoint /path/to/best.pt --out results/hindi_generation.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF

from scripts.evaluation.checkpoint_utils import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]

PREFIX_TOKENS = 32
REF_TOKENS = 64
N_EXAMPLES = 50
N_QUALITATIVE = 5
DECODING_CONFIGS = [
    ("greedy", {"greedy": True}),
    ("temp_0.5", {"greedy": False, "temperature": 0.5}),
    ("temp_1.0", {"greedy": False, "temperature": 1.0}),
    ("temp_1.5", {"greedy": False, "temperature": 1.5}),
]


def build_examples(sp: spm.SentencePieceProcessor, test_text_path: Path, n: int):
    examples = []
    with open(test_text_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids = sp.encode(line)
            if len(ids) < PREFIX_TOKENS + REF_TOKENS:
                continue
            examples.append({
                "prefix_ids": ids[:PREFIX_TOKENS],
                "ref_ids": ids[PREFIX_TOKENS:PREFIX_TOKENS + REF_TOKENS],
            })
            if len(examples) >= n:
                break
    return examples


def distinct_n(token_lists: list[list[int]], n: int) -> float:
    all_ngrams = []
    for tokens in token_lists:
        all_ngrams.extend(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(token_lists: list[list[int]], n: int = 4) -> float:
    total_ngrams = 0
    repeated_ngrams = 0
    for tokens in token_lists:
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        total_ngrams += len(ngrams)
        repeated_ngrams += sum(c - 1 for c in counts.values() if c > 1)
    return repeated_ngrams / total_ngrams if total_ngrams else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None,
                     help="Default: <lang>/tokenizer/<lang>_unigram_16k.model")
    ap.add_argument("--test-text", type=Path, default=None,
                     help="Default: <lang>/data/processed/test.txt")
    ap.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tokenizer_path = args.tokenizer or (
        REPO_ROOT / args.lang / "tokenizer" / f"{args.lang}_unigram_16k.model")
    test_text_path = args.test_text or (REPO_ROOT / args.lang / "data" / "processed" / "test.txt")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model, ckpt = load_model(args.checkpoint, args.device)
    print(f"[{args.lang}] loaded checkpoint from iter {ckpt.get('iter_num')}", file=sys.stderr)

    examples = build_examples(sp, test_text_path, args.n_examples)
    print(f"[{args.lang}] built {len(examples)} prefix/reference examples", file=sys.stderr)
    references_text = [sp.decode(ex["ref_ids"]) for ex in examples]

    bleu = BLEU()
    chrf = CHRF(word_order=2)  # chrF++
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    per_config_results = {}
    qualitative = []

    for name, kwargs in DECODING_CONFIGS:
        gen_token_lists, hyps_text = [], []
        for ex in examples:
            prompt = torch.tensor([ex["prefix_ids"]], device=args.device)
            out = model.generate(prompt, max_new_tokens=REF_TOKENS, **kwargs)
            new_ids = out[0, len(ex["prefix_ids"]):].tolist()
            gen_token_lists.append(new_ids)
            hyps_text.append(sp.decode(new_ids))

        bleu_score = bleu.corpus_score(hyps_text, [references_text]).score
        chrf_score = chrf.corpus_score(hyps_text, [references_text]).score
        rouge_l = sum(rouge.score(ref, hyp)["rougeL"].fmeasure
                      for ref, hyp in zip(references_text, hyps_text)) / len(hyps_text)

        per_config_results[name] = {
            "bleu4": bleu_score,
            "chrf++": chrf_score,
            "rougeL_f1": rouge_l,
            "repetition_rate_4gram": repetition_rate(gen_token_lists, n=4),
            "distinct_1": distinct_n(gen_token_lists, 1),
            "distinct_2": distinct_n(gen_token_lists, 2),
        }
        print(f"[{args.lang}] {name}: {per_config_results[name]}", file=sys.stderr)

        for i in range(min(N_QUALITATIVE, len(examples))):
            entry = next((q for q in qualitative if q["example_id"] == i), None)
            if entry is None:
                entry = {
                    "example_id": i,
                    "prefix": sp.decode(examples[i]["prefix_ids"]),
                    "reference": references_text[i],
                    "generations": {},
                }
                qualitative.append(entry)
            entry["generations"][name] = hyps_text[i]

    results = {
        "lang": args.lang,
        "checkpoint": str(args.checkpoint),
        "n_examples": len(examples),
        "prefix_tokens": PREFIX_TOKENS,
        "ref_tokens": REF_TOKENS,
        "metrics": per_config_results,
        "qualitative_examples": qualitative,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.lang}] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
