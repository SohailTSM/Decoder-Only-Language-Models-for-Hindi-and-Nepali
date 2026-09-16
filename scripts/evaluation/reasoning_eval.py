"""Reasoning evaluation, per assignment §3.1 "Report pretrained vs. finetuned accuracy
(or exact match) on the synthetic test set": greedy-generate a short continuation from
each reasoning test prompt and score it by exact string match against the gold answer.

Run once per checkpoint (pretrained Phase 2 `best.pt` as the zero-shot baseline, and the
finetuned checkpoint) with a different `--label`; results accumulate into the same
`--out` JSON file so the pretrained-vs-finetuned comparison lives in one place.

Usage:
    python -m scripts.evaluation.reasoning_eval --lang hindi --label pretrained \
        --checkpoint /path/to/hindi/best.pt --out results/hindi_reasoning_metrics.json
    python -m scripts.evaluation.reasoning_eval --lang hindi --label finetuned \
        --checkpoint /path/to/hindi/reasoning_best.pt --out results/hindi_reasoning_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

from scripts.evaluation.checkpoint_utils import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_NEW_TOKENS = 12
N_QUALITATIVE = 8


def generate_answer(model, sp: spm.SentencePieceProcessor, prompt_ids: list[int],
                     device: str) -> str:
    prompt = torch.tensor([prompt_ids], device=device)
    out = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, greedy=True)
    new_ids = out[0, len(prompt_ids):].tolist()
    eos_id = sp.eos_id()
    if eos_id in new_ids:
        new_ids = new_ids[:new_ids.index(eos_id)]
    return sp.decode(new_ids).strip()


def evaluate(model, sp, records: list[dict], device: str) -> dict:
    per_category: dict[str, list[int]] = {}
    per_form: dict[str, list[int]] = {}
    qualitative_correct, qualitative_incorrect = [], []
    n_correct = 0

    for rec in records:
        prompt_ids = sp.encode(rec["prompt"])
        prediction = generate_answer(model, sp, prompt_ids, device)
        correct = prediction == rec["answer"]
        n_correct += int(correct)

        per_category.setdefault(rec["category"], [0, 0])
        per_category[rec["category"]][0] += int(correct)
        per_category[rec["category"]][1] += 1
        per_form.setdefault(rec["form"], [0, 0])
        per_form[rec["form"]][0] += int(correct)
        per_form[rec["form"]][1] += 1

        example = {"prompt": rec["prompt"], "gold": rec["answer"], "prediction": prediction}
        if correct and len(qualitative_correct) < N_QUALITATIVE:
            qualitative_correct.append(example)
        elif not correct and len(qualitative_incorrect) < N_QUALITATIVE:
            qualitative_incorrect.append(example)

    n = len(records)
    return {
        "n_examples": n,
        "accuracy": n_correct / n if n else 0.0,
        "accuracy_by_category": {k: v[0] / v[1] for k, v in per_category.items()},
        "accuracy_by_form": {k: v[0] / v[1] for k, v in per_form.items()},
        "qualitative_correct": qualitative_correct,
        "qualitative_incorrect": qualitative_incorrect,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--label", required=True, help="e.g. 'pretrained' or 'finetuned'")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None,
                     help="Default: <lang>/tokenizer/<lang>_unigram_16k.model")
    ap.add_argument("--test-jsonl", type=Path, default=None,
                     help="Default: <lang>/reasoning/data/test.jsonl")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tokenizer_path = args.tokenizer or (
        REPO_ROOT / args.lang / "tokenizer" / f"{args.lang}_unigram_16k.model")
    test_jsonl_path = args.test_jsonl or (REPO_ROOT / args.lang / "reasoning" / "data" / "test.jsonl")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model, ckpt = load_model(args.checkpoint, args.device)
    print(f"[{args.lang}] loaded '{args.label}' checkpoint from {args.checkpoint} "
          f"(iter {ckpt.get('iter_num')})", file=sys.stderr)

    with open(test_jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"[{args.lang}] evaluating {len(records)} test examples", file=sys.stderr)

    result = evaluate(model, sp, records, args.device)
    result["lang"] = args.lang
    result["label"] = args.label
    result["checkpoint"] = str(args.checkpoint)
    print(f"[{args.lang}] {args.label}: accuracy={result['accuracy']:.4f} "
          f"by_category={result['accuracy_by_category']} by_form={result['accuracy_by_form']}",
          file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if args.out.exists():
        all_results = json.loads(args.out.read_text(encoding="utf-8"))
    all_results[args.label] = result
    args.out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.lang}] wrote {args.out} (labels: {list(all_results.keys())})", file=sys.stderr)


if __name__ == "__main__":
    main()
