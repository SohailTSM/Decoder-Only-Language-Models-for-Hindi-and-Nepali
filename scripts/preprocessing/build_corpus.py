"""Orchestrates the full Tier-1 preprocessing pipeline for one language: load manual
(scraped) + public (IndicCorpV2) sources, clean each document, dedupe across the whole
combined corpus, shuffle, split 95/2.5/2.5, and write train/val/test files + stats.

Memory design: this machine has ~2.7GB free RAM, and the public corpus alone is a 4GB+
text file with millions of short documents — loading it whole (`f.read()`) crashed
things once already. Everything here is a generator/streaming pass: documents are read
one at a time, never held in a single big Python list, and each stage writes its
survivors to an intermediate JSONL file on disk rather than accumulating in memory.
Dedup stages keep only hashes (not text) resident in RAM.

Usage: python -m scripts.preprocessing.build_corpus --lang hindi
"""

import argparse
import glob
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing.clean import clean_document, load_blocklist

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_RATIOS = (0.95, 0.025, 0.025)  # train, val, test
SEED = 42
LOG_EVERY = 200_000
# Both kept small deliberately: measured 5.43ms/doc at num_perm=32 with unlimited
# shingles -> ~8.4h projected for the full corpus (~5.6M docs). num_perm=16 + capping
# shingles per doc to 40 (randomly sampled, not just the first 40) brought that to
# ~1.1ms/doc (~1.7h projected) — an approximation tradeoff, but LSH near-dup detection
# is inherently approximate already, and "good enough" duplicate reduction at a runtime
# that actually fits the time budget beats exact-but-infeasible.
MINHASH_NUM_PERM = 16
MINHASH_MAX_SHINGLES = 40

PUBLIC_FILES = {
    "hindi": "indiccorpv2_hi1.txt",
    "nepali": "indiccorpv2_ne.txt",
}


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _hash_text(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def iter_manual_documents(lang: str) -> Iterator[tuple[str, str]]:
    for fp in sorted(glob.glob(str(REPO_ROOT / lang / "data" / "manual" / "*.jsonl"))):
        name = Path(fp).stem
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", "")
                if text:
                    yield f"manual:{name}", text


def iter_public_documents(lang: str, filename: str) -> Iterator[tuple[str, str]]:
    """Streams the file line-by-line; a document is the text between blank lines.
    Never reads the whole file into memory."""
    path = REPO_ROOT / lang / "data" / "raw_public" / filename
    buffer: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if buffer:
                    yield "public:indiccorpv2", "\n".join(buffer)
                    buffer = []
            else:
                buffer.append(line.rstrip("\n"))
        if buffer:
            yield "public:indiccorpv2", "\n".join(buffer)


def iter_all_documents(lang: str) -> Iterator[tuple[str, str]]:
    yield from iter_manual_documents(lang)
    yield from iter_public_documents(lang, PUBLIC_FILES[lang])


def iter_jsonl(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["text"]


def write_jsonl(path: Path, texts: Iterator[str]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            n += 1
    return n


def stage1_clean_and_exact_dedup(lang: str, blocklist: set[str], out_path: Path,
                                  limit: int | None = None) -> dict:
    """Single streaming pass: clean each document, drop exact duplicates immediately
    (hash-only set, not text), write survivors to out_path."""
    import itertools
    seen_hashes: set[str] = set()
    n_in = n_dropped_clean = n_dropped_dup = n_kept = 0
    t0 = time.time()
    doc_iter = iter_all_documents(lang)
    if limit is not None:
        doc_iter = itertools.islice(doc_iter, limit)
    with out_path.open("w", encoding="utf-8") as out_f:
        for source, text in doc_iter:
            n_in += 1
            c = clean_document(text, blocklist)
            if c is None:
                n_dropped_clean += 1
            else:
                h = _hash_text(c)
                if h in seen_hashes:
                    n_dropped_dup += 1
                else:
                    seen_hashes.add(h)
                    out_f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
                    n_kept += 1
            if n_in % LOG_EVERY == 0:
                print(f"[{lang}] stage1: {n_in:,} read, {n_kept:,} kept, "
                      f"{time.time() - t0:.0f}s, rss={_rss_mb():.0f}MB", file=sys.stderr)
    return {"raw_documents": n_in, "dropped_by_cleaning": n_dropped_clean,
            "dropped_exact_dup": n_dropped_dup, "kept": n_kept}


def stage2_paragraph_dedup(in_path: Path, out_path: Path, lang: str,
                            min_paragraph_words: int = 8) -> dict:
    seen_para_hashes: set[str] = set()
    n_in = n_kept = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out_f:
        for text in iter_jsonl(in_path):
            n_in += 1
            kept_paragraphs = []
            for para in text.split("\n\n"):
                stripped = para.strip()
                if not stripped or len(stripped.split()) < min_paragraph_words:
                    kept_paragraphs.append(para)
                    continue
                h = _hash_text(stripped)
                if h not in seen_para_hashes:
                    seen_para_hashes.add(h)
                    kept_paragraphs.append(para)
            cleaned = "\n\n".join(p for p in kept_paragraphs if p.strip())
            if cleaned.strip():
                out_f.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
                n_kept += 1
            if n_in % LOG_EVERY == 0:
                print(f"[{lang}] stage2 (paragraph dedup): {n_in:,} read, {n_kept:,} kept, "
                      f"{time.time() - t0:.0f}s, rss={_rss_mb():.0f}MB", file=sys.stderr)
    return {"in": n_in, "kept": n_kept}


def _shingles(text: str, k: int = 5, max_shingles: int = MINHASH_MAX_SHINGLES) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    all_shingles = [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]
    if len(all_shingles) > max_shingles:
        rng = random.Random(hash(all_shingles[0]))  # deterministic per-doc, cheap
        rng.shuffle(all_shingles)
        all_shingles = all_shingles[:max_shingles]
    return set(all_shingles)


def _compute_minhash_signature(text: str) -> list[int]:
    """Top-level (picklable, for multiprocessing) worker: returns just the digest
    values, not the MinHash object, to keep inter-process payloads small."""
    from datasketch import MinHash
    mh = MinHash(num_perm=MINHASH_NUM_PERM)
    for shingle in _shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh.hashvalues.tolist()


# A live MinHashLSH's memory grows without bound as more documents are inserted (measured:
# ~671 bytes/doc from RSS growth during a real run — 2064MB at 2.4M inserted docs, up from
# 587MB at 200K). At Hindi's ~3.8M-document scale this alone approached 3GB and crashed the
# whole stage with a MemoryError just short of finishing. Fix: periodically start a *fresh*
# LSH index every SHARD_SIZE documents, bounding memory to a fixed ceiling regardless of
# total corpus size. Tradeoff: near-duplicates that happen to fall in different shards won't
# be caught (only within-shard near-dups are detected) — accepted given exact + paragraph
# dedup (stages 1-2) already catch the exact-overlap cases precisely; this stage is on top
# of those, not the only line of defense.
MINHASH_SHARD_SIZE = 800_000  # ~800_000 * 671 bytes ≈ 537MB peak for the LSH index itself


def stage3_minhash_dedup(in_path: Path, out_path: Path, lang: str,
                          threshold: float = 0.8, workers: int | None = None,
                          batch_size: int = 2000,
                          shard_size: int = MINHASH_SHARD_SIZE) -> dict:
    """Signature computation (measured bottleneck: ~1.1ms/doc single-threaded) is
    parallelized across worker processes; LSH insert/query stays single-threaded in
    the main process, since duplicate detection must see documents in a stable order
    to be deterministic. The LSH index itself is reset every `shard_size` documents to
    bound memory (see MINHASH_SHARD_SIZE comment)."""
    import multiprocessing

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    workers = workers or max(1, multiprocessing.cpu_count() - 2)
    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
    shard_inserted = 0
    n_in = n_kept = batch_count = 0
    t0 = time.time()
    log_every_n_batches = max(1, LOG_EVERY // batch_size)

    def batches():
        batch = []
        for text in iter_jsonl(in_path):
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    with out_path.open("w", encoding="utf-8") as out_f, \
         multiprocessing.Pool(workers) as pool:
        for batch in batches():
            signatures = pool.map(_compute_minhash_signature, batch)
            for text, sig in zip(batch, signatures):
                mh = MinHash(num_perm=MINHASH_NUM_PERM,
                              hashvalues=np.array(sig, dtype=np.uint64),
                              scheme="affine32")
                if not lsh.query(mh):
                    lsh.insert(f"doc_{n_in}", mh)
                    shard_inserted += 1
                    out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    n_kept += 1
                    if shard_inserted >= shard_size:
                        lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
                        shard_inserted = 0
                n_in += 1
            batch_count += 1
            if batch_count % log_every_n_batches == 0:
                print(f"[{lang}] stage3 (minhash dedup, {workers} workers): "
                      f"{n_in:,} read, {n_kept:,} kept, "
                      f"{time.time() - t0:.0f}s, rss={_rss_mb():.0f}MB", file=sys.stderr)
    return {"in": n_in, "kept": n_kept}


def stage4_split(in_path: Path, out_dir: Path, lang: str) -> dict:
    """Two-pass, memory-light shuffle+split: holding every surviving document's full
    text in one Python list to shuffle it is the same class of bug that crashed things
    earlier (millions of docs x hundreds-to-thousands of chars each). Instead: pass 1
    counts lines and builds a per-line split assignment (train/val/test) — an array of
    single characters, a few MB even at millions of docs, not the text itself. Pass 2
    streams the input once, routing each line straight to its assigned output file."""
    with in_path.open(encoding="utf-8") as f:
        n = sum(1 for _ in f)

    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    assignment = (["train"] * n_train + ["val"] * n_val
                  + ["test"] * (n - n_train - n_val))
    random.Random(SEED).shuffle(assignment)

    out_paths = {name: out_dir / f"{name}.txt" for name in ("train", "val", "test")}
    out_files = {name: p.open("w", encoding="utf-8") for name, p in out_paths.items()}
    counts = {name: 0 for name in out_files}
    chars = {name: 0 for name in out_files}
    try:
        with in_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                text = json.loads(line)["text"]
                split_name = assignment[i]
                out_files[split_name].write(text.strip() + "\n\n")
                counts[split_name] += 1
                chars[split_name] += len(text)
    finally:
        for fh in out_files.values():
            fh.close()

    split_stats = {}
    for name in out_files:
        split_stats[name] = {"documents": counts[name], "chars": chars[name]}
        print(f"[{lang}] wrote {name}: {counts[name]:,} docs, {chars[name]:,} chars",
              file=sys.stderr)
    return split_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--skip-minhash", action="store_true",
                     help="Skip the MinHash stage (for a quick test run).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only read this many raw documents (for a quick, safe test run).")
    ap.add_argument("--resume-from-stage2", action="store_true",
                     help="Skip stages 1-2 and reuse the existing stage2.jsonl (for "
                          "recovering after a crash partway through stage 3, without "
                          "redoing the earlier, already-completed stages).")
    args = ap.parse_args()
    lang = args.lang

    work_dir = REPO_ROOT / lang / "data" / "_intermediate"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO_ROOT / lang / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    blocklist = load_blocklist(lang)
    stats = {"language": lang}
    s1_path = work_dir / "stage1.jsonl"
    s2_path = work_dir / "stage2.jsonl"

    if args.resume_from_stage2 and s2_path.exists():
        print(f"[{lang}] resuming: reusing existing {s2_path}", file=sys.stderr)
        stats["stage1"] = {"skipped": "resumed"}
        stats["stage2"] = {"skipped": "resumed"}
    else:
        print(f"[{lang}] stage 1: clean + exact dedup...", file=sys.stderr)
        stats["stage1"] = stage1_clean_and_exact_dedup(lang, blocklist, s1_path, limit=args.limit)
        print(f"[{lang}] stage1 done: {stats['stage1']}", file=sys.stderr)

        print(f"[{lang}] stage 2: paragraph-level dedup...", file=sys.stderr)
        stats["stage2"] = stage2_paragraph_dedup(s1_path, s2_path, lang)
        print(f"[{lang}] stage2 done: {stats['stage2']}", file=sys.stderr)

    if args.skip_minhash:
        s3_path = s2_path
        stats["stage3"] = {"skipped": True}
    else:
        print(f"[{lang}] stage 3: MinHash near-duplicate dedup...", file=sys.stderr)
        s3_path = work_dir / "stage3.jsonl"
        stats["stage3"] = stage3_minhash_dedup(s2_path, s3_path, lang)
        print(f"[{lang}] stage3 done: {stats['stage3']}", file=sys.stderr)

    print(f"[{lang}] stage 4: shuffle + split {SPLIT_RATIOS}...", file=sys.stderr)
    stats["splits"] = stage4_split(s3_path, out_dir, lang)

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[{lang}] DONE. Stats written to {out_dir / 'stats.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
