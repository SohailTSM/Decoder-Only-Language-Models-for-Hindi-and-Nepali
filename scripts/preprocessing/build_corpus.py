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

# Final-corpus token cap: both languages landed comfortably over the ~500M-token target
# after preprocessing survival rates turned out better than the pre-run estimate (see
# report/phase1_data_report.md) — Hindi ~520.5M, Nepali ~589.8M. Capped to ~510M tokens
# (a little above 500M, not razor-thin against it) using the same chars-per-token
# estimates as the rest of the pipeline.
TARGET_TOKENS = 510_000_000
CHARS_PER_TOKEN = {"hindi": 4.0, "nepali": 4.5}


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
    """Bottom-k sampling by shingle hash, NOT a per-document random sample. This matters:
    an earlier version seeded the sample from hash(first_shingle), so two documents that
    are near-identical except for an edit near the *start* got completely different
    random subsets and the near-duplicate went undetected (caught by a synthetic-edit
    test). Selecting the k shingles with the smallest hash value instead depends only on
    each shingle's own content, not the document's position/context, so documents
    sharing most of their shingles will independently select mostly the same subset."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    all_shingles = list({" ".join(words[i:i + k]) for i in range(len(words) - k + 1)})
    if len(all_shingles) > max_shingles:
        all_shingles.sort(key=hash)
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


# --- History of this stage, kept because both attempts taught something real ---
# v1: a single live MinHashLSH grew without bound as documents were inserted (measured
# ~671 bytes/doc) and crashed with a MemoryError near the end of a full run.
# v2 (sharded): reset the LSH index every 800K documents to bound memory. This worked
# (no crash) but only catches near-dups *within* the same 800K-doc shard — with ~6
# shards, a true near-duplicate pair had only a ~1-in-6 chance of landing in the same
# shard, so most cross-shard near-dups (e.g. different outlets covering the same event,
# scattered across the corpus by source) were silently missed.
# v3 (this version): proper LSH banding via signatures-on-disk + sort-based bucketing +
# union-find, giving full corpus-wide coverage with *better* memory bounds than v1, by
# never holding a live growing Python object graph — just compact numpy arrays.
MINHASH_NUM_BANDS = 4  # with num_perm=16 -> 4 rows/band -> threshold ~= (1/4)^(1/4) ~= 0.71


def stage3_minhash_dedup(in_path: Path, out_path: Path, lang: str,
                          workers: int | None = None, batch_size: int = 2000,
                          num_bands: int = MINHASH_NUM_BANDS) -> dict:
    """Pass 1: compute a MinHash signature per document (parallelized across worker
    processes — this is the measured bottleneck, ~1.1ms/doc single-threaded) and store
    them as one compact numpy array (N x num_perm uint64, ~128 bytes/doc — e.g. ~1GB for
    8M documents, far less than a live LSH index at the same scale). Bucket documents by
    each band's hash via numpy lexsort (vectorized, not a Python-level hash() per row)
    and union same-bucket documents via a plain array-backed union-find. A document is
    kept iff it's the smallest-index representative of its connected component — i.e.
    the first-seen copy of each near-duplicate cluster survives.
    Pass 2: re-read the input and write only the kept documents."""
    import multiprocessing

    import numpy as np

    workers = workers or max(1, multiprocessing.cpu_count() - 2)
    t0 = time.time()

    # --- Pass 1a: signatures ---
    print(f"[{lang}] stage3 pass 1a: computing MinHash signatures ({workers} workers)...",
          file=sys.stderr)
    sig_chunks = []
    n_sig = 0
    with multiprocessing.Pool(workers) as pool:
        batch = []

        def flush(texts):
            nonlocal n_sig
            sigs = pool.map(_compute_minhash_signature, texts)
            sig_chunks.append(np.array(sigs, dtype=np.uint64))
            n_sig += len(texts)
            if n_sig % LOG_EVERY < batch_size:
                print(f"[{lang}] stage3: {n_sig:,} signatures computed, "
                      f"{time.time() - t0:.0f}s, rss={_rss_mb():.0f}MB", file=sys.stderr)

        for text in iter_jsonl(in_path):
            batch.append(text)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    sig_array = np.concatenate(sig_chunks) if sig_chunks else np.zeros((0, MINHASH_NUM_PERM),
                                                                        dtype=np.uint64)
    del sig_chunks
    n = len(sig_array)
    print(f"[{lang}] stage3: {n:,} signatures ready, {time.time() - t0:.0f}s, "
          f"rss={_rss_mb():.0f}MB", file=sys.stderr)

    # --- Pass 1b: LSH banding + union-find ---
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    rows_per_band = MINHASH_NUM_PERM // num_bands
    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        band = sig_array[:, start:start + rows_per_band]
        # lexsort: primary key last in the list; this sorts rows so identical bands
        # (i.e. candidate near-duplicates for this band) become adjacent — vectorized,
        # no per-row Python hash() calls.
        order = np.lexsort([band[:, c] for c in reversed(range(rows_per_band))])
        sorted_band = band[order]
        same_as_prev = np.all(sorted_band[1:] == sorted_band[:-1], axis=1)
        # union each run of equal consecutive rows (only iterates over actual matches,
        # which is a small fraction of n given real near-dup rates). same_as_prev[k] means
        # sorted_band[k] == sorted_band[k+1]; a maximal run of True at [i, j-1] means rows
        # [i, j] (inclusive) are all equal, so union order[i] with order[i+1..j].
        i = 0
        m = len(same_as_prev)
        while i < m:
            if same_as_prev[i]:
                j = i
                while j < m and same_as_prev[j]:
                    j += 1
                # equal-row run spans sorted_band indices [i, j] inclusive
                first_doc = int(order[i])
                for k in range(i + 1, j + 1):
                    union(first_doc, int(order[k]))
                i = j + 1
            else:
                i += 1
        print(f"[{lang}] stage3: band {band_idx + 1}/{num_bands} bucketed, "
              f"{time.time() - t0:.0f}s, rss={_rss_mb():.0f}MB", file=sys.stderr)

    roots = np.array([find(i) for i in range(n)])
    keep_mask = roots == np.arange(n)
    n_kept = int(keep_mask.sum())
    print(f"[{lang}] stage3: {n:,} -> {n_kept:,} after near-dup merge "
          f"({time.time() - t0:.0f}s)", file=sys.stderr)

    # --- Pass 2: write survivors ---
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, text in enumerate(iter_jsonl(in_path)):
            if keep_mask[i]:
                out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    return {"in": n, "kept": n_kept}


def stage4_split(in_path: Path, out_dir: Path, lang: str,
                  target_chars: int | None = None) -> dict:
    """Two-pass, memory-light shuffle+split: holding every surviving document's full
    text in one Python list to shuffle it is the same class of bug that crashed things
    earlier (millions of docs x hundreds-to-thousands of chars each). Instead: pass 1
    reads document lengths and builds a per-line split-label array (int8, ~1 byte/doc —
    a few MB even at millions of documents, not the text itself). Pass 2 streams the
    input once, routing each line straight to its assigned output file (or skipping it,
    if excluded by target_chars).

    If target_chars is given, documents are visited in random order and included until
    the running total reaches target_chars — a genuine random sample of the corpus, not
    just "the first N by file order" — and only that sample is split into train/val/test;
    the rest are excluded entirely. Use this to cap final corpus size (e.g. survival
    rates turning out better than estimated can otherwise land meaningfully over target)."""
    import numpy as np

    with in_path.open(encoding="utf-8") as f:
        lengths = np.array([len(json.loads(line)["text"]) for line in f], dtype=np.int64)
    n = len(lengths)

    order = np.arange(n)
    np.random.RandomState(SEED).shuffle(order)

    if target_chars is not None:
        cumulative = np.cumsum(lengths[order])
        # smallest n_included such that the first n_included docs' chars sum >= target;
        # searchsorted(side='right') gives the count of cumulative entries <= target, so
        # +1 includes the one document that actually crosses the threshold (unless the
        # cap is hit exactly, in which case that +1 would overshoot by one document —
        # negligible at this scale, and erring toward "at least the target" over "just
        # under" is the safer direction for a token-budget cap).
        n_included = int(np.searchsorted(cumulative, target_chars, side="right")) + 1
        n_included = min(n_included, n)
        order = order[:n_included]
        print(f"[{lang}] stage4: capping to ~{target_chars:,} chars -> "
              f"{n_included:,}/{n:,} documents included "
              f"({int(cumulative[n_included - 1]):,} chars), "
              f"{n - n_included:,} excluded", file=sys.stderr)

    n_included = len(order)
    n_train = int(n_included * SPLIT_RATIOS[0])
    n_val = int(n_included * SPLIT_RATIOS[1])

    TRAIN, VAL, TEST = 1, 2, 3
    split_labels = np.zeros(n, dtype=np.int8)  # 0 = excluded (only relevant if target_chars set)
    split_labels[order[:n_train]] = TRAIN
    split_labels[order[n_train:n_train + n_val]] = VAL
    split_labels[order[n_train + n_val:]] = TEST

    label_name = {TRAIN: "train", VAL: "val", TEST: "test"}
    out_paths = {name: out_dir / f"{name}.txt" for name in ("train", "val", "test")}
    out_files = {name: p.open("w", encoding="utf-8") for name, p in out_paths.items()}
    counts = {name: 0 for name in out_files}
    chars = {name: 0 for name in out_files}
    try:
        with in_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                label = int(split_labels[i])
                if label == 0:
                    continue
                split_name = label_name[label]
                text = json.loads(line)["text"]
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

    target_chars = int(TARGET_TOKENS * CHARS_PER_TOKEN[lang])
    print(f"[{lang}] stage 4: shuffle + split {SPLIT_RATIOS}, capped to "
          f"~{TARGET_TOKENS:,} tokens (~{target_chars:,} chars)...", file=sys.stderr)
    stats["splits"] = stage4_split(s3_path, out_dir, lang, target_chars=target_chars)

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[{lang}] DONE. Stats written to {out_dir / 'stats.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
