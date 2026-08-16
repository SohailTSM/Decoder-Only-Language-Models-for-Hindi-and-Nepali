"""Download the public (non-manual) portion of the corpus: the first ~400M tokens'
worth of text from IndicCorpV2's Hindi and Nepali shards, sequentially (Hindi then
Nepali, one at a time — not concurrent).

Why a partial stream instead of the full file: both hi-1.txt (26.7GB) and ne.txt
(14.8GB) are confirmed shuffled (see report/phase1_data_report.md — all three Hindi
shards are byte-identical in size, which only happens if a single pre-shuffled stream
was mechanically cut into equal pieces; content sampled from multiple points in each
file showed unrelated topics with no clustering). So the first N characters of the
(already-shuffled) file is a valid random sample — no need to download the whole thing.

Stops on an exact character count (not a bytes estimate), decoding incrementally so a
multi-byte UTF-8 character split across two download chunks is never corrupted, and only
writes complete lines (holds back a trailing partial line) so the output is never
truncated mid-document.
"""

import codecs
import os
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", 400_000_000))
CHUNK_SIZE = 1024 * 1024  # 1 MB

# chars_per_token estimates: see report/phase1_data_report.md
SOURCES = {
    "hindi": {
        "url": "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/hi-1.txt",
        "out": REPO_ROOT / "hindi" / "data" / "raw_public" / "indiccorpv2_hi1.txt",
        "chars_per_token": 4.0,
    },
    "nepali": {
        "url": "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/ne.txt",
        "out": REPO_ROOT / "nepali" / "data" / "raw_public" / "indiccorpv2_ne.txt",
        "chars_per_token": 4.5,
    },
}


def existing_char_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(len(line) for line in f)


def download_language(name: str, spec: dict) -> None:
    target_chars = int(TARGET_TOKENS * spec["chars_per_token"])
    out_path = spec["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    have = existing_char_count(out_path)
    if have >= target_chars:
        print(f"[{name}] already have {have:,} chars (target {target_chars:,}) — skipping",
              file=sys.stderr)
        return
    if have > 0:
        print(f"[{name}] found a partial download ({have:,} chars) — restarting cleanly "
              f"(resuming a UTF-8 stream mid-file isn't worth the complexity for a "
              f"multi-GB download)", file=sys.stderr)

    print(f"[{name}] downloading toward {target_chars:,} chars "
          f"(~{TARGET_TOKENS:,} tokens at {spec['chars_per_token']} chars/token)...",
          file=sys.stderr)

    decoder = codecs.getincrementaldecoder("utf-8")()
    start = time.time()
    last_log = start
    bytes_downloaded = 0
    chars_written = 0
    buffer = ""

    with httpx.Client(follow_redirects=True, timeout=60) as client, \
         out_path.open("w", encoding="utf-8") as out_f, \
         client.stream("GET", spec["url"]) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(CHUNK_SIZE):
            bytes_downloaded += len(chunk)
            buffer += decoder.decode(chunk)
            if "\n" in buffer:
                *complete_lines, buffer = buffer.split("\n")
                text = "\n".join(complete_lines) + "\n"
                out_f.write(text)
                chars_written += len(text)

            now = time.time()
            if now - last_log >= 5:
                mb = bytes_downloaded / 1e6
                print(f"[{name}] {chars_written:,}/{target_chars:,} chars "
                      f"({mb:.0f} MB downloaded), {now - start:.0f}s elapsed",
                      file=sys.stderr)
                last_log = now

            if chars_written >= target_chars:
                break

    elapsed = time.time() - start
    est_tokens = chars_written / spec["chars_per_token"]
    print(f"[{name}] DONE: {chars_written:,} chars (~{est_tokens:,.0f} tokens), "
          f"{bytes_downloaded / 1e6:.0f} MB downloaded, {elapsed:.0f}s", file=sys.stderr)


MAX_RETRIES = 5


def download_language_with_retry(name: str, spec: dict) -> None:
    """download_language() restarts the whole stream cleanly on retry (see the
    "restarting cleanly" log line) rather than resuming mid-file — simple and safe for
    a multi-GB download, at the cost of re-downloading progress made before a transient
    failure. Retries with backoff since this has hit real transient errors twice now
    (a DNS resolution failure, and a ReadTimeout) over the course of this project."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            download_language(name, spec)
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[{name}] FATAL after {MAX_RETRIES} retries: {e!r}", file=sys.stderr)
                sys.exit(1)
            backoff = 10 * (attempt + 1)
            print(f"[{name}] error: {e!r} — retry {attempt + 1}/{MAX_RETRIES} in {backoff}s",
                  file=sys.stderr)
            time.sleep(backoff)


def main() -> None:
    for name, spec in SOURCES.items():  # sequential, Hindi then Nepali
        download_language_with_retry(name, spec)


if __name__ == "__main__":
    main()
