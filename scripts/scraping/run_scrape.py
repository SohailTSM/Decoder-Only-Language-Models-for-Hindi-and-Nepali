"""Scrape one news domain's post-cutoff articles into a JSONL file.

Two discovery modes:
  --sitemap URL
      Recursively walk a sitemap/sitemap-index (see sitemap_utils.discover_urls).
  --date-template "https://site.com/news-{dd}-{mm}-{yyyy}.xml" --start-date YYYY-MM-DD --end-date YYYY-MM-DD
      For sites that expose one dated sitemap file per day with no master index (e.g.
      abplive.com). Processed in day-batches, most recent first, so the shared budget can
      cut discovery short instead of always enumerating the full date range up front.

Usage:
    python -m scripts.scraping.run_scrape \\
        --name bhaskar --sitemap https://www.bhaskar.com/sitemaps-v1--sitemap-google-news-index.xml \\
        --out hindi/data/manual/bhaskar.jsonl --cutoff 2023-01-01 --max-articles 50000 --workers 8

Resumable: URLs already present in --out are skipped on re-run. Writes progress to stderr
every 200 articles (count, approx chars, elapsed).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

from scripts.scraping import progress
from scripts.scraping.date_archive import daily_sitemap_urls
from scripts.scraping.fetch_extract import USER_AGENT, content_hash, fetch_article_text
from scripts.scraping.sitemap_utils import discover_urls

DAY_BATCH = 20  # for --date-template mode: how many daily sitemaps to discover per round


def load_seen(out_path: Path) -> tuple[set, set]:
    seen_urls, seen_hashes = set(), set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    seen_urls.add(row["url"])
                    seen_hashes.add(row["hash"])
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a partially-written last line from a killed run
    return seen_urls, seen_hashes


def _budget_met(progress_file, target_chars) -> bool:
    return bool(progress_file and target_chars and
                progress.read_chars(progress_file) >= target_chars)


def _fetch_batch(candidates, out_f, fetch_client, workers, seen_hashes, args, state, cutoff) -> bool:
    """Fetch+write one batch of candidate URLs. Mutates state (dict with written/approx_chars/i).
    Returns True if the shared budget was hit during this batch (caller should stop)."""

    def task(url: str):
        return url, fetch_article_text(fetch_client, url, source_name=args.name, cutoff=cutoff)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, u): u for u in candidates}
        for fut in as_completed(futures):
            state["i"] += 1
            try:
                url, text = fut.result()
            except Exception as e:
                print(f"  [warn] task failed for {futures[fut]}: {e!r}", file=sys.stderr)
                continue
            if text:
                h = content_hash(text)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    state["approx_chars"] += len(text)
                    out_f.write(json.dumps(
                        {"url": url, "source": args.name, "hash": h,
                         "approx_tokens": len(text.split()), "text": text},
                        ensure_ascii=False) + "\n")
                    out_f.flush()
                    state["written"] += 1
                    if args.progress_file and args.target_chars:
                        shared_total = progress.add_chars(args.progress_file, len(text))
                        if shared_total >= args.target_chars:
                            print(f"[{args.name}] shared budget reached "
                                  f"({shared_total:,}/{args.target_chars:,} chars) — "
                                  "stopping (other domains for this language will too)",
                                  file=sys.stderr)
                            for pending in futures:
                                pending.cancel()
                            return True
            if state["i"] % 200 == 0:
                elapsed = time.time() - state["start"]
                print(f"[{args.name}] {state['i']} fetched, {state['written']} kept, "
                      f"~{state['approx_chars']:,} chars, {elapsed:.0f}s elapsed", file=sys.stderr)
    return False


def run(args) -> None:
    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen_urls, seen_hashes = load_seen(args.out)

    if _budget_met(args.progress_file, args.target_chars):
        total_now = progress.read_chars(args.progress_file)
        print(f"[{args.name}] shared budget already met "
              f"({total_now:,}/{args.target_chars:,} chars) — nothing to do", file=sys.stderr)
        return

    headers = {"User-Agent": USER_AGENT}
    discover_client = httpx.Client(headers=headers)
    fetch_client = httpx.Client(headers=headers, limits=httpx.Limits(max_connections=args.workers))
    state = {"start": time.time(), "written": 0, "approx_chars": 0, "i": 0}
    budget_hit = False

    with args.out.open("a", encoding="utf-8") as out_f:
        if args.date_template:
            day_urls = list(daily_sitemap_urls(args.date_template, args.start_date, args.end_date))
            print(f"[{args.name}] date-archive mode: {len(day_urls)} daily sitemaps to scan "
                  f"({args.start_date} .. {args.end_date}), batches of {DAY_BATCH}", file=sys.stderr)
            for b in range(0, len(day_urls), DAY_BATCH):
                if len(seen_urls) + state["written"] >= args.max_articles:
                    break
                if _budget_met(args.progress_file, args.target_chars):
                    print(f"[{args.name}] shared budget already met before next batch — stopping",
                          file=sys.stderr)
                    break
                batch_candidates = []
                for day_url in day_urls[b:b + DAY_BATCH]:
                    try:
                        for url, _date in discover_urls(discover_client, day_url, cutoff,
                                                          max_page=args.max_sitemap_page):
                            if url not in seen_urls and url not in batch_candidates:
                                batch_candidates.append(url)
                    except Exception as e:
                        print(f"[{args.name}] [warn] failed to discover {day_url}: {e!r}",
                              file=sys.stderr)
                room = args.max_articles - (len(seen_urls) + state["written"])
                if room < len(batch_candidates):
                    batch_candidates = batch_candidates[:max(room, 0)]
                print(f"[{args.name}] batch {b // DAY_BATCH + 1}: {len(batch_candidates)} "
                      f"new candidate URLs", file=sys.stderr)
                if not batch_candidates:
                    continue
                seen_urls.update(batch_candidates)
                budget_hit = _fetch_batch(batch_candidates, out_f, fetch_client, args.workers,
                                           seen_hashes, args, state, cutoff)
                if budget_hit:
                    break
        else:
            print(f"[{args.name}] discovering URLs from sitemap..."
                  + (f" (capped at page {args.max_sitemap_page})" if args.max_sitemap_page else ""),
                  file=sys.stderr)
            candidates = []
            discovery_start = time.time()
            last_heartbeat = discovery_start
            try:
                for url, _date in discover_urls(discover_client, args.sitemap, cutoff,
                                                  max_page=args.max_sitemap_page):
                    if url not in seen_urls:
                        candidates.append(url)
                    now = time.time()
                    if now - last_heartbeat >= 15:
                        # Some sites (e.g. deeply paginated, uncached sitemap pages behind a
                        # CDN) take many minutes to fully enumerate — without this, a long
                        # discovery phase looks identical to a hang.
                        print(f"[{args.name}] discovery in progress: {len(candidates)} "
                              f"candidates so far, {now - discovery_start:.0f}s elapsed",
                              file=sys.stderr)
                        last_heartbeat = now
                    if len(candidates) + len(seen_urls) >= args.max_articles:
                        break
            except Exception as e:
                # Keep whatever candidates were discovered before the failure — a partial
                # candidate list still makes progress instead of losing the whole run.
                print(f"[{args.name}] [warn] discovery stopped early: {e!r}", file=sys.stderr)
            print(f"[{args.name}] {len(candidates)} new candidate URLs "
                  f"(already have {len(seen_urls)}), discovery took "
                  f"{time.time() - discovery_start:.0f}s", file=sys.stderr)
            if candidates:
                budget_hit = _fetch_batch(candidates, out_f, fetch_client, args.workers,
                                           seen_hashes, args, state, cutoff)

    discover_client.close()
    fetch_client.close()
    elapsed = time.time() - state["start"]
    status = "budget reached" if budget_hit else "candidates exhausted"
    print(f"[{args.name}] DONE ({status}): {state['written']} articles kept, "
          f"~{state['approx_chars']:,} chars, {elapsed:.0f}s", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sitemap", default=None)
    ap.add_argument("--date-template", default=None,
                     help="e.g. 'https://site.com/news-{dd}-{mm}-{yyyy}.xml'; alternative to "
                          "--sitemap for sites with one dated sitemap file per day.")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD, required with --date-template")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD, required with --date-template")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cutoff", default="2023-01-01")
    ap.add_argument("--max-articles", type=int, default=500_000,
                     help="Hard per-domain safety ceiling; the real stop condition is "
                          "--target-chars shared across all domains via --progress-file.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-sitemap-page", type=int, default=None,
                     help="For paginated sitemap indexes using a `?p=N` param, don't follow "
                          "child sitemaps beyond this page number.")
    ap.add_argument("--target-chars", type=int, default=None,
                     help="Shared character budget for this language (across all its "
                          "domains). Once the total in --progress-file reaches this, this "
                          "and sibling domain processes stop cleanly.")
    ap.add_argument("--progress-file", type=Path, default=None,
                     help="Shared JSON counter file (one per language) used to coordinate "
                          "--target-chars across the parallel per-domain processes.")
    args = ap.parse_args()

    if not args.sitemap and not args.date_template:
        ap.error("one of --sitemap or --date-template is required")
    if args.date_template and not (args.start_date and args.end_date):
        ap.error("--date-template requires --start-date and --end-date")

    try:
        run(args)
    except Exception as e:
        # Non-zero exit lets an outer shell retry-loop (see launch_all.sh) know this
        # domain's process died and should be restarted; already-written rows persist
        # in --out and load_seen() will skip them on the next attempt.
        print(f"[{args.name}] FATAL: {e!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
