#!/usr/bin/env bash
# Launches one scraper process per source listed in <lang>/data/sources.json, writing
# JSONL to <lang>/data/manual/<name>.jsonl and logs to <lang>/data/manual/<name>.log.
#
# Stop condition: each language has a shared TARGET_TOKENS budget (default 130M), converted
# to a character budget via a language-specific chars-per-token estimate (see README below),
# tracked in <lang>/data/manual/_progress.json across all of that language's domain processes.
# Once the shared total is reached, every domain process for that language exits cleanly (no
# retry) — they don't each independently try to hit 130M.
#
# chars-per-token estimates (grounded, not guessed — see report/phase1_data_report.md):
#   Hindi:  ~4.0 chars/token  (32k-vocab monolingual BPE/SentencePiece fertility ~1.4 tok/word,
#           avg Hindi word length ~4.8 chars + 1 space)
#   Nepali: ~4.5 chars/token  (avg Nepali word length 5.75 chars [NepaliGPT, arXiv:2506.16399]
#           + 1 space, assumed similar ~1.5 tok/word fertility given richer verb morphology)
#
# Safe to re-run: run_scrape.py resumes (skips URLs/hashes already in --out), and this script
# recomputes each language's progress baseline from the actual JSONL files on disk every time
# it starts, so restarts (including across days) stay correct even if _progress.json is stale
# or missing.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

MAX_ARTICLES="${MAX_ARTICLES:-500000}"   # per-domain safety ceiling only; see header above
# WORKERS=5: this machine (i5-8250U, 4C/8T, ~2.7GB free RAM) has plenty of local CPU/RAM
# headroom for this I/O-bound workload (a single process measured ~53MB RSS) — the actual
# limiter is each remote site's own rate tolerance. Applies to Hindi's 7 domains, which are
# all healthy at this level (running well in parallel already).
WORKERS="${WORKERS:-5}"
# NEPALI_WORKERS=50: annapurnapost.com specifically was benchmarked directly (genuinely
# uncached URLs, not CDN cache hits) — throughput peaked around workers=50 (4.37 req/s, up
# from ~1.3 req/s at workers=5-15) and *degraded* at workers=80 (3.16 req/s), meaning the
# origin server saturates and slows down past that point rather than erroring out. 50 is the
# measured sweet spot, not a guess.
NEPALI_WORKERS="${NEPALI_WORKERS:-50}"
CUTOFF="${CUTOFF:-2023-01-01}"
MAX_RETRIES="${MAX_RETRIES:-5}"

TARGET_TOKENS="${TARGET_TOKENS:-130000000}"
HINDI_CHARS_PER_TOKEN="${HINDI_CHARS_PER_TOKEN:-4.0}"
# NEPALI_CHARS_PER_TOKEN=5.0 is intentionally inflated above the ~4.5 chars/token estimate
# (see report/phase1_data_report.md) — it sets the operational stop-ceiling at ~650M
# chars, not a literal chars/token claim. Reason: these 4 Nepali sources are independent
# news outlets that often cover the same events, so some cross-source near-duplicate
# content is expected; the live per-domain hash dedup only catches exact matches within
# a single source, not across sources. A final cross-source dedup pass still needs to run
# before reporting the true unique token count — this buffer just ensures we don't stop
# collecting at exactly 130M-tokens-worth of raw (pre-dedup) text and end up short after
# that pass removes overlap. Report the post-dedup count, not this raw total, as the
# manual-token figure.
NEPALI_CHARS_PER_TOKEN="${NEPALI_CHARS_PER_TOKEN:-5.0}"

target_chars_for() {
  local lang="$1" cpt
  case "$lang" in
    hindi) cpt="$HINDI_CHARS_PER_TOKEN" ;;
    nepali) cpt="$NEPALI_CHARS_PER_TOKEN" ;;
  esac
  python3 -c "print(int($TARGET_TOKENS * $cpt))"
}

recompute_progress_baseline() {
  local lang="$1"
  python3 -c "
import glob, json
total = 0
for fp in glob.glob('$lang/data/manual/*.jsonl'):
    with open(fp, encoding='utf-8') as f:
        for line in f:
            try:
                total += len(json.loads(line).get('text', ''))
            except json.JSONDecodeError:
                continue
with open('$lang/data/manual/_progress.json', 'w', encoding='utf-8') as f:
    json.dump({'total_chars': total}, f)
print(f'[$lang] progress baseline recomputed from disk: {total:,} chars already collected')
"
}

# Runs one domain's scrape; on a crash (non-zero exit) retries with backoff, since
# run_scrape.py resumes from --out on each attempt. A clean exit (0) — whether from
# exhausting candidates or hitting the shared language budget — is NOT retried.
# $3 is either "sitemap <url>" or "date_archive <template> <start_date>".
run_with_retry() {
  local lang="$1" name="$2" out="$3" log="$4" target_chars="$5" workers="$6"; shift 6
  local discovery_args=("$@")
  local attempt=0
  while true; do
    python -m scripts.scraping.run_scrape \
      --name "$name" --out "$out" "${discovery_args[@]}" \
      --cutoff "$CUTOFF" --max-articles "$MAX_ARTICLES" --workers "$workers" \
      --target-chars "$target_chars" --progress-file "$lang/data/manual/_progress.json" \
      >>"$log" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
      echo "$(date -Iseconds) [$lang/$name] finished cleanly" >>"$log"
      return 0
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_RETRIES" ]; then
      echo "$(date -Iseconds) [$lang/$name] gave up after $attempt crashes" >>"$log"
      return 1
    fi
    backoff=$((10 * attempt))
    echo "$(date -Iseconds) [$lang/$name] crashed (exit $status), retry $attempt/$MAX_RETRIES in ${backoff}s" >>"$log"
    sleep "$backoff"
  done
}

TODAY="$(date +%Y-%m-%d)"

for lang in hindi nepali; do
  recompute_progress_baseline "$lang"
  target_chars="$(target_chars_for "$lang")"
  if [ "$lang" = "nepali" ]; then workers="$NEPALI_WORKERS"; else workers="$WORKERS"; fi
  echo "[$lang] target: $TARGET_TOKENS tokens -> $target_chars chars, workers=$workers"
  while IFS='|' read -r type name a b; do
    out="$lang/data/manual/${name}.jsonl"
    log="$lang/data/manual/${name}.log"
    echo "launching $lang/$name -> $out (log: $log)"
    if [ "$type" = "date_archive" ]; then
      run_with_retry "$lang" "$name" "$out" "$log" "$target_chars" "$workers" \
        --date-template "$a" --start-date "$b" --end-date "$TODAY" &
    elif [ -n "$b" ]; then
      run_with_retry "$lang" "$name" "$out" "$log" "$target_chars" "$workers" \
        --sitemap "$a" --max-sitemap-page "$b" &
    else
      run_with_retry "$lang" "$name" "$out" "$log" "$target_chars" "$workers" --sitemap "$a" &
    fi
  done < <(python3 -c "
import json
for s in json.load(open('$lang/data/sources.json')):
    t = s.get('type', 'sitemap')
    if t == 'date_archive':
        print(f\"date_archive|{s['name']}|{s['date_template']}|{s['start_date']}\")
    else:
        print(f\"sitemap|{s['name']}|{s['sitemap']}|{s.get('max_sitemap_page', '')}\")
")
done

wait
echo "all scrapers finished"
