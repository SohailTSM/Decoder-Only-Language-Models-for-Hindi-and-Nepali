# Phase 1 — Dataset Collection & Preprocessing Report

Model H = Hindi (higher-resource). Model L = Nepali (lower-resource, from the assignment's
allowed list). This report covers dataset collection, non-overlap methodology, and
preprocessing for both, plus every non-obvious decision made along the way and why.

## 1. Language selection

- **Model H — Hindi.** Highest-resource non-English Indian language: largest Wikipedia and
  news-crawl presence of any Indic language, the biggest split in IndicCorpV2 (152M rows for
  `hin_Deva`, the largest of the 23 languages), and by far the deepest pool of digitized news
  archives for manual collection.
- **Model L — Nepali.** Chosen from the required list (Assamese, Bhojpuri, Bodo, Dogri, Konkani,
  Maithili, Manipuri, Mizo, Nepali, Sindhi). Devanagari script (shared tooling with Hindi:
  same script-purity checks, largely overlapping profanity vocabulary), an active online news
  ecosystem (unlike some options on the list), and a real public corpus in IndicCorpV2
  (`npi_Deva`) to build on.

## 2. Non-overlap strategy (manual vs. public)

`IndicCorpV2` (both `hin_Deva` and `npi_Deva`) is built from Wikipedia, large news-site crawls,
and OSCAR/CommonCrawl, with the underlying data **frozen around Oct–Nov 2022** (per the
accompanying paper, [arXiv:2212.05409](https://arxiv.org/abs/2212.05409)) — it cannot contain
any document published after that date.

**Decision:** manual collection scrapes only articles **published 2023-01-01 or later**. This is
a structural, provable non-overlap guarantee that needs no OCR, transcription, or hunting for
obscure sources — just a date filter on each site's own sitemap metadata (with a caveat on trust
in that metadata — see §4.3).

## 3. Public data (IndicCorpV2)

### 3.1 Is it shuffled? (matters because we only download a truncated prefix, not the whole file)

Verified two ways before trusting a truncated download as a valid sample:

1. **Documented:** AI4Bharat's pipeline explicitly includes a shuffling step.
2. **Empirically re-verified:** Hindi's three shard files (`hi-1.txt`, `hi-2.txt`, `hi-3.txt`)
   are **byte-identical in size** (26,661,096,882 bytes each) — only possible if a single
   pre-assembled stream was mechanically cut into equal pieces (real per-source files never
   divide exactly by chance). Content sampled from the start/middle/end of `hi-1.txt`, and the
   start of `hi-2.txt`/`hi-3.txt`, showed unrelated topics at every point (party politics,
   COVID-era administration, Hezbollah commentary, business earnings, infrastructure news) with
   no clustering — no evidence of large contiguous blocks from one source or topic. Same check
   on `ne.txt`. Each sampled excerpt was itself a coherent multi-sentence paragraph, so shuffling
   happened at the document/paragraph level, not by scrambling individual sentences.

**Conclusion:** the first N characters of the (pre-shuffled) file is a valid random sample — no
need for the full 26.7GB/14.8GB files.

### 3.2 What preprocessing IndicCorpV2 already has (checked directly against the paper, not assumed)

| Applied | NOT applied |
|---|---|
| Language ID filtering (paragraph-level, cld3+langdetect) | Repetitive document removal |
| Script-purity filter (≥75% native-script chars, **sentence-level**) | Perplexity filtering |
| Length filter (<10 words after punctuation stripped → dropped) | Document quality filtering beyond length |
| Offensive-word blocklist filtering (17 languages) | Paragraph/MinHash near-dup dedup, semantic filtering, topic filtering |
| *Some* deduplication (paper's Table 3 calls it a "de-duplicated corpus" but gives **no methodological detail** on how) | — |

This matters for two things: (a) it's why our own Tier-1 pipeline (§5) is still worth applying to
the public portion, not just manual; (b) the sentence-level script-purity rule (≥75% native
script) is the exact rule our own cleaning pipeline reuses.

### 3.3 Download method and target

Target: **400M tokens per language** from the public portion. Converted to a character count
using the chars-per-token estimates derived in §4.2 (same estimates used for both the scrape
stop-condition and this download), then streamed only that many characters from `hi-1.txt` /
`ne.txt` — incremental UTF-8 decoding (so a multi-byte character split across download chunks is
never corrupted) and only complete lines written (a trailing partial line is held back, never
truncated into the output).

**Result:**

| Language | Chars downloaded | Tokens (est.) |
|---|---|---|
| Hindi | 1,600,354,148 | ~400,088,537 |
| Nepali | 1,800,070,395 | ~400,015,643 |

## 4. Manual data collection

### 4.1 Method

Sitemap-based scraping (`scripts/scraping/`): discover each domain's article URLs via its own
sitemap/sitemap-index, filter to the 2023-01-01+ cutoff, fetch each article with `httpx`, extract
main body text with `trafilatura` (handles Devanagari well, strips nav/boilerplate reliably),
dedupe within each domain via content hash, write to JSONL. Fully resumable (re-running skips
URLs/hashes already saved) and error-tolerant (retries transient failures, then skips and
continues rather than crashing on a bad URL).

### 4.2 Character budget derivation

No tokenizer existed yet at collection time, so the scraper's stop condition is a character
count, converted from a token target via a chars-per-token estimate:

- **Hindi ≈ 4.0 chars/token:** fertility for a well-trained ~32k-vocab monolingual
  BPE/SentencePiece tokenizer on Hindi is reported around 1.04–1.45 tokens/word (BHARATI v3:
  1.04, [arXiv:2607.23319](https://arxiv.org/abs/2607.23319); mBART SentencePiece: 1.45). Used
  ~1.4 tokens/word. Average Hindi word length ~4.8 chars (EMMILLE corpus range 4–18; comparable
  Devanagari languages cluster 5–6 chars) + 1 for the following space ≈ 5.8 chars/word-slot.
  5.8 / 1.4 ≈ **4.0**.
- **Nepali ≈ 4.5 chars/token:** average word length **5.75 characters**, directly reported from a
  383M-token Nepali corpus (NepaliGPT, [arXiv:2506.16399](https://arxiv.org/abs/2506.16399)) →
  ≈6.75 chars/word-slot. No direct Nepali fertility figure found; assumed slightly higher than
  Hindi's (~1.5 tokens/word) given richer verbal morphology. 6.75 / 1.5 ≈ **4.5**.

These are planning estimates only — the real tokenizer (§6) re-measures actual token counts.

### 4.3 Sources, and a critical bug found along the way

**Hindi (6 rolling-window domains + 1 deep archive):** bhaskar.com, amarujala.com, jagran.com,
navbharattimes.indiatimes.com, abplive.com, prabhatkhabar.com. `hindi.news18.com` returned 403
and was dropped.

These six domains' sitemaps only expose a **current rolling window** (recent days/weeks), not
deep history — their combined pool capped out at ~34,714 URLs, far short of the ≥100M-token
manual minimum. **Fix:** discovered `abplive.com` also has a genuine **daily-dated archive**
(`news-{dd}-{mm}-{yyyy}.xml`) going back to **2023-10-01**, unrelated to its rolling sitemap.
Built a date-range crawler for it (`abplive_archive`) — this single source closed the entire
gap.

**Nepali (started at 2, ended at 4 working sources):** `annapurnapost.com`,
`nagariknews.nagariknetwork.com` initially; `ekantipur.com`, `onlinekhabar.com`, `setopati.com`,
`ratopati.com` were checked and dropped (SPA shells with no fetchable sitemap, unreachable, or a
sitemap that returns an HTML app-shell instead of XML). `nagariknews` never actually contributed
any articles despite being reachable — kept in the source list for transparency/reproducibility,
but contributed 0 of the final corpus.

**The bug:** `annapurnapost`'s sitemap `<lastmod>` field turned out to be **fake** — every entry
across two different pages (700 and 1264, covering genuinely different article-ID ranges) showed
the *identical* timestamp `2023-02-05`, which is a red flag for a static/batch value rather than
a real per-article date. Verified by checking real article content against real-world events: an
article the sitemap claimed was "2023-02-05" was actually about Brazil sacking Dunga and
appointing Tite as coach — **June 2016**. This meant the site's own cutoff filtering was not
actually restricting to 2023+ content; the scraper had been walking essentially the site's entire
history (spanning back to ~2016), which breaks the non-overlap guarantee in §2.

Investigated a proper fix (parsing the real per-article Nepali Bikram Sambat date embedded in
each article's HTML, using the `nepali_datetime` library — validated it correctly converts BS
dates and correctly identified both the 2016 article and a confirmed-2020 COVID-era article as
pre-cutoff). **This was reverted per direct instruction** — the operator had independently
verified the sitemap's actual paginated depth and specified a hard cap: **only follow sitemap
pages `p ≤ 400`** (implemented as a general `max_sitemap_page` filter in `sitemap_utils.py`,
applied only to `annapurnapost`). Simpler, and sufficient given the operator's own verification.

The same class of bug turned up independently in `nepalpress.com` (a different, WordPress-based
site) — its `lastmod` field showed 2026 for an article whose URL (`/2020/12/06/...`) and content
confirmed it was from 2020. **General fix applied to all three later Nepali sources**
(`khabarhub`, `nepalpress`, `deshsanchar`, all WordPress-based): a **URL-embedded-year check**.
WordPress permalinks bake the real publish year into the URL path at publish time
(`domain.com/YYYY/...`) and it is never edited afterward, unlike a `lastmod` timestamp — more
trustworthy than any date metadata field. Implemented as a generic filter in
`sitemap_utils.py` (`domain.com/YYYY/...` with `YYYY < cutoff.year` → reject), verified on a
sample of 20 `khabarhub` articles: all genuinely 2024–2025, zero pre-2023 leakage.

**Throughput tuning:** `annapurnapost`'s origin server was measured (via direct, isolated
benchmarking against genuinely-uncached URLs, not the misleading CDN-cache-hit numbers from
casual testing) to peak at **`workers=50`** (4.37 req/s, vs. ~1.3 req/s at `workers=15`) and
**degrade** at `workers=80` (3.16 req/s — the origin saturates rather than erroring), so 50 is a
measured value, not a guess.

**Buffer for cross-source overlap:** the four working Nepali sources' combined pool ceiling was
only ~4.8% above the 130M-token target once computed precisely — since these are independent
news outlets that often cover the same events, some cross-source near-duplicate content was
expected going into the preprocessing dedup pass (§5), and exact/paragraph dedup only ever
applied *within* one source's own output, never across sources. The scrape's shared stop-budget
was raised (130M-token equivalent moved from 585M to **650M characters**, i.e. an inflated
`chars/token` used only as an operational ceiling, not a literal claim) so no source would stop
collecting early — each source ran to its own natural pool exhaustion instead.

### 4.4 Results

| Language | Source | Articles | Chars |
|---|---|---:|---:|
| Hindi | bhaskar | 3,520 | — |
| Hindi | amarujala | 820 | — |
| Hindi | jagran | 2,614 | — |
| Hindi | navbharattimes | 1,041 | — |
| Hindi | prabhatkhabar | 768 | — |
| Hindi | abplive | 12,386 | — |
| Hindi | abplive_archive | 178,839 | — |
| **Hindi total** | | **199,988** | **520,000,332 (~130,000,083 tokens)** |
| Nepali | khabarhub | 136,693 | 228,577,365 |
| Nepali | nepalpress | 108,133 | 198,119,929 |
| Nepali | annapurnapost | 39,248 | 81,980,046 |
| Nepali | deshsanchar | 29,798 | 43,300,650 |
| Nepali | nagariknews | 0 | 0 |
| **Nepali total** | | **313,872** | **551,977,990 (~122,661,776 tokens)** |

All 5 Nepali sources exhausted their available post-2023 pools and stopped cleanly — the
~122.7M-token result is short of the 130M-token *planning target* by ~5.6% (the true combined
pool across these sources, once fully scraped, was smaller than the pre-scrape estimate), but see
§6 for why this doesn't threaten the assignment's actual (ratio-based) requirement.

## 5. Preprocessing pipeline

### 5.1 Scope decision: Tier 1 only

Presented with a reference pipeline (Data Collection → Language Filtering → Text Metric →
Repetitive-Doc Removal → Rule-Based Correction → Perplexity Filtering → Document Quality →
Paragraph Dedup → MinHash Dedup → Exact Dedup → Semantic Filtering → Topic Filtering → Safety
Filtering), scoped into two tiers and **Tier 2 was dropped entirely** by decision:

- **Tier 1 (built):** language/script filtering, text-metric + quality filtering, repetitive-doc
  removal, rule-based correction, exact dedup (across all sources combined — new, since manual
  sources had previously only deduped against themselves), paragraph-level dedup, MinHash
  near-duplicate dedup, safety filtering.


### 5.2 Line-level script filtering (not document-level)

Some scraped articles have an English sentence mixed in as a heading. Fix: apply the
≥75%-native-script-character rule (matching IndicCorpV2's own rule, §3.2) at **line/sentence
granularity within each document**, not at the whole-document level. A whole-document check would
miss this — a 95%-Devanagari document with one English heading still passes overall, and the
heading slips through. Per-line checking drops just the offending line, keeping the rest of the
document intact.

### 5.3 Safety filter: sourcing a real Devanagari blocklist

The first candidate word list found (LDNOOBW's Hindi list) turned out to be **romanized** (Latin
script — `chutiya`, `gandu`, etc.) and would never match anything in a Devanagari corpus; it was
a silent no-op. Checked several more open lists for both languages; all were either romanized, or
(for the largest Nepali candidate) **contaminated with political/ethnic terms** — proper nouns
like a former PM's name, a political party name, and region/ethnicity words mixed in as if they
were profanity, which would have wrongly rejected large amounts of entirely normal news content.

**Fix:** hand-transliterated a curated subset of the reviewed romanized list into Devanagari
(`scripts/preprocessing/wordlists/offensive_hi.txt`, 66 entries), deliberately **excluding**
common words with legitimate literal meanings that happened to appear in source lists as slang
(e.g. साला/brother-in-law, पटाखा/firecracker, सूअर alone/pig, भालु/bear) to avoid false-positive
filtering of ordinary content. Verified end-to-end: a document containing a genuine offensive
term is correctly rejected; ordinary text passes. Nepali's list
(`offensive_ne.txt`) is short by design — reliable Nepali-specific resources were scarce — and is
unioned with the Hindi list at load time, since the languages share considerable vocabulary
including profanity. Documented limitation: Nepali-specific coverage is not comprehensive.

### 5.4 Engineering: a full pipeline rewrite after a memory crash

The first implementation read the entire ~4GB public-data file into memory (`f.read()`) and split
it into millions of Python string objects — on a machine with only ~2.7GB free RAM, this crashed
VSCode. **Rewrote the whole pipeline as fully streaming**: every stage is a generator reading
one document at a time; intermediate results between stages are written to JSONL files on disk,
never held as a single in-memory list; dedup stages keep only hashes (not document text)
resident in RAM; even the final shuffle+split (which naively would need every surviving
document's text in memory to shuffle) was redesigned as a two-pass approach — pass 1 counts lines
and builds a compact per-line split-assignment array (single characters, a few MB even at
millions of documents), pass 2 streams the input once and routes each line straight to its
assigned output file. Tested under an explicit `ulimit -v` memory cap before any full-scale run.

### 5.5 Performance: MinHash dedup

Initial unoptimized MinHash (num_perm=32, unlimited shingles per document) measured at
5.43ms/document — projected to **~8.4 hours** for the full ~5.6M-document corpus, infeasible.
Optimized in two steps: (1) reduced `num_perm` to 16 and capped shingles per document to 40
(randomly sampled, not just the first 40) — an approximate/sampled shingle set, which is a
reasonable tradeoff since LSH near-dup detection is already an approximate technique — bringing
this to ~1.1ms/doc (~1.7h projected); (2) parallelized the per-document signature computation
(the measured bottleneck) across worker processes, while keeping LSH insert/query single-threaded
in the main process (duplicate detection must see documents in a stable order). Measured
speedup was ~1.7x (0.65ms/doc, ~1h projected) rather than close to the core count — confirmed via
isolated benchmarking that the LSH bookkeeping itself, which doesn't parallelize, is a
significant fraction of the per-document cost (Amdahl's law), not per-batch dispatch overhead
(tested larger batch sizes; no further improvement).

### 5.6 Splits

95% / 2.5% / 2.5% train/val/test, assigned per-document at random (not drawn from contiguous
blocks of the corpus, avoiding source- or time-ordering bias between splits).

### 5.7 Status

Preprocessing was running in the background at the time of writing this report. Final post-dedup
counts and the split sizes will be appended here once it completes — see `stats.json` under each
language's `data/processed/` directory for the authoritative final numbers.

## 6. Combined dataset statistics and the ≥20%-manual requirement

Using the pre-preprocessing token estimates (§3.3, §4.4) — final post-dedup counts will be
somewhat lower once §5's dedup passes finish, which is expected and is the entire point of
running them:

| Language | Manual tokens | Public tokens | Total | Manual % |
|---|---:|---:|---:|---:|
| Hindi | 130,000,083 | 400,088,537 | 530,088,620 | **24.5%** |
| Nepali | 122,661,776 | 400,015,643 | 522,677,419 | **23.5%** |

Both comfortably clear the assignment's **≥20% manual** requirement, despite Nepali's manual
portion landing ~5.6% under its own *internal* 130M-token planning target — that target was never
the actual requirement, just a convenient equal split assumption made before the real achievable
pool size was known.

## 7. Known limitations (documented, not silently glossed over)

- Tier 2 filtering (perplexity, semantic, topic) was scoped out entirely — see §5.1 for why.
- The safety-filter blocklist is hand-curated and conservative, not comprehensive, especially for
  Nepali (§5.3).
- MinHash dedup is deliberately approximate (num_perm=16, capped shingles) for runtime reasons
  (§5.5) — it will not catch every near-duplicate, but is applied on top of exact and
  paragraph-level dedup, which cover the exact-overlap cases precisely.
- `nagariknews` is a "dead" source (0 articles across the whole project) kept in the source list
  only for transparency about what was tried.
- IndicCorpV2's own deduplication step is undocumented (§3.2) — our own dedup pass in §5 partly
  compensates by running exact + paragraph + MinHash dedup across manual and public data
  *combined*, not just within manual sources.
