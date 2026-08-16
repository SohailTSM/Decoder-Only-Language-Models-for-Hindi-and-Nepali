# Phase 1 — Dataset Collection, Preprocessing & Tokenizer Report

Model H = Hindi (higher-resource). Model L = Nepali (lower-resource, from the assignment's
allowed list). This report covers dataset collection, non-overlap methodology, preprocessing,
and tokenizer construction for both, plus every non-obvious decision made along the way and why.

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

Initial target: **400M tokens per language** from the public portion. Converted to a character
count using the chars-per-token estimates derived in §4.2 (same estimates used for both the
scrape stop-condition and this download), then streamed only that many characters from
`hi-1.txt` / `ne.txt` — incremental UTF-8 decoding (so a multi-byte character split across
download chunks is never corrupted) and only complete lines written (a trailing partial line is
held back, never truncated into the output).

**Initial result (400M-token target):**

| Language | Chars downloaded | Tokens (est.) |
|---|---|---|
| Hindi | 1,600,354,148 | ~400,088,537 |
| Nepali | 1,800,070,395 | ~400,015,643 |

**Revised upward after a sufficiency check.** Once Hindi's preprocessing reached stage 2
(cleaning + exact + paragraph dedup, before MinHash), the surviving corpus measured
1,646,140,837 chars (~411.5M tokens) against a combined raw input of 2,120,354,480 chars
(manual + public) — a **77.63% survival rate**. Extrapolating that rate, the original 400M-token
public target would land the *final* corpus meaningfully under the assignment's ~500M-token
target (roughly 370-410M final, depending on how much more MinHash removes on top). To compensate:
raw-token target raised to **520M tokens per language** (500M ÷ 0.7763 ≈ 644M combined raw needed;
manual tokens are fixed since those sources are exhausted — 130.0M Hindi / 122.7M Nepali — so the
gap is made up entirely on the public side: ≈514-521M tokens depending on language, rounded to
520M for both for simplicity). Re-downloaded from scratch to the new target (restarting rather
than resuming mid-stream — see the download script's own reasoning: not worth the complexity of a
byte-range resume for a multi-GB stream) after hardening it with retry+backoff (this download had
by then hit two real transient network failures over the course of the project — a DNS resolution
failure and a `ReadTimeout` — with no retry logic at the time; both are now handled automatically).

**Final result (520M-token target):**

| Language | Chars downloaded | Tokens (est.) |
|---|---|---|
| Hindi | 2,080,379,402 | ~520,094,850 |
| Nepali | 2,340,313,117 | ~520,069,582 |

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
§7 for why this doesn't threaten the assignment's actual (ratio-based) requirement.

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
(`scripts/preprocessing/wordlists/offensive_hi.txt`), deliberately **excluding** common words
with legitimate literal meanings that happened to appear in source lists as slang (e.g.
साला/brother-in-law, पटाखा/firecracker, सूअर alone/pig, भालु/bear) to avoid false-positive
filtering of ordinary content. Nepali's list (`offensive_ne.txt`) is short by design — reliable
Nepali-specific resources were scarce — and is unioned with the Hindi list at load time, since the
languages share considerable vocabulary including profanity.

**Two further bugs found once real preprocessing runs surfaced actual false positives:**

1. **Substring matching, not word matching.** `passes_safety_filter` originally checked
   `word in lowered_text` — plain substring containment. On a live sample this silently rejected
   5,000-document batches of entirely clean content: the slur "रांड" matched as a substring
   inside "ग्रांडसन" (grandson, from a Bollywood movie title), "मिरांडा" (a footballer's
   surname), "मरांडी" (a common surname — including a former Jharkhand CM), and "ब्रांडेड"
   (branded). Tried Python's regex `\b` word-boundary first; found it **unreliable on
   Devanagari** (it still matched inside some of the same false positives, likely due to how
   combining vowel signs interact with the Unicode word-character class). Fixed with explicit
   whitespace tokenization + whole-token set lookup for single-word entries, and
   contiguous-token-sequence matching for multi-word phrases — verified against all four false
   positives above (all now correctly pass) and the genuine slur (still correctly rejected).
2. **Two classes of bad word choices in the list itself**, found by inspecting what was actually
   triggering rejections on real data: (a) **`बलात्कार`/`बलात्कारी`** (rape/rapist) — standard
   legal/journalistic vocabulary for reporting on sexual-assault crimes, not profanity; keeping
   them would have silently deleted an entire category of legitimate crime journalism. (b)
   **`छक्का`** and **`गश्ती`** — real vulgar slang senses, but both are also extremely common
   *innocent* words (छक्का = "six" as in a cricket six; गश्ती = police "patrol") that a
   context-free keyword blocklist can't disambiguate; removed since the false-positive cost
   clearly outweighs the benefit for words this common. Also removed on a full list review (not
   triggered by data, just recognized as a categorically different problem): **`भंगी`**,
   **`कंजर`**, **`हिजड़ा`/`हिजडा`** — these are caste/tribe/gender-identity community names, not
   profanity; including them both risks rejecting legitimate journalism about caste issues or
   transgender rights, and is itself the kind of harm a "safety" filter shouldn't encode.

Verified after all fixes on a 5,000-document live sample of public data: safety-filter false
positives dropped from 21/5,000 (0.42%) to **0/5,000**, with the length/quality filter (§5.1,
unrelated) still correctly removing the genuinely-short entries.

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

### 5.6 A second crash: unbounded LSH memory growth mid-run

Even after the speed fix in §5.5, a full production run on Hindi's ~3.8M-document stage-2 output
crashed with a clean Python `MemoryError` after ~1,133s, right near the end (2.4M of 3.8M
documents processed). Root cause: `MinHashLSH`'s memory grows without bound as more signatures
are inserted — measured ~671 bytes/document from the run's own RSS log (587MB at 200K documents
inserted, 2,064MB at 2.4M) — and at this document count the *live* LSH index in the main process
alone approached the 3GB `ulimit -v` safety cap (itself put in place after the earlier crash in
§5.4) before the stage could finish.

**Fix (v2, superseded by §5.7 below):** the LSH index was reset to a fresh, empty one every
800,000 inserted documents, bounding its memory to a fixed ceiling regardless of total corpus size
(≈800,000 × 671 bytes ≈ 537MB peak for the index itself). This worked (no crash), but the
tradeoff was real, not just theoretical: near-duplicates landing in *different* 800K-document
shards were never compared against each other — with ~6 shards for Hindi's corpus, a true
near-duplicate pair had only roughly a 1-in-6 chance of landing in the same shard, so an estimated
~83% of cross-shard near-duplicates (e.g. different outlets covering the same event, which is
exactly the motivating case for adding MinHash in the first place — see §4.3's "buffer for
cross-source overlap") were silently missed. Recovery from the crash used a `--resume-from-stage2`
flag added to the pipeline specifically for this: since stages 1-2 had already completed
successfully and their output was intact on disk, recovery skipped straight to a fresh stage 3
rather than re-running the ~14 minutes of already-completed work — this flag remained useful for
every subsequent stage-3 rewrite too.

### 5.7 Doing MinHash properly: signature-and-bucket LSH (v3)

The sharding compromise in §5.6 was accepted as a fast fix under time pressure, not because it was
the right technique. Replaced with the standard way LSH deduplication is actually done at scale:

1. Compute a MinHash signature per document (same parallelized computation as before) and store
   **only the signatures**, not a live growing index — as one compact NumPy array
   (N × 16 uint64 ≈ 128 bytes/document, e.g. ~1GB for Nepali's ~7.8M documents), which uses *less*
   memory than the v1 live LSH index did at a fraction of that document count.
2. For each of 4 LSH bands (4 rows/band from num_perm=16, giving an implied similarity threshold
   of ≈0.71), bucket documents by their band's hash value using `numpy.lexsort` — a vectorized,
   C-level sort, not a Python-level `hash()` call per document — then scan the sorted array for
   runs of identical band values (candidate near-duplicate groups).
3. Union same-bucket documents via a plain array-backed union-find (disjoint-set), across *all*
   bands and the *entire* corpus at once — no arbitrary shard boundary. A document is kept iff it
   is the smallest-index representative of its connected component (the first-seen copy of each
   near-duplicate cluster survives; every later copy is dropped).
4. A second streaming pass over the input writes only the kept documents.

This gives full corpus-wide near-duplicate coverage with *better* memory bounds than either
previous version, because it never holds a live, ever-growing Python object graph — just compact
arrays that are discarded once no longer needed.

**A real bug found while testing this, via a synthetic near-duplicate, not just eyeballing
aggregate counts:** took a real document, changed only its first 5 words, and confirmed the
pipeline should flag it as a near-duplicate of the original. It didn't. Root cause: the
shingle-sampling function (`_shingles`, used to cap shingles/document to 40 for speed, §5.5)
seeded its random sample from `hash(first_shingle)` — since the edit changed the very first
shingle, the near-duplicate document sampled a *completely different* random 40-shingle subset
than the original, even though the rest of the document was identical, and their MinHash
signatures ended up looking unrelated. **Fix:** replaced position-dependent random sampling with
deterministic bottom-k selection — sort all shingles by their own hash value and keep the lowest
40. Since this depends only on each shingle's own content, not the document's position or context,
two documents sharing most of their shingles will independently select mostly the same subset
regardless of *where* they were edited. Re-ran the synthetic test after the fix: the near-duplicate
was correctly caught (removed), and a genuinely unrelated document in the same test batch
correctly survived.

### 5.8 Splits and the final token-count cap

95% / 2.5% / 2.5% train/val/test, assigned per-document at random (not drawn from contiguous
blocks of the corpus, avoiding source- or time-ordering bias between splits).

**Cap added:** both languages' preprocessing survival rate (§3.3) turned out better than the
pre-run estimate, and after the public-data top-up both finished meaningfully over the ~500M-token
target (Hindi ~520.5M, Nepali ~589.8M from the pre-fix v2 run). Capped the final corpus to
**~510M tokens** per language — comfortably above 500M without overshooting by as much as the
uncapped runs did. Implemented in `stage4_split`: documents are visited in the same random order
used for the train/val/test shuffle, and included until the running character total reaches the
target (510M tokens × the language's chars/token estimate, §4.2) — a genuine random sample of the
corpus, not just "the first N documents by file order" — with any remaining documents beyond the
cap excluded entirely rather than assigned to a split.

### 5.9 Final results

Both languages completed the full pipeline (v3 MinHash, 510M-token cap). The v3 rewrite's
coverage difference versus the v2 sharded approach is stark and language-dependent: Hindi's
near-dup rate was 0.55% (4,928,475 → 4,901,226, 27,249 removed — 12.6x more than v2's sharded
result on the same stage-2 input); Nepali's was **4.58%** (7,835,734 → 7,476,841, 358,893
removed — 31x more than v2 caught). The much higher Nepali rate is consistent with, not
surprising given, the cross-source overlap concern raised back in §4.3 (the reason the Nepali
scrape's collection buffer was raised in the first place) — v3's full corpus-wide coverage is
catching exactly the kind of cross-outlet duplication v2's shard boundaries were structurally
blind to.

| Language | Train | Val | Test | Total chars | Total tokens (est.) | Target |
|---|---:|---:|---:|---:|---:|---:|
| Hindi | 4,581,781 docs / 1,937,829,626 chars | 120,573 / 51,055,460 | 120,574 / 51,115,161 | 2,040,000,247 | ~510,000,062 | 510M |
| Nepali | 6,387,191 docs / 2,180,306,282 chars | 168,083 / 57,319,438 | 168,085 / 57,374,283 | 2,295,000,003 | ~510,000,001 | 510M |

The token-budget cap (§5.8) landed within a few hundred characters of the exact target for both
languages (247 chars over for Hindi, 3 chars over for Nepali, both against a ~2 billion-character
target — effectively exact).

**Manual/public provenance is not tracked through this pipeline** — `iter_all_documents` yields a
`(source, text)` tuple, but stage 1 discards the source tag when writing its output, so the exact
post-processing manual-token count isn't directly available from `stats.json`. The §7 ratio below
is therefore still the pre-processing estimate from the raw collected counts, not a post-dedup
measurement; treat it as an approximation, not the authoritative number, unless/until
source-tracking is added and the pipeline is re-run.

## 6. Tokenizer construction

SentencePiece Unigram, 32,000 vocab, trained **independently per language** on that language's
own `train.txt` only — no shared vocabulary, no pretrained tokenizer, per the assignment's
constraints. `character_coverage=0.9995`, standard special-token ids
(`pad=0, unk=1, bos=2, eos=3`), shuffled sampling.

### 6.1 The memory story: why the training sample is far smaller than planned

The original plan was to sample 5,000,000 sentences from each language's `train.txt` (standard
practice — SentencePiece's own docs note that a large shuffled subsample trains a vocabulary
just as well as reading the full multi-million-line corpus, at a fraction of the cost). In
practice, the SentencePiece unigram trainer's EM/seed-vocabulary phase (immediately after
`"Done! preprocessed N sentences..."` in its logs) needs dramatically more memory than the raw
sampled text, and scales worse than linearly with sentence count:

| Attempt | Environment | RAM budget | Sentences | Result |
|---|---|---:|---:|---|
| 1 | Local | 7.6GB total | 5,000,000 | OS OOM-killer killed the process outright (~4.9GB RSS and still climbing); destabilized the whole desktop enough that `gnome-shell` itself was also invoking the OOM killer |
| 2 | Local, `ulimit -v` capped | 3GB (explicit cap) | 1,000,000 | Clean `std::bad_alloc` right after preprocessing 292MB of text, before the EM phase could proceed |
| 3 | Local, `ulimit -v` capped | 3GB (explicit cap) | 150,000 | **Succeeded** — trained and evaluated with healthy metrics |
| 4 | Colab free tier | 12.7GB | 2,000,000 | Kernel OOM-crashed at the identical log transition point (right after `"Done! preprocessed 1,992,402 sentences"`) |
| 5 | Colab free tier | 12.7GB | 1,000,000 | **Succeeded** — used for the final models reported below |

The jump from attempt 2 to attempt 4 is the most informative data point: doubling the RAM budget
(3GB → 12.7GB, a 4.2x increase) was still not enough to support 2x the sentence count. This
suggests the unigram trainer's EM-phase memory use scales closer to quadratically than linearly
with corpus size on this SentencePiece version — a genuine constraint of the tool on
memory-limited hardware, not a tuning oversight. The explicit `ulimit -v` caps used locally were a
deliberate choice too: letting the process hit the cap and fail cleanly (`MemoryError`) is far
safer than letting the Linux OOM-killer pick a victim process system-wide, which is what happened
in attempt 1.

The final training runs (attempt 5) used **1,000,000 sentences** per language — well short of the
original 5M target, but the largest value that reliably completed given the hardware actually
available. To gauge whether this mattered: the 150k-sample Hindi model (attempt 3, local) and the
1M-sample Hindi model (attempt 5, Colab, reported below) produced nearly identical fertility, UNK
rate, and avg-chars-per-token — suggesting the vocabulary was already fairly stable well before
1M sentences, which is some reassurance that the sample-size shortfall isn't materially hurting
tokenizer quality, though it may still under-represent rarer domain-specific subwords relative to
a full-corpus or 5M-sentence training run.

### 6.2 Final results

Evaluated on each language's held-out `val.txt` (never seen during training):

| Metric | Hindi | Nepali |
|---|---:|---:|
| Vocab size | 32,000 | 32,000 |
| Training sample size | 1,000,000 sentences | 1,000,000 sentences |
| Training sample as % of `train.txt` | ~9.4% (of 10,622,764 lines) | ~6.8% (of 14,691,412 lines) |
| Eval lines sampled (from val.txt) | 158,729 | 200,000 |
| Fertility (tokens/word) | 1.1575 | 1.2499 |
| UNK rate | 0.241% | 0.214% |
| Avg chars/token | 4.4084 | 5.1823 |

Both languages show fertility close to 1 (most words map to a single token or a small handful of
subword pieces, rather than excessive fragmentation) and UNK rates well under 1%, indicating good
coverage of each language's held-out text despite the reduced training sample size. Nepali's
higher fertility and avg-chars-per-token relative to Hindi is consistent with Nepali's smaller,
more repetitive collected corpus (§4.4) producing a slightly less efficient subword vocabulary at
the same 32k budget.

Output files: `hindi/tokenizer/hindi_unigram_32k.{model,vocab}`,
`nepali/tokenizer/nepali_unigram_32k.{model,vocab}`,
`report/{hindi,nepali}_tokenizer_stats.json` (full stats: top-20 tokens, example
tokenizations).

## 7. Combined dataset statistics and the ≥20%-manual requirement

Using the raw (pre-preprocessing) token counts after the §3.3 top-up — **⚠️ this ratio needs
active monitoring, not just a passing mention:**

| Language | Manual tokens | Public tokens | Total | Manual % |
|---|---:|---:|---:|---:|
| Hindi | 130,000,083 | 520,094,850 | 650,094,933 | **20.00%** |
| Nepali | 122,661,776 | 520,069,582 | 642,731,358 | **19.08%** |

Raising the public-data target to compensate for preprocessing loss (§3.3) fixed the *absolute*
token-count shortfall but pushed the manual **ratio** down — Hindi is now sitting exactly on the
20% line and Nepali has dropped **below** the assignment's required minimum, using raw counts.
Manual sources are exhausted (§4.4) and can't easily be grown further, so the lever here is on the
public side, or on carrying manual content through preprocessing more gently than public content.
The true, decisive number is the **post-preprocessing** ratio once §5 finishes for both languages
(manual and public content go through the identical pipeline, so the ratio could shift either way
depending on relative survival rates) — this table must be recomputed with real post-dedup counts
before Phase 1 is considered done, and if it's still under 20%, the public download target needs
to come back down (or manual needs a genuinely new source) rather than just noting the shortfall.

## 8. Known limitations (documented, not silently glossed over)

- **Open, unresolved as of writing:** the ≥20%-manual requirement is no longer comfortably clear
  of the line (§7) after the public-data top-up — Nepali is currently under it on raw counts.
  Needs the real post-preprocessing ratio and, if still short, a correction to the public target.
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
- The tokenizer training sample (§6.1) is far smaller than originally planned (1,000,000
  sentences vs. a 5,000,000-sentence target) due to memory constraints on both the local machine
  and the Colab free tier — evaluation metrics look healthy, but rarer domain-specific subwords
  may be under-represented in the learned vocabulary relative to a full-corpus training run.
