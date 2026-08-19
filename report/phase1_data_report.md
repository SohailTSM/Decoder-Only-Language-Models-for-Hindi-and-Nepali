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

### 3.1 Is it shuffled? (matters because I only download a truncated prefix, not the whole file)

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

| Applied                                                                                                                | NOT applied                                                           |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Language ID filtering (paragraph-level, cld3+langdetect)                                                               | Repetitive document removal                                           |
| Script-purity filter (≥75% native-script chars, **sentence-level**)                                                    | Perplexity filtering                                                  |
| Length filter (<10 words after punctuation stripped → dropped)                                                         | Document quality filtering beyond length                              |
| Offensive-word blocklist filtering (17 languages)                                                                      | Paragraph/MinHash near-dup dedup, semantic filtering, topic filtering |
| _Some_ deduplication (paper's Table 3 calls it a "de-duplicated corpus" but gives **no methodological detail** on how) | —                                                                     |

This matters for two things: (a) it's why my own cleaning pipeline (§5) is still worth applying to
the public portion, not just manual; (b) the sentence-level script-purity rule (≥75% native
script) is the exact rule my own cleaning pipeline reuses.

### 3.3 Download method and target

Initial target: **400M tokens per language** from the public portion. Converted to a character
count using the chars-per-token estimates derived in §4.2 (same estimates used for both the
scrape stop-condition and this download), then streamed only that many characters from
`hi-1.txt` / `ne.txt` — incremental UTF-8 decoding (so a multi-byte character split across
download chunks is never corrupted) and only complete lines written (a trailing partial line is
held back, never truncated into the output).

**Initial result (400M-token target):**

| Language | Chars downloaded | Tokens (est.) |
| -------- | ---------------- | ------------- |
| Hindi    | 1,600,354,148    | ~400,088,537  |
| Nepali   | 1,800,070,395    | ~400,015,643  |

**Revised upward after a sufficiency check.** Once Hindi's preprocessing reached stage 2
(cleaning + exact + paragraph dedup, before MinHash), the surviving corpus measured
1,646,140,837 chars (~411.5M tokens) against a combined raw input of 2,120,354,480 chars
(manual + public) — a **77.63% survival rate**. Extrapolating that rate, the original 400M-token
public target would land the _final_ corpus meaningfully under the assignment's ~500M-token
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
| -------- | ---------------- | ------------- |
| Hindi    | 2,080,379,402    | ~520,094,850  |
| Nepali   | 2,340,313,117    | ~520,069,582  |

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
the _identical_ timestamp `2023-02-05`, which is a red flag for a static/batch value rather than
a real per-article date. Verified by checking real article content against real-world events: an
article the sitemap claimed was "2023-02-05" was actually about Brazil sacking Dunga and
appointing Tite as coach — **June 2016**. This meant the site's own cutoff filtering was not
actually restricting to 2023+ content; the scraper had been walking essentially the site's entire
history (spanning back to ~2016), which breaks the non-overlap guarantee in §2.

Investigated a proper fix (parsing the real per-article Nepali Bikram Sambat date embedded in
each article's HTML, using the `nepali_datetime` library — validated it correctly converts BS
dates and correctly identified both the 2016 article and a confirmed-2020 COVID-era article as
pre-cutoff). **Reverted this in favor of a simpler fix** — I had independently verified the
sitemap's actual paginated depth and opted for a hard cap instead: **only follow sitemap pages
`p ≤ 400`** (implemented as a general `max_sitemap_page` filter in `sitemap_utils.py`, applied
only to `annapurnapost`). Simpler, and sufficient given that verification.

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
applied _within_ one source's own output, never across sources. The scrape's shared stop-budget
was raised (130M-token equivalent moved from 585M to **650M characters**, i.e. an inflated
`chars/token` used only as an operational ceiling, not a literal claim) so no source would stop
collecting early — each source ran to its own natural pool exhaustion instead.

### 4.4 Results

| Language         | Source          |    Articles |                                 Chars |
| ---------------- | --------------- | ----------: | ------------------------------------: |
| Hindi            | bhaskar         |       3,520 |                                     — |
| Hindi            | amarujala       |         820 |                                     — |
| Hindi            | jagran          |       2,614 |                                     — |
| Hindi            | navbharattimes  |       1,041 |                                     — |
| Hindi            | prabhatkhabar   |         768 |                                     — |
| Hindi            | abplive         |      12,386 |                                     — |
| Hindi            | abplive_archive |     178,839 |                                     — |
| **Hindi total**  |                 | **199,988** | **520,000,332 (~130,000,083 tokens)** |
| Nepali           | khabarhub       |     136,693 |                           228,577,365 |
| Nepali           | nepalpress      |     108,133 |                           198,119,929 |
| Nepali           | annapurnapost   |      39,248 |                            81,980,046 |
| Nepali           | deshsanchar     |      29,798 |                            43,300,650 |
| Nepali           | nagariknews     |           0 |                                     0 |
| **Nepali total** |                 | **313,872** | **551,977,990 (~122,661,776 tokens)** |

![Per-source article counts for Hindi (dominated by the abplive_archive date-archive) and per-source character counts for Nepali (khabarhub and nepalpress dominate; nagariknews contributed nothing).](figures/01_manual_sources.png)

All 5 Nepali sources exhausted their available post-2023 pools and stopped cleanly — the
~122.7M-token result is short of the 130M-token _planning target_ by ~5.6% (the true combined
pool across these sources, once fully scraped, was smaller than the pre-scrape estimate), but see
§7 for why this doesn't threaten the assignment's actual (ratio-based) requirement.

## 5. Preprocessing pipeline

### 5.1 Scope decision: what's actually in the pipeline

Out of a broader menu of possible filtering/cleaning stages (language filtering, text-metric
filtering, repetitive-document removal, rule-based correction, perplexity filtering, document
quality filtering, paragraph dedup, MinHash dedup, exact dedup, semantic filtering, topic
filtering, safety filtering), I scoped this pipeline down to: language/script filtering,
text-metric + quality filtering, repetitive-doc removal, rule-based correction, exact dedup
(across all sources combined — new, since manual sources had previously only deduped against
themselves), paragraph-level dedup, MinHash near-duplicate dedup, and safety filtering.

Every document also passes through **Unicode NFC normalization** (`unicodedata.normalize("NFC",
text)`, `clean.py`) before any filtering runs, so Devanagari text using visually-identical but
differently-encoded codepoint sequences (a real issue in scraped/mixed-source text) is
canonicalized before script-purity checks, dedup hashing, or the tokenizer ever see it.

### 5.2 Line-level script filtering (not document-level)

Some scraped articles have an English sentence mixed in as a heading. Fix: apply the
≥75%-native-script-character rule (matching IndicCorpV2's own rule, §3.2) at **line/sentence
granularity within each document**, not at the whole-document level. A whole-document check would
miss this — a 95%-Devanagari document with one English heading still passes overall, and the
heading slips through. Per-line checking drops just the offending line, keeping the rest of the
document intact.

### 5.3 Safety filter: sourcing a real Devanagari blocklist

Hand-transliterated a curated subset of the reviewed romanized list into Devanagari
(`scripts/preprocessing/wordlists/offensive_hi.txt`), deliberately **excluding** common words
with legitimate literal meanings that happened to appear in source lists(downloaded from the internet in latin script) as slang (e.g. साला/brother-in-law, पटाखा/firecracker, सूअर/pig, भालु/bear) to avoid false-positive
filtering of ordinary content. Nepali's list (`offensive_ne.txt`) is short by design — reliable
Nepali-specific resources were scarce — and is unioned with the Hindi list at load time, since the
languages share considerable vocabulary including profanity.

**Matching method:** `passes_safety_filter` tokenizes each document on whitespace and checks for
whole-token membership against the blocklist set (single-word entries) or contiguous-token-sequence
matches (multi-word phrases) — not substring containment, and not a regex `\b` word boundary
(found unreliable on Devanagari). The wordlist itself was also reviewed to remove entries that
were either legitimate non-slang vocabulary or caste/tribe/gender-identity community names rather
than profanity.

### 5.4 Streaming pipeline design

Every stage is a generator that processes one document at a time; intermediate results between
stages are written to JSONL files on disk, never held as a single in-memory list; dedup stages
keep only hashes (not document text) resident in RAM; the final shuffle+split uses a two-pass
approach — pass 1 counts lines and builds a compact per-line split-assignment array, pass 2
streams the input once and routes each line straight to its assigned output file — rather than
holding every surviving document in memory to shuffle. Tested under an explicit `ulimit -v`
memory cap before any full-scale run.

### 5.5 MinHash near-duplicate deduplication

Signature-and-bucket LSH with union-find, applied across the entire combined (manual + public)
corpus at once:

1. Compute a MinHash signature per document — `num_perm=16`, shingles capped to 40 per document
   via deterministic bottom-k selection (sort each document's shingles by their own hash value,
   keep the lowest 40; content-dependent, not position-dependent, so two near-duplicate documents
   independently select mostly the same shingle subset regardless of where they differ) — computed
   in parallel across worker processes, stored as one compact NumPy array (N × 16 uint64).
2. For each of 4 LSH bands (4 rows/band, giving an implied similarity threshold of ≈0.71), bucket
   documents by their band's hash value via `numpy.lexsort` (a vectorized, C-level sort), then scan
   the sorted array for runs of identical band values — candidate near-duplicate groups.
3. Union same-bucket documents via an array-backed union-find (disjoint-set), across all bands and
   the entire corpus at once. A document is kept iff it is the smallest-index representative of its
   connected component (the first-seen copy of each near-duplicate cluster survives).
4. A second streaming pass over the input writes only the kept documents.

**Result:** full corpus-wide near-duplicate coverage with bounded, predictable memory (compact
arrays only — no live, ever-growing index) — see §5.7 for the exact per-language removal counts.

### 5.6 Splits and the final token-count cap

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

### 5.7 Final results

Both languages completed the full pipeline (v3 MinHash, 510M-token cap). The v3 rewrite's
coverage difference versus the v2 sharded approach is stark and language-dependent: Hindi's
near-dup rate was 0.55% (4,928,475 → 4,901,226, 27,249 removed — 12.6x more than v2's sharded
result on the same stage-2 input); Nepali's was **4.58%** (7,835,734 → 7,476,841, 358,893
removed — 31x more than v2 caught). The much higher Nepali rate is consistent with, not
surprising given, the cross-source overlap concern raised back in §4.3 (the reason the Nepali
scrape's collection buffer was raised in the first place) — v3's full corpus-wide coverage is
catching exactly the kind of cross-outlet duplication v2's shard boundaries were structurally
blind to.

![Final MinHash near-duplicate rate at stage 3: 0.55% for Hindi versus 4.58% for Nepali, roughly an 8x difference.](figures/02_near_dup_rate.png)

| Language |                                Train |                  Val |                 Test |   Total chars | Total tokens (est.) | Target |
| -------- | -----------------------------------: | -------------------: | -------------------: | ------------: | ------------------: | -----: |
| Hindi    | 4,581,781 docs / 1,937,829,626 chars | 120,573 / 51,055,460 | 120,574 / 51,115,161 | 2,040,000,247 |        ~510,000,062 |   510M |
| Nepali   | 6,387,191 docs / 2,180,306,282 chars | 168,083 / 57,319,438 | 168,085 / 57,374,283 | 2,295,000,003 |        ~510,000,001 |   510M |

The token-budget cap (§5.6) landed within a few hundred characters of the exact target for both
languages (247 chars over for Hindi, 3 chars over for Nepali, both against a ~2 billion-character
target — effectively exact).

**Manual/public provenance is not tracked through this pipeline** — `iter_all_documents` yields a
`(source, text)` tuple, but stage 1 discards the source tag when writing its output, so the exact
post-processing manual-token count isn't directly available from `stats.json`. The §7 ratio below
is therefore still the pre-processing estimate from the raw collected counts, not a post-dedup
measurement; treat it as an approximation, not the authoritative number, unless/until
source-tracking is added and the pipeline is re-run.

## 6. Tokenizer construction

SentencePiece Unigram, 16,000 vocab, trained **independently per language** on that language's
own `train.txt` only — no shared vocabulary, no pretrained tokenizer, per the assignment's
constraints. `character_coverage=0.9995`, standard special-token ids
(`pad=0, unk=1, bos=2, eos=3`), shuffled sampling.

### 6.1 Vocab size selection: fertility/UNK sweep

The assignment specifies vocab size in the "tens of thousands" and says to choose it "using
fertility / unknown-token rate on held-out text" — so rather than picking 32,000 by convention, I
trained 7 candidate vocab sizes (4,000 to 48,000) per language and measured fertility and UNK rate
on the same held-out `val.txt` sample for each. This sweep used the 150,000-sentence sample size
(not the final 1M-sample models, §6.3) since it needed 14 separate training runs and 150k is the
value already proven to train reliably within this machine's memory limits (`scripts/tokenizer/vocab_size_sweep.py`).

**Hindi:**

| Vocab size | Fertility (tokens/word) |   UNK rate | Δ fertility vs. previous size |
| ---------: | ----------------------: | ---------: | ----------------------------: |
|      4,000 |                  1.4516 |     0.192% |                             — |
|      8,000 |                  1.3017 |     0.214% |                       −0.1499 |
| **16,000** |              **1.2099** | **0.231%** |                   **−0.0918** |
|     24,000 |                  1.1766 |     0.237% |                       −0.0333 |
|     32,000 |                  1.1598 |     0.241% |                       −0.0168 |
|     40,000 |                  1.1500 |     0.243% |                       −0.0098 |
|     48,000 |                  1.1438 |     0.244% |                       −0.0062 |

**Nepali:**

| Vocab size | Fertility (tokens/word) |   UNK rate | Δ fertility vs. previous size |
| ---------: | ----------------------: | ---------: | ----------------------------: |
|      4,000 |                  1.7126 |     0.153% |                             — |
|      8,000 |                  1.4937 |     0.175% |                       −0.2189 |
| **16,000** |              **1.3480** | **0.194%** |                   **−0.1457** |
|     24,000 |                  1.2879 |     0.203% |                       −0.0601 |
|     32,000 |                  1.2537 |     0.209% |                       −0.0342 |
|     40,000 |                  1.2312 |     0.213% |                       −0.0225 |
|     48,000 |                  1.2155 |     0.216% |                       −0.0157 |

![Fertility (tokens per word) versus vocab size for Hindi and Nepali, from 4,000 to 48,000. Both curves drop steeply through 16k, then flatten — the chosen 16,000 sits just past the elbow.](figures/03_vocab_fertility.png)

**On fertility/UNK alone, 24k-32k is the better elbow** — both languages show the same shape:
fertility drops steeply from 4k through 16k-24k, then flattens (Hindi: −0.017 at the 24k→32k
step, shrinking to −0.006 by 40k→48k; Nepali: −0.034 shrinking to −0.016 over the same range).
UNK rate moves the _opposite_ direction as vocab grows — it rises slightly (Hindi 0.192%→0.244%,
Nepali 0.153%→0.216%) since a larger vocabulary spreads probability mass across more, rarer
pieces — though the absolute rates stay negligible throughout (<0.25%) regardless of size.

**Why 16,000 was chosen instead of 32,000.** The fertility/UNK sweep by itself favored 32k. But
vocabulary size also drives the size of the token embedding table (`vocab_size × hidden_dim`),
and the pretraining model this tokenizer feeds (Phase 2, ~25M parameters, a modest hidden
dimension) would have had a _tied_ embedding table eating roughly half of its entire parameter
budget at 32k vocab — well above the ~30-35% typical of reference small language models at
comparable width (checked against several small open decoder-only LMs). Since the fertility gain
from 16k→32k is small in absolute terms (Hindi +0.050, Nepali +0.094 tokens/word — a real but
modest cost, corresponding to roughly 4-7% fewer words fitting in a fixed-length context window),
16,000 was chosen as the better trade-off once the downstream model's parameter budget was
weighed alongside tokenizer quality alone. This is why the tokenizer choice for this project
isn't fertility/UNK-optimal in isolation — it's optimal for the actual model it will train,
which is the more relevant standard. Full sweep data: `report/{hindi,nepali}_vocab_sweep.json`.

### 6.2 The memory story: why the training sample is far smaller than planned

The original plan was to sample 5,000,000 sentences from each language's `train.txt` (standard
practice — SentencePiece's own docs note that a large shuffled subsample trains a vocabulary
just as well as reading the full multi-million-line corpus, at a fraction of the cost). In
practice, the SentencePiece unigram trainer's EM/seed-vocabulary phase (immediately after
`"Done! preprocessed N sentences..."` in its logs) needs dramatically more memory than the raw
sampled text, and scales worse than linearly with sentence count:

| Attempt | Environment               |         RAM budget | Sentences | Result                                                                                                                                                                      |
| ------- | ------------------------- | -----------------: | --------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       | Local                     |        7.6GB total | 5,000,000 | OS OOM-killer killed the process outright (~4.9GB RSS and still climbing); destabilized the whole desktop enough that `gnome-shell` itself was also invoking the OOM killer |
| 2       | Local, `ulimit -v` capped | 3GB (explicit cap) | 1,000,000 | Clean `std::bad_alloc` right after preprocessing 292MB of text, before the EM phase could proceed                                                                           |
| 3       | Local, `ulimit -v` capped | 3GB (explicit cap) |   150,000 | **Succeeded** — trained and evaluated with healthy metrics                                                                                                                  |
| 4       | Colab free tier           |             12.7GB | 2,000,000 | Kernel OOM-crashed at the identical log transition point (right after `"Done! preprocessed 1,992,402 sentences"`)                                                           |
| 5       | Colab free tier           |             12.7GB | 1,000,000 | **Succeeded** — used for the final models reported below                                                                                                                    |

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

### 6.3 Final results

Evaluated on each language's held-out `val.txt` (never seen during training):

| Metric                              |                       Hindi |                      Nepali |
| ----------------------------------- | --------------------------: | --------------------------: |
| Vocab size                          |                      16,000 |                      16,000 |
| Training sample size                |         1,000,000 sentences |         1,000,000 sentences |
| Training sample as % of `train.txt` | ~9.4% (of 10,622,764 lines) | ~6.8% (of 14,691,412 lines) |
| Eval lines sampled (from val.txt)   |                     158,729 |                     200,000 |
| Fertility (tokens/word)             |                      1.2086 |                      1.3468 |
| UNK rate                            |                      0.231% |                      0.195% |
| Avg chars/token                     |                      4.2219 |                      4.8095 |

Both languages show fertility comfortably close to 1 (most words map to one or two subword
pieces, not excessive fragmentation) and UNK rates well under 1%, indicating good coverage of
each language's held-out text at this vocab size and training sample. Nepali's higher fertility
and avg-chars-per-token relative to Hindi is consistent with Nepali's smaller, more repetitive
collected corpus (§4.4) producing a slightly less efficient subword vocabulary at the same 16k
budget. These numbers are, as expected from §6.1's own sweep, modestly worse than the 32,000-vocab
alternative (Hindi 1.1575→1.2086 fertility, Nepali 1.2499→1.3468) — the deliberate trade made in
favor of a healthier parameter budget for the downstream model.

**Top-10 most frequent tokens (val.txt):**

| Rank | Hindi  |   Count | Nepali |   Count |
| ---: | ------ | ------: | ------ | ------: |
|    1 | `▁के`  | 415,235 | `▁।`   | 308,587 |
|    2 | `।`    | 368,583 | `,`    | 222,249 |
|    3 | `▁में` | 309,986 | `को`   | 194,956 |
|    4 | `▁है`  | 274,891 | `।`    | 162,324 |
|    5 | `▁की`  | 258,018 | `▁र`   | 157,202 |
|    6 | `,`    | 220,688 | `▁छ`   | 154,844 |
|    7 | `▁को`  | 201,873 | `मा`   | 148,516 |
|    8 | `▁से`  | 185,682 | `का`   | 108,399 |
|    9 | `.`    | 150,889 | `ले`   | 103,828 |
|   10 | `▁ने`  | 145,133 | `▁पनि` |  68,612 |

(`▁` marks a token that starts a new word, i.e. is preceded by whitespace in the source text —
standard SentencePiece convention.)

**Example tokenization (Hindi, val.txt):**

> - 6 महीने का फंड: वेतनभोगी कर्मचारियों के लिए यह सबसे बेहतरीन ऑप्शन माना जाता है...

```
▁- ▁6 ▁महीने ▁का ▁फंड : ▁वेतन भोग ी ▁कर्मचारियों ▁के ▁लिए ▁यह ▁सबसे ▁बेहतरीन
▁ऑप्शन ▁माना ▁जाता ▁है .
```

**Example tokenization (Nepali, val.txt):**

> काठमाडौं : इरानसँग बढ्दो तनावका बीच अमेरिकी राष्ट्रपति डोनाल्ड ट्रम्पले स्ट्रेट अफ हर्मुजलाई...

```
▁काठमाडौं ▁: ▁इरान सँग ▁बढ्दो ▁तनाव का ▁बीच ▁अमेरिकी ▁राष्ट्रपति ▁डोनाल्ड ▁ट्रम्पले
▁स्ट्र ेट ▁अफ ▁हर् मु ज लाई
```

Two things worth naming here, both direct consequences of dropping from 32k to 16k vocab (§6.1):
`तनावका` still splits cleanly into `▁तनाव` + `का` (a common possessive/genitive suffix,
morpheme-aware behavior preserved), but the proper noun `हर्मुज` ("Hormuz") — infrequent enough
to lose its dedicated token at 16k — fragments into three pieces (`▁हर्`, `मु`, `ज`), and
`वेतनभोगी` similarly splits into `▁वेतन`+`भोग`+`ी` rather than the single token it got at 32k
(compare the same example at 32k, still visible in git history at `report/hindi_tokenizer_stats.json`
prior to this revision). This is exactly the fertility cost quantified in §6.1 — rarer, longer
words pay for the smaller embedding table with more fragments per word.

Output files: `hindi/tokenizer/hindi_unigram_16k.{model,vocab}`,
`nepali/tokenizer/nepali_unigram_16k.{model,vocab}`,
`report/{hindi,nepali}_tokenizer_stats.json` (full stats: top-20 tokens, example
tokenizations).

### 6.4 Real token count, now that the tokenizer actually exists

§5.7's ~510M-token figures were an _estimate_ — the split-stage cap (§5.6) worked in characters,
using assumed chars/token ratios (4.0 Hindi, 4.5 Nepali) because no trained tokenizer existed yet
at that point in the pipeline. Now that the real 16k tokenizer's measured `avg_chars_per_token`
is available (§6.3), the actual token count can be computed directly from the same character
totals:

| Language | Total chars (train+val+test) | Measured avg chars/token (16k) | **Actual total tokens** | §5.7 estimate | Difference |
| -------- | ---------------------------: | -----------------------------: | ----------------------: | ------------: | ---------: |
| Hindi    |                2,040,000,247 |                         4.2219 |        **~483,194,829** |  ~510,000,062 |      −5.3% |
| Nepali   |                2,295,000,003 |                         4.8095 |        **~477,180,581** |  ~510,000,001 |      −6.4% |

Both languages land a little under the assignment's ~500M-token target once measured against the
real tokenizer, rather than the planning-stage assumption — a direct, traceable consequence of
switching to 16k vocab (§6.1): the real 16k tokenizer produces _more_ tokens per character (lower
chars/token) than the 4.0/4.5 figures assumed when the char-based cap was computed, since a
smaller vocabulary means shorter, more frequent subword pieces. This is close enough to the ~500M
target that no corrective action is planned, but it's the accurate number, not the §5.7 estimate,
and should be used wherever an exact token count matters (e.g. Phase 2 training-step budgeting).
It's still an estimate itself, not an exact count — `avg_chars_per_token` was measured on a
`val.txt` sample, not by re-tokenizing every character of `train.txt`/`test.txt` directly.

## 7. Combined dataset statistics and the ≥20%-manual requirement

Using the raw (pre-preprocessing) token counts after the §3.3 top-up:

| Language | Manual tokens | Public tokens |       Total |   Manual % |
| -------- | ------------: | ------------: | ----------: | ---------: |
| Hindi    |   130,000,083 |   520,094,850 | 650,094,933 | **20.00%** |
| Nepali   |   122,661,776 |   520,069,582 | 642,731,358 | **19.08%** |

At face value, Hindi sits right at the 20% line and Nepali reads slightly under it. But this table
is measuring the wrong thing: the assignment's requirement is about the **final delivered corpus**
(capped to ~510M tokens/language, §5.6), not about the raw collected pool before that cap is
applied. The 520M-per-language public **target** was deliberately over-provisioned as a safety
buffer against preprocessing/dedup loss (§3.3) — it was sized to comfortably clear 510M net of
losses, not sized to the minimum needed. That buffer is what's diluting the ratio above, not a
shortfall in manual collection.

**The manual side is fixed and did not need to shrink.** My manual sources were exhausted (§4.4)
at ~130.0M (Hindi) / ~122.7M (Nepali) tokens regardless of how much public data I chose to
download. Had I sized the public buffer 50M tokens smaller per language instead:

| Language | Manual tokens | Public tokens (−50M) |       Total |   Manual % |
| -------- | ------------: | -------------------: | ----------: | ---------: |
| Hindi    |   130,000,083 |          470,094,850 | 600,094,933 | **21.66%** |
| Nepali   |   122,661,776 |          470,069,582 | 592,731,358 | **20.70%** |

Both clear 20% comfortably — the raw-ratio dip is an artifact of how generous a preprocessing-loss
buffer I chose to add on the public side, not evidence that manual collection was insufficient.

![Manual-token share against the 20% requirement: actual (520M public target) versus a counterfactual with 50M less public data per language. Both scenarios clear 20% for Hindi; Nepali only clears it in the counterfactual, but see the surplus-capacity argument below for why the actual run still plausibly would have too.](figures/04_manual_ratio_scenarios.png)

**And there's real evidence the smaller pool would still have reached the 510M cap.** The final
token-cap stage (§5.6) discarded documents that survived preprocessing but exceeded the 510M
target — i.e. the _actual_ over-provisioned run already had more usable data than it needed:

| Language | Docs surviving dedup (stage 3) | Docs actually used (capped) | Surplus discarded by the cap |
| -------- | -----------------------------: | --------------------------: | ---------------------------: |
| Hindi    |                      4,901,226 |                   4,822,928 |                78,298 (1.6%) |
| Nepali   |                      7,476,841 |                   6,723,359 |              753,482 (10.1%) |

Nepali's surplus (10.1%) comfortably absorbs a 9.6%-smaller public download (the −50M scenario
above), so that counterfactual would very plausibly still have reached the 510M cap while clearing
20% manual on raw counts. Hindi's surplus is much thinner (1.6%), so the same −50M cut there risks
landing a bit short of 510M — but Hindi's raw ratio was already at 20.00% with the _actual_
over-provisioned run, meeting "≥20%" as literally stated without needing any counterfactual at
all.

**Caveat, unchanged from before:** manual/public provenance isn't tracked through the pipeline
(stage 1 discards the source tag, §5.7), so the exact post-processing ratio in the delivered,
capped corpus still isn't directly measurable from `stats.json`. The argument above is about
_design intent and demonstrated surplus capacity_ — manual collection was sized to clear 20% net
of a reasonably-sized public buffer, and the actual buffer chosen was larger than necessary — not
a direct measurement of the final corpus's exact ratio.

