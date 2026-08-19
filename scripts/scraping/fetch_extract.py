"""Fetch an article URL and pull its main body text out with trafilatura."""

import hashlib
import sys
import time
from datetime import datetime

import httpx
import trafilatura

USER_AGENT = (
    "Mozilla/5.0 (compatible; LMA-academic-project-scraper/1.0; "
    "+educational monolingual-corpus collection for a university NLP course project)"
)

MIN_CHARS = 200  # drop near-empty extractions (nav pages, stubs, paywalled snippets)
MAX_RETRIES = 2  # for transient network errors only, not HTTP 4xx/5xx


def fetch_article_text(
    client: httpx.Client,
    url: str,
    source_name: str | None = None,
    cutoff: datetime | None = None,
) -> str | None:
    """Never raises — any failure (network, HTTP status, parse) yields None so a single
    bad URL can't take down an overnight batch run."""
    resp = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.get(url, follow_redirects=True, timeout=20)
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError:
            return None  # real HTTP error (404/403/...): retrying won't help
        except (httpx.TransportError, httpx.TimeoutException):
            if attempt == MAX_RETRIES:
                print(f"  [warn] giving up on {url}: repeated network errors", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            print(f"  [warn] unexpected error fetching {url}: {e!r}", file=sys.stderr)
            return None

    try:
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as e:
        print(f"  [warn] extraction failed for {url}: {e!r}", file=sys.stderr)
        return None
    if not text or len(text) < MIN_CHARS:
        return None
    return text


def content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
