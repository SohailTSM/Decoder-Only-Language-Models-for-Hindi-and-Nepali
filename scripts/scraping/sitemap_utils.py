"""Sitemap discovery: recursively expand a sitemap index into dated article URLs."""

import gzip
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Iterable

import httpx
from lxml import etree

_TAG = lambda el: etree.QName(el).localname  # strip XML namespace from a tag name
_MAX_RETRIES = 2


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw[:25].replace("Z", "+0000"), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _fetch_xml(client: httpx.Client, url: str) -> etree._Element | None:
    """Never raises — any failure (network, HTTP status, parse) yields None so one bad
    sitemap URL can't take down an overnight discovery run."""
    resp = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.get(url, follow_redirects=True, timeout=20)
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError:
            return None
        except (httpx.TransportError, httpx.TimeoutException):
            if attempt == _MAX_RETRIES:
                print(f"  [warn] giving up on sitemap {url}: repeated network errors",
                      file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            print(f"  [warn] unexpected error fetching sitemap {url}: {e!r}", file=sys.stderr)
            return None

    body = resp.content
    if body[:2] == b"\x1f\x8b":  # gzip magic
        try:
            body = gzip.decompress(body)
        except OSError:
            return None
    try:
        return etree.fromstring(body, parser=etree.XMLParser(recover=True, huge_tree=True))
    except Exception as e:
        print(f"  [warn] failed to parse sitemap {url}: {e!r}", file=sys.stderr)
        return None


_PAGE_PARAM_RE = re.compile(r"[?&]p=(\d+)")
_URL_YEAR_RE = re.compile(r"[^/]+/(\d{4})/")  # domain.com/YYYY/... (WordPress-style permalinks)


def _parse_sitemap_node(root: etree._Element, cutoff: datetime, max_page: int | None = None):
    """Returns (kind, items) where kind is 'index' or 'urlset'. For 'index', items is a
    list of (loc, lastmod) child sitemap URLs (already cutoff-filtered on their own
    lastmod, when present). For 'urlset', items is a list of (article_url, lastmod)
    filtered to cutoff."""
    root_tag = _TAG(root)
    if root_tag == "sitemapindex":
        out = []
        for sm in root:
            loc, lastmod = None, None
            for child in sm:
                if _TAG(child) == "loc":
                    loc = (child.text or "").strip()
                elif _TAG(child) == "lastmod":
                    lastmod = _parse_date(child.text)
            if not loc:
                continue
            # Skip whole sub-sitemaps that are clearly stale (dated sub-sitemap, e.g.
            # monthly archives) to avoid needless fetching.
            if lastmod and lastmod < cutoff:
                continue
            if max_page is not None:
                pm = _PAGE_PARAM_RE.search(loc)
                if pm and int(pm.group(1)) > max_page:
                    continue
            out.append((loc, lastmod))
        return "index", out
    elif root_tag == "urlset":
        ns = {"n": "http://www.google.com/schemas/sitemap-news/0.9"}
        out = []
        for u in root:
            loc, lastmod = None, None
            for child in u:
                tag = _TAG(child)
                if tag == "loc":
                    loc = (child.text or "").strip()
                elif tag == "lastmod":
                    lastmod = _parse_date(child.text)
                elif tag == "news":
                    pub = child.find("n:publication_date", namespaces=ns)
                    if pub is not None and pub.text:
                        lastmod = _parse_date(pub.text) or lastmod
            if not loc:
                continue
            if lastmod is not None and lastmod < cutoff:
                continue
            # Extra safety net: WordPress-style permalinks embed the real publish year in
            # the URL path (e.g. /2020/12/04/...), set once at publish time and never
            # touched again — unlike some sites' <lastmod>, which can reflect a later
            # regeneration/edit rather than the true original date (verified: nepalpress's
            # lastmod showed 2026 for an article whose URL and content confirm it's from
            # 2020). Reject on a URL year that's clearly pre-cutoff, regardless of lastmod.
            ym = _URL_YEAR_RE.match(loc.removeprefix("https://").removeprefix("http://"))
            if ym and int(ym.group(1)) < cutoff.year:
                continue
            out.append((loc, lastmod))
        return "urlset", out
    return None, []


def discover_urls(
    client: httpx.Client,
    sitemap_url: str,
    cutoff: datetime,
    max_workers: int = 12,
    max_page: int | None = None,
) -> Iterable[tuple[str, datetime | None]]:
    """Breadth-first walk of a sitemap/sitemap-index, yielding (article_url, published_date)
    for entries dated on/after `cutoff`. Entries with no discoverable date are yielded with
    date=None so the caller can decide.

    Sites like annapurnapost.com paginate their news sitemap across 600+ child files with
    no per-file date info visible from the index alone, so each level is fetched
    concurrently (httpx.Client is thread-safe) rather than one sub-sitemap at a time —
    otherwise discovery alone can take minutes before a single article fetch even starts.

    max_page: for paginated indexes using a `?p=N` query param, skip child sitemaps beyond
    this page number. Useful when a site's own lastmod metadata for deep pages can't be
    trusted, to bound how far back into old content discovery goes."""
    seen = {sitemap_url}
    frontier = [sitemap_url]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while frontier:
            futures = {pool.submit(_fetch_xml, client, u): u for u in frontier}
            frontier = []
            for fut in as_completed(futures):
                root = fut.result()
                if root is None:
                    continue
                kind, items = _parse_sitemap_node(root, cutoff, max_page)
                if kind == "index":
                    for loc, _lastmod in items:
                        if loc not in seen:
                            seen.add(loc)
                            frontier.append(loc)
                elif kind == "urlset":
                    yield from items
