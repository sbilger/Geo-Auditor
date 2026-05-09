"""Apify wrapper for the GEO Auditor.

Optional layer: when APIFY_API_KEY is unset (or apify-client isn't installed),
crawl_website() returns None and the caller falls back to its existing
requests.get + BeautifulSoup pipeline.

Returns the same shape as Hogtron-Geo-Auditor's _scrape() so it's a drop-in:
  {url, title, meta_description, headings, body_text, word_count,
   has_schema, schema_types, has_faq}
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from bs4 import BeautifulSoup

_client = None


def _get_client():
    global _client
    if not os.environ.get("APIFY_API_KEY"):
        return None
    if _client is not None:
        return _client
    try:
        from apify_client import ApifyClient
    except ImportError:
        print("[Apify] apify-client not installed; pip install apify-client to enable.")
        return None
    _client = ApifyClient(os.environ["APIFY_API_KEY"])
    return _client


def is_enabled() -> bool:
    return _get_client() is not None


def crawl_website(url: str) -> Optional[dict]:
    """Headless-rendered crawl — returns the same dict shape as _scrape() in app.py.

    Falls through (returns None) on any failure so the caller can use its
    existing requests-based scraper.
    """
    client = _get_client()
    if not client:
        return None
    try:
        run = client.actor("apify/website-content-crawler").call(
            run_input={
                "startUrls": [{"url": url}],
                "maxCrawlPages": 1,
                "crawlerType": "playwright:adaptive",
                "saveHtml": True,
                "saveMarkdown": True,
                "removeCookieWarnings": True,
                "readableTextCharThreshold": 100,
            },
            timeout_secs=240,
        )
        if not run or not run.get("defaultDatasetId"):
            return None
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        if not items:
            return None
        first = items[0]
        html = first.get("html") or ""
        if not html:
            return None
        return _normalize(url, first, html)
    except Exception as exc:
        print(f"[Apify] website-content-crawler failed: {exc}")
        return None


def _normalize(url: str, item: dict, html: str) -> dict:
    """Convert Apify dataset output → the same shape _scrape() returns."""
    soup = BeautifulSoup(html, "html.parser")

    meta = item.get("metadata") or {}
    title = meta.get("title") or (soup.title.string.strip() if soup.title and soup.title.string else "")

    meta_desc = meta.get("description") or ""
    if not meta_desc:
        for tag in soup.find_all("meta"):
            if tag.get("name", "").lower() == "description":
                meta_desc = tag.get("content", "").strip()
                break

    headings = []
    for level in ["h1", "h2", "h3"]:
        for tag in soup.find_all(level):
            text = tag.get_text(strip=True)
            if text:
                headings.append({"level": level.upper(), "text": text})

    schema_tags = soup.find_all("script", {"type": "application/ld+json"})
    has_schema = bool(schema_tags)
    schema_types = []
    for tag in schema_tags:
        try:
            data = json.loads(tag.string or "")
            t = data.get("@type") or (data.get("@graph", [{}])[0].get("@type", ""))
            if t:
                schema_types.append(t if isinstance(t, str) else ", ".join(t))
        except Exception:
            pass

    has_faq = bool(re.search(r"\b(faq|frequently asked|q&a)\b", html, re.I))

    # Prefer the crawler's rendered text; fall back to BS4 extraction.
    rendered = item.get("text") or item.get("markdown") or ""
    if rendered:
        word_count = len(rendered.split())
        lines = [ln for ln in rendered.splitlines() if len(ln.strip()) > 30]
        body_text = "\n".join(lines[:200])
    else:
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        raw_body = soup.get_text(separator="\n", strip=True)
        word_count = len(raw_body.split())
        lines = [ln for ln in raw_body.splitlines() if len(ln.strip()) > 30]
        body_text = "\n".join(lines[:200])

    return {
        "url": url,
        "title": title,
        "meta_description": meta_desc,
        "headings": headings[:25],
        "body_text": body_text,
        "word_count": word_count,
        "has_schema": has_schema,
        "schema_types": schema_types,
        "has_faq": has_faq,
    }
