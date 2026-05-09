import os
import json
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import apify_helper

app = Flask(__name__, static_folder="static")
CORS(app)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
XAI_API_KEY       = os.environ.get("XAI_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
XAI_MODEL       = os.environ.get("XAI_MODEL",       "grok-4-fast-non-reasoning")
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL",     "gemini-2.5-flash")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_SYSTEM_PROMPT = (
    "You are a GEO audit assistant. You ALWAYS respond with valid JSON only — "
    "no explanations, no refusals, no markdown."
)

_AUDIT_PROMPT = """You are an expert in Generative Engine Optimization (GEO) — the practice of optimizing websites so that AI search engines (ChatGPT, Google AI Overviews, Perplexity, Claude) cite and surface them in answers.

Analyze the following scraped website data and produce a structured GEO audit.

## Website Data
URL: {url}
Title: {title}
Meta Description: {meta_description}
Has Schema Markup: {has_schema}
Schema Types Found: {schema_types}
Has FAQ Section: {has_faq}
Word Count: {word_count}

### Headings
{headings}

### Page Content (excerpt)
{body_text}

## Your Task
Score the page on each of the 5 GEO pillars below (0–100). Be precise and critical — very few pages will score above 75.

For each pillar provide:
- score (integer 0–100)
- grade (A=90+, B=75-89, C=60-74, D=45-59, F=below 45)
- summary (1 concise sentence)
- top_issues (list of 2–3 specific problems found on THIS page)
- quick_wins (list of 2–3 concrete, actionable fixes)

### The 5 GEO Pillars

1. **conversational_clarity** — Does the content directly answer natural-language questions an AI would be asked?
2. **entity_density** — How rich is the page in named entities that ground it geographically and topically?
3. **direct_answer_formatting** — Is the content structured so an AI can extract a crisp answer?
4. **citation_worthiness** — Does the page contain specific, quotable, authoritative information an AI would want to cite?
5. **schema_and_structure** — Does the page have technical signals that help AI crawlers understand the business?

Return ONLY valid JSON in this exact shape (no markdown fences, no extra text):
{{
  "overall_score": <integer>,
  "overall_grade": "<letter>",
  "business_type_detected": "<string>",
  "one_line_verdict": "<string>",
  "pillars": {{
    "conversational_clarity":  {{"score": <int>, "grade": "<letter>", "summary": "<string>", "top_issues": ["<string>", "<string>"], "quick_wins": ["<string>", "<string>"]}},
    "entity_density":          {{"score": <int>, "grade": "<letter>", "summary": "<string>", "top_issues": ["<string>", "<string>"], "quick_wins": ["<string>", "<string>"]}},
    "direct_answer_formatting":{{"score": <int>, "grade": "<letter>", "summary": "<string>", "top_issues": ["<string>", "<string>"], "quick_wins": ["<string>", "<string>"]}},
    "citation_worthiness":     {{"score": <int>, "grade": "<letter>", "summary": "<string>", "top_issues": ["<string>", "<string>"], "quick_wins": ["<string>", "<string>"]}},
    "schema_and_structure":    {{"score": <int>, "grade": "<letter>", "summary": "<string>", "top_issues": ["<string>", "<string>"], "quick_wins": ["<string>", "<string>"]}}}},
  "priority_action": "<The single highest-impact thing this business should do this week>"
}}"""


# ---------------------------------------------------------------------------
# Scraping — ported directly from combined_seo_geo_audit.py in the Dashboard
# ---------------------------------------------------------------------------

def _scrape(url: str) -> dict:
    # Apify path: handles JS-rendered sites (React, Webflow, Squarespace) where
    # plain requests.get returns half-empty HTML and audits score artificially low.
    apify_result = apify_helper.crawl_website(url)
    if apify_result is not None:
        return apify_result

    resp = requests.get(url, headers=_HEADERS, timeout=20, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    meta_desc = ""
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

    has_faq = bool(re.search(r"\b(faq|frequently asked|q&a)\b", resp.text, re.I))

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    raw_body  = soup.get_text(separator="\n", strip=True)
    word_count = len(raw_body.split())
    lines     = [ln for ln in raw_body.splitlines() if len(ln.strip()) > 30]
    body_text = "\n".join(lines[:200])

    return {
        "url":             url,
        "title":           title,
        "meta_description": meta_desc,
        "headings":        headings[:25],
        "body_text":       body_text,
        "word_count":      word_count,
        "has_schema":      has_schema,
        "schema_types":    schema_types,
        "has_faq":         has_faq,
    }


# ---------------------------------------------------------------------------
# JSON parsing — ported from combined_seo_geo_audit.py
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    relaxed = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return json.loads(relaxed)


# ---------------------------------------------------------------------------
# Provider callers — ported from combined_seo_geo_audit.py
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str) -> str:
    payload = {
        "model":    ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "system":   _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    for _ in range(3):
        resp = requests.post("https://api.anthropic.com/v1/messages",
                             headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            time.sleep(8)
            continue
        if resp.status_code in (401, 403):
            raise ValueError("Invalid Anthropic API key.")
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    raise RuntimeError("Anthropic rate-limited after 3 attempts")


def _call_xai(prompt: str) -> str:
    payload = {
        "model":   XAI_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    for _ in range(3):
        resp = requests.post("https://api.x.ai/v1/chat/completions",
                             headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            time.sleep(8)
            continue
        if resp.status_code in (401, 403):
            raise ValueError("Invalid xAI API key.")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError("xAI rate-limited after 3 attempts")


def _call_gemini(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }
    for _ in range(3):
        resp = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=60)
        if resp.status_code == 429:
            time.sleep(8)
            continue
        if resp.status_code in (401, 403):
            raise ValueError("Invalid Gemini API key.")
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("Gemini rate-limited after 3 attempts")


# ---------------------------------------------------------------------------
# Audit orchestration — Anthropic → XAI → Gemini
# ---------------------------------------------------------------------------

def _build_prompt(s: dict) -> str:
    headings_text = "\n".join(f"  {h['level']}: {h['text']}" for h in s["headings"]) or "(none)"
    return _AUDIT_PROMPT.format(
        url=s["url"],
        title=s["title"],
        meta_description=s["meta_description"],
        has_schema=s["has_schema"],
        schema_types=", ".join(s["schema_types"]) or "None detected",
        has_faq=s["has_faq"],
        word_count=s["word_count"],
        headings=headings_text,
        body_text=s["body_text"][:2000],
    )


def _run_audit(s: dict) -> dict:
    prompt = _build_prompt(s)
    providers = [
        ("Anthropic", ANTHROPIC_API_KEY, _call_anthropic),
        ("XAI",       XAI_API_KEY,       _call_xai),
        ("Gemini",    GEMINI_API_KEY,     _call_gemini),
    ]
    errors = []
    for name, key, caller in providers:
        if not key:
            errors.append(f"{name}: not configured")
            continue
        try:
            raw = caller(prompt)
            return _parse_json(raw)
        except (ValueError, RuntimeError) as e:
            errors.append(f"{name}: {e}")
        except requests.exceptions.RequestException as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("All providers failed — " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/audit", methods=["POST"])
def audit():
    if not (ANTHROPIC_API_KEY or XAI_API_KEY or GEMINI_API_KEY):
        return jsonify({"error": "No LLM API key configured."}), 503

    body = request.get_json(force=True)
    url  = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        scraped = _scrape(url)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not fetch the page: {e}"}), 422

    try:
        analysis = _run_audit(scraped)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify({
        "meta": {
            "url":          scraped["url"],
            "title":        scraped["title"],
            "word_count":   scraped["word_count"],
            "has_schema":   scraped["has_schema"],
            "schema_types": scraped["schema_types"],
            "has_faq":      scraped["has_faq"],
        },
        "audit": analysis,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
