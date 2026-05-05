import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

app = Flask(__name__, static_folder="static")
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-3-mini")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "Sec-CH-UA": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Cache-Control": "max-age=0",
}

# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_page(url: str) -> dict:
    import time
    from urllib.parse import urlparse

    session = requests.Session()
    session.headers.update(HEADERS)

    # Warm up the session with a HEAD request to establish cookies
    try:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        session.head(root, timeout=8, allow_redirects=True)
        time.sleep(1)
    except Exception:
        pass

    for attempt in range(3):
        resp = session.get(url, timeout=15, allow_redirects=True)
        if resp.status_code == 429:
            if attempt < 2:
                time.sleep(6 * (attempt + 1))
                continue
            raise requests.exceptions.RequestException(
                "This site is blocking automated access. "
                "It may use JavaScript-based bot protection that requires a real browser."
            )
        resp.raise_for_status()
        break
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.string.strip() if soup.title else ""
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

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    body_text = soup.get_text(separator="\n", strip=True)
    lines = [ln for ln in body_text.splitlines() if len(ln.strip()) > 30]
    body_text = "\n".join(lines[:200])

    has_schema = bool(soup.find("script", {"type": "application/ld+json"}))
    schema_types = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
            t = data.get("@type") or (data.get("@graph", [{}])[0].get("@type", ""))
            if t:
                schema_types.append(t if isinstance(t, str) else ", ".join(t))
        except Exception:
            pass

    faq_patterns = re.compile(r"\b(faq|frequently asked|q&a|questions)\b", re.I)
    has_faq = bool(faq_patterns.search(resp.text))

    return {
        "url": url,
        "title": title,
        "meta_description": meta_desc,
        "headings": headings[:20],
        "body_text": body_text,
        "has_schema_markup": has_schema,
        "schema_types": schema_types,
        "has_faq_section": has_faq,
        "word_count": len(body_text.split()),
    }


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

AUDIT_PROMPT = """You are an expert in Generative Engine Optimization (GEO) — the practice of optimizing websites so that AI search engines (ChatGPT, Google AI Overviews, Perplexity, Claude) cite and surface them in answers.

Analyze the following scraped website data and produce a structured GEO audit.

## Website Data
URL: {url}
Title: {title}
Meta Description: {meta_description}
Has Schema Markup: {has_schema_markup}
Schema Types Found: {schema_types}
Has FAQ Section: {has_faq_section}
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

1. **conversational_clarity** — Does the content directly answer natural-language questions an AI would be asked? Look for: question-and-answer patterns, clear definitions, use of "what/how/why/when" sentence structures.

2. **entity_density** — How rich is the page in named entities that ground it geographically and topically? Look for: business name, address, phone, hours, service names, staff names, prices, neighborhoods served.

3. **direct_answer_formatting** — Is the content structured so an AI can extract a crisp answer to paste into a response? Look for: numbered lists, bullet points, short declarative paragraphs, FAQ sections, "X is Y" sentence structures.

4. **citation_worthiness** — Does the page contain specific, quotable, authoritative information an AI would want to cite? Look for: statistics, named credentials/awards, specific prices, before/after results, unique methodology names.

5. **schema_and_structure** — Does the page have technical signals that help AI crawlers understand the business? Look for: LocalBusiness schema, FAQPage schema, Review schema, BreadcrumbList, proper H1/H2 hierarchy.

Return ONLY valid JSON in this exact shape (no markdown fences, no extra text):
{{
  "overall_score": <integer>,
  "overall_grade": "<letter>",
  "business_type_detected": "<string>",
  "one_line_verdict": "<string — one punchy sentence describing the biggest GEO problem>",
  "pillars": {{
    "conversational_clarity": {{
      "score": <int>,
      "grade": "<letter>",
      "summary": "<string>",
      "top_issues": ["<string>", "<string>"],
      "quick_wins": ["<string>", "<string>"]
    }},
    "entity_density": {{
      "score": <int>,
      "grade": "<letter>",
      "summary": "<string>",
      "top_issues": ["<string>", "<string>"],
      "quick_wins": ["<string>", "<string>"]
    }},
    "direct_answer_formatting": {{
      "score": <int>,
      "grade": "<letter>",
      "summary": "<string>",
      "top_issues": ["<string>", "<string>"],
      "quick_wins": ["<string>", "<string>"]
    }},
    "citation_worthiness": {{
      "score": <int>,
      "grade": "<letter>",
      "summary": "<string>",
      "top_issues": ["<string>", "<string>"],
      "quick_wins": ["<string>", "<string>"]
    }},
    "schema_and_structure": {{
      "score": <int>,
      "grade": "<letter>",
      "summary": "<string>",
      "top_issues": ["<string>", "<string>"],
      "quick_wins": ["<string>", "<string>"]
    }}
  }},
  "priority_action": "<The single highest-impact thing this business should do this week>"
}}"""


def analyze_with_llm(scraped: dict) -> dict:
    import time

    headings_text = "\n".join(
        f"  {h['level']}: {h['text']}" for h in scraped["headings"]
    )
    prompt = AUDIT_PROMPT.format(
        url=scraped["url"],
        title=scraped["title"],
        meta_description=scraped["meta_description"],
        has_schema_markup=scraped["has_schema_markup"],
        schema_types=", ".join(scraped["schema_types"]) or "None detected",
        has_faq_section=scraped["has_faq_section"],
        word_count=scraped["word_count"],
        headings=headings_text or "(none found)",
        body_text=scraped["body_text"][:2000],
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a GEO audit assistant. You ALWAYS respond with valid JSON only — "
                    "no explanations, no refusals, no markdown. If the page has little or no content, "
                    "still return the full JSON structure with scores of 0 and notes reflecting the lack of content."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.2,
    }

    last_json_error = None
    for attempt in range(3):
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        if resp.status_code == 429:
            if attempt < 2:
                time.sleep(10)
                continue
            raise ValueError("Rate limit reached. Please wait a moment and try again.")

        if resp.status_code == 401:
            raise ValueError("Invalid Groq API key — check it at console.groq.com.")

        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        if not raw:
            last_json_error = json.JSONDecodeError("LLM returned an empty response", "", 0)
            time.sleep(3)
            continue

        # Try direct parse first, then fall back to extracting the JSON object
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Find the outermost { ... } block in case the model added surrounding text
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                last_json_error = e
        else:
            last_json_error = json.JSONDecodeError("No JSON object found in LLM response", raw, 0)

        time.sleep(3)
        continue

    raise last_json_error


def analyze_with_xai(scraped: dict) -> dict:
    """Fallback to xAI Grok when Groq is unavailable. Uses OpenAI-compatible API."""
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY is not set.")

    import time

    headings_text = "\n".join(
        f"  {h['level']}: {h['text']}" for h in scraped["headings"]
    )
    prompt = AUDIT_PROMPT.format(
        url=scraped["url"],
        title=scraped["title"],
        meta_description=scraped["meta_description"],
        has_schema_markup=scraped["has_schema_markup"],
        schema_types=", ".join(scraped["schema_types"]) or "None detected",
        has_faq_section=scraped["has_faq_section"],
        word_count=scraped["word_count"],
        headings=headings_text or "(none found)",
        body_text=scraped["body_text"][:2000],
    )

    payload = {
        "model": XAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a GEO audit assistant. You ALWAYS respond with valid JSON only — "
                    "no explanations, no refusals, no markdown. If the page has little or no content, "
                    "still return the full JSON structure with scores of 0 and notes reflecting the lack of content."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.2,
    }

    last_json_error = None
    for attempt in range(3):
        resp = requests.post(
            XAI_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        if resp.status_code == 429:
            if attempt < 2:
                time.sleep(8)
                continue
            raise ValueError("xAI rate limit reached.")

        if resp.status_code in (401, 403):
            raise ValueError("Invalid xAI API key — check it at console.x.ai.")

        if resp.status_code == 400:
            # Surface the actual xAI error so we can see what's wrong with the request
            try:
                err_detail = resp.json().get("error", resp.text[:200])
            except Exception:
                err_detail = resp.text[:200]
            raise ValueError(f"xAI rejected request (400): {err_detail}")

        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        if not raw:
            last_json_error = json.JSONDecodeError("xAI returned an empty response", "", 0)
            time.sleep(3)
            continue

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                last_json_error = e
        else:
            last_json_error = json.JSONDecodeError("No JSON object found in xAI response", raw, 0)

        time.sleep(3)

    raise last_json_error


def analyze_with_gemini(scraped: dict) -> dict:
    """Fallback to Google Gemini when Groq is unavailable or rate-limited."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set — add it in Render environment variables.")

    import time

    headings_text = "\n".join(
        f"  {h['level']}: {h['text']}" for h in scraped["headings"]
    )
    prompt = AUDIT_PROMPT.format(
        url=scraped["url"],
        title=scraped["title"],
        meta_description=scraped["meta_description"],
        has_schema_markup=scraped["has_schema_markup"],
        schema_types=", ".join(scraped["schema_types"]) or "None detected",
        has_faq_section=scraped["has_faq_section"],
        word_count=scraped["word_count"],
        headings=headings_text or "(none found)",
        body_text=scraped["body_text"][:2000],
    )

    system_instruction = (
        "You are a GEO audit assistant. You ALWAYS respond with valid JSON only — "
        "no explanations, no refusals, no markdown. If the page has little or no content, "
        "still return the full JSON structure with scores of 0 and notes reflecting the lack of content."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json",
        },
    }

    last_json_error = None
    for attempt in range(3):
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )

        if resp.status_code == 429:
            if attempt < 2:
                time.sleep(8)
                continue
            raise ValueError("Gemini rate limit reached.")

        if resp.status_code in (401, 403):
            raise ValueError("Invalid Gemini API key — check it at aistudio.google.com.")

        resp.raise_for_status()

        data = resp.json()
        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            last_json_error = json.JSONDecodeError("Gemini returned no content", "", 0)
            time.sleep(3)
            continue

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        if not raw:
            last_json_error = json.JSONDecodeError("Gemini returned an empty response", "", 0)
            time.sleep(3)
            continue

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                last_json_error = e
        else:
            last_json_error = json.JSONDecodeError("No JSON object found in Gemini response", raw, 0)

        time.sleep(3)

    raise last_json_error


def run_audit_with_fallbacks(scraped: dict):
    """Try LLM providers in order: Groq → Gemini → synthetic.
    Returns either an audit dict, or a (jsonify_response, status_code) tuple on hard failure."""
    providers = [
        ("Groq", GROQ_API_KEY, analyze_with_llm),
        ("Gemini", GEMINI_API_KEY, analyze_with_gemini),
    ]

    errors = []
    for name, key, fn in providers:
        if not key:
            errors.append(f"{name}: not configured")
            continue
        try:
            return fn(scraped)
        except json.JSONDecodeError:
            return synthetic_audit_for_empty_page(scraped)
        except ValueError as e:
            errors.append(f"{name}: {e}")
            continue
        except requests.exceptions.RequestException as e:
            errors.append(f"{name}: {e}")
            continue

    return jsonify({"error": "All LLM providers failed → " + " | ".join(errors)}), 503


def synthetic_audit_for_empty_page(scraped: dict) -> dict:
    """Return a deterministic GEO audit for pages with little or no scrapable content
    (typically JavaScript-rendered SPAs like Shopify, Wix, Squarespace storefronts)."""
    pillar = lambda summary, issues, wins: {
        "score": 0,
        "grade": "F",
        "summary": summary,
        "top_issues": issues,
        "quick_wins": wins,
    }
    return {
        "overall_score": 5,
        "overall_grade": "F",
        "business_type_detected": "JS-rendered site",
        "one_line_verdict": "AI engines see almost nothing — your site relies on JavaScript to display its content, which most AI crawlers cannot read.",
        "pillars": {
            "conversational_clarity": pillar(
                "AI crawlers see no readable content because the page renders client-side.",
                ["Page returns minimal HTML before JavaScript runs", "No question/answer text visible to crawlers"],
                ["Add server-rendered text content", "Use SSR or pre-rendering for crawlers"],
            ),
            "entity_density": pillar(
                "No business entities (name, address, services) are visible in the raw HTML.",
                ["Business name and details not in static HTML", "No structured contact information"],
                ["Add a server-rendered footer with NAP details", "Include business info in the static page source"],
            ),
            "direct_answer_formatting": pillar(
                "No structured content (lists, FAQs, headings) detected in the static HTML.",
                ["No FAQ section in raw HTML", "No bullet points or direct-answer formatting"],
                ["Add a server-rendered FAQ block", "Include key product/service descriptions in static HTML"],
            ),
            "citation_worthiness": pillar(
                "No quotable, authoritative content is visible to AI crawlers.",
                ["No statistics, credentials, or unique value props in HTML", "No reviews or testimonials in static markup"],
                ["Render testimonials server-side", "Add credentials and awards to the static page"],
            ),
            "schema_and_structure": pillar(
                "Schema and heading structure cannot be assessed — page content loads via JavaScript.",
                ["No detectable schema markup in static HTML", "Heading hierarchy not rendered server-side"],
                ["Inject LocalBusiness schema in the document head", "Pre-render H1/H2 hierarchy"],
            ),
        },
        "priority_action": (
            "Enable server-side rendering or static pre-rendering so AI crawlers can read your content. "
            "This is the single biggest GEO blocker for this site."
        ),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/audit", methods=["POST"])
def audit():
    if not (GROQ_API_KEY or GEMINI_API_KEY):
        return jsonify({"error": "No LLM API key configured. Set GROQ_API_KEY or GEMINI_API_KEY in Render."}), 503

    body = request.get_json(force=True)
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        scraped = scrape_page(url)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not fetch the page: {e}"}), 422

    is_near_empty = (
        scraped["word_count"] < 10
        and len(scraped["headings"]) <= 2
        and not scraped["meta_description"]
    )

    if is_near_empty:
        analysis = synthetic_audit_for_empty_page(scraped)
    else:
        analysis = run_audit_with_fallbacks(scraped)
        if isinstance(analysis, tuple):  # error response
            return analysis

    return jsonify({
        "meta": {
            "url": scraped["url"],
            "title": scraped["title"],
            "word_count": scraped["word_count"],
            "has_schema": scraped["has_schema_markup"],
            "schema_types": scraped["schema_types"],
            "has_faq": scraped["has_faq_section"],
        },
        "audit": analysis,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
