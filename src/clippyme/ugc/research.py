"""Website scraping and online research module for AI Shorts."""
import json
import os
import re
import httpx
from typing import Dict, Any, Optional
from urllib.parse import urljoin

from clippyme.netutil import resolve_host_addresses


def _validate_safe_url(url: str) -> str:
    """Validate that the URL is public and non-internal."""
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https schemes are permitted.")
    host = parsed.host
    if not host or host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise ValueError("Invalid target host.")
    addrs = resolve_host_addresses(host, timeout=5.0)
    if not addrs:
        raise ValueError("Target host could not be resolved.")
    for a in addrs:
        if (a.is_private or a.is_loopback or a.is_link_local
                or a.is_reserved or a.is_multicast or a.is_unspecified):
            raise ValueError("Target host resolves to a non-public IP address.")
    return str(parsed)


from html.parser import HTMLParser


class _ProductHTMLParser(HTMLParser):
    """Clean stdlib HTML parser extracting title, meta description, headings, and text."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.meta_desc = ""
        self.headings = []
        self.text_chunks = []
        self._in_title = False
        self._in_heading = False
        self._current_heading = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"}

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        if t == "title":
            self._in_title = True
        elif t in ("h1", "h2", "h3"):
            self._in_heading = True
            self._current_heading = []
        elif t == "meta":
            attr_dict = {k.lower(): (v or "") for k, v in attrs}
            if attr_dict.get("name", "").lower() == "description":
                self.meta_desc = attr_dict.get("content", "")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self._skip_tags:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return

        if t == "title":
            self._in_title = False
        elif t in ("h1", "h2", "h3"):
            self._in_heading = False
            heading_str = " ".join(self._current_heading).strip()
            if heading_str and len(heading_str) < 200:
                self.headings.append(heading_str)

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += (" " if self.title else "") + text
        if self._in_heading:
            self._current_heading.append(text)
        self.text_chunks.append(text)


def scrape_product_website(url: str) -> Dict[str, Any]:
    """Scrape product website to extract text, headings, and metadata safely."""
    current_url = _validate_safe_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    max_redirects = 5
    resp = None
    with httpx.Client(timeout=30.0, follow_redirects=False) as client:
        for _ in range(max_redirects):
            resp = client.get(current_url, headers=headers)
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    break
                next_url = str(httpx.URL(current_url).join(location))
                current_url = _validate_safe_url(next_url)
                continue
            resp.raise_for_status()
            break
        else:
            raise ValueError("Too many redirects.")

    if resp is None:
        raise ValueError("Failed to fetch product website.")

    parser = _ProductHTMLParser()
    parser.feed(resp.text)

    title = parser.title.strip()
    meta_desc = parser.meta_desc.strip()
    headings = parser.headings
    body_text = "\n".join(parser.text_chunks)
    body_text = re.sub(r"\n{3,}", "\n\n", body_text)[:8000]

    return {
        "url": current_url,
        "title": title,
        "description": meta_desc,
        "headings": headings,
        "body_text": body_text,
    }


def research_product_online(url_or_description: str, gemini_key: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Research product online using Gemini with Google Search grounding."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_key)
    model = model_name or os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"

    prompt = f"""You are a senior product researcher. Research this product/website and extract market insights, real user pain points, competitor comparisons, and key value propositions:

TARGET: {url_or_description}

OUTPUT JSON:
{{
  "product_name": "...",
  "tagline": "...",
  "target_audience": "...",
  "core_problem": "...",
  "unique_solution": "...",
  "key_features": ["feature 1", "feature 2", ...],
  "customer_reviews_summary": "...",
  "viral_angles": ["Angle 1 (shocking statistic / problem)", "Angle 2 (competitor alternative)", "Angle 3 (insane productivity hack)"]
}}
"""

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text.strip())
