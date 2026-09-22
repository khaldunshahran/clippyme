"""US Trend Sources Integration — pure, host-testable.

Fetches and parses live RSS feeds for US topics:
- Google News US (Politics, Entertainment, World/Breaking, Nation)
- Google Trends US (Real-time query spikes and search traffic)

Zero API keys required, safe HTTP timeouts, bounded reads, and clean XML parsing.
"""
from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("clippyme")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_FEED_BYTES = 2 * 1024 * 1024  # 2 MB cap
DEFAULT_TIMEOUT = 12.0

GOOGLE_NEWS_FEEDS = {
    "politics": "https://news.google.com/rss/headlines/section/topic/POLITICS?ceid=US:en&gl=US&hl=en-US",
    "entertainment": "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?ceid=US:en&gl=US&hl=en-US",
    "world": "https://news.google.com/rss/headlines/section/topic/WORLD?ceid=US:en&gl=US&hl=en-US",
    "nation": "https://news.google.com/rss/headlines/section/topic/NATION?ceid=US:en&gl=US&hl=en-US",
}

GOOGLE_TRENDS_FEED = "https://trends.google.com/trending/rss?geo=US"

_CLEAN_HTML_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    if not text:
        return ""
    cleaned = _CLEAN_HTML_RE.sub(" ", text)
    return " ".join(cleaned.split()).strip()


@dataclass
class TrendSourceItem:
    title: str
    source: str
    category: str
    link: str = ""
    pub_date: str = ""
    approx_traffic: str = ""
    snippet: str = ""


def fetch_url_content(url: str, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """Fetch URL with safety limits and a standard user agent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec B310: safe RSS endpoints
        data = response.read(MAX_FEED_BYTES + 1)
        if len(data) > MAX_FEED_BYTES:
            raise ValueError(f"Feed at {url} exceeded maximum size limit")
        return data


def parse_google_news_rss(xml_bytes: bytes, category: str = "general", max_items: int = 15) -> List[TrendSourceItem]:
    """Parse Google News RSS XML into TrendSourceItems."""
    items: List[TrendSourceItem] = []
    if not xml_bytes:
        return items

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("Failed to parse Google News RSS XML: %s", exc)
        return items

    for item_el in root.findall(".//item"):
        title_el = item_el.find("title")
        raw_title = (title_el.text or "").strip() if title_el is not None else ""
        if not raw_title:
            continue

        # Google News titles usually end with " - Source Name"
        source_name = "News"
        title = raw_title
        if " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            title = parts[0].strip()
            source_name = parts[1].strip()

        link_el = item_el.find("link")
        link = (link_el.text or "").strip() if link_el is not None else ""

        pub_el = item_el.find("pubDate")
        pub_date = (pub_el.text or "").strip() if pub_el is not None else ""

        desc_el = item_el.find("description")
        desc = clean_html(desc_el.text or "") if desc_el is not None else ""

        items.append(
            TrendSourceItem(
                title=title,
                source=source_name,
                category=category,
                link=link,
                pub_date=pub_date,
                snippet=desc[:300] if desc else "",
            )
        )
        if len(items) >= max_items:
            break

    return items


def parse_google_trends_rss(xml_bytes: bytes, max_items: int = 15) -> List[TrendSourceItem]:
    """Parse Google Trends RSS XML into TrendSourceItems with approx traffic."""
    items: List[TrendSourceItem] = []
    if not xml_bytes:
        return items

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("Failed to parse Google Trends RSS XML: %s", exc)
        return items

    ns = {"ht": "https://trends.google.com/trending/rss"}

    for item_el in root.findall(".//item"):
        title_el = item_el.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue

        traffic_el = item_el.find("ht:approx_traffic", ns)
        approx_traffic = (traffic_el.text or "").strip() if traffic_el is not None else ""

        link_el = item_el.find("link")
        link = (link_el.text or "").strip() if link_el is not None else ""

        pub_el = item_el.find("pubDate")
        pub_date = (pub_el.text or "").strip() if pub_el is not None else ""

        # Extract news snippet from ht:news_item if available
        snippet = ""
        news_item = item_el.find("ht:news_item", ns)
        if news_item is not None:
            news_title = news_item.find("ht:news_item_title", ns)
            news_snippet = news_item.find("ht:news_item_snippet", ns)
            t_text = clean_html(news_title.text if news_title is not None and news_title.text else "")
            s_text = clean_html(news_snippet.text if news_snippet is not None and news_snippet.text else "")
            snippet = f"{t_text}: {s_text}" if t_text and s_text else t_text or s_text

        items.append(
            TrendSourceItem(
                title=title,
                source="Google Trends",
                category="trending_spike",
                link=link,
                pub_date=pub_date,
                approx_traffic=approx_traffic,
                snippet=snippet[:300],
            )
        )
        if len(items) >= max_items:
            break

    return items


def collect_raw_us_trends(
    categories: Optional[List[str]] = None,
    include_google_trends: bool = True,
    items_per_category: int = 8,
) -> List[TrendSourceItem]:
    """Collect raw trend items from Google News US categories and Google Trends."""
    active_categories = categories or ["politics", "world", "entertainment"]
    collected: List[TrendSourceItem] = []

    for cat in active_categories:
        url = GOOGLE_NEWS_FEEDS.get(cat)
        if not url:
            continue
        try:
            xml_bytes = fetch_url_content(url)
            news_items = parse_google_news_rss(xml_bytes, category=cat, max_items=items_per_category)
            collected.extend(news_items)
        except Exception as exc:
            logger.warning("Error collecting Google News category '%s': %s", cat, exc)

    if include_google_trends:
        try:
            trends_bytes = fetch_url_content(GOOGLE_TRENDS_FEED)
            trend_items = parse_google_trends_rss(trends_bytes, max_items=items_per_category)
            collected.extend(trend_items)
        except Exception as exc:
            logger.warning("Error collecting Google Trends: %s", exc)

    return collected
