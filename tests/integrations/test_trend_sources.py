"""Tests for clippyme.integrations.trend_sources pure helpers (no network)."""
import pytest
from clippyme.integrations.trend_sources import (
    clean_html,
    parse_google_news_rss,
    parse_google_trends_rss,
    collect_raw_us_trends,
)

SAMPLE_NEWS_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News - Politics</title>
    <item>
      <title>Senate Judiciary Committee Debates AI Regulations - Washington Post</title>
      <link>https://news.google.com/rss/articles/CBMi123</link>
      <pubDate>Wed, 16 Sep 2026 01:30:00 GMT</pubDate>
      <description>&lt;a href="..."&gt;Full coverage&lt;/a&gt; Senators questioned tech CEOs on safety standards.</description>
    </item>
    <item>
      <title>White House Announces Infrastructure Grants - Reuters</title>
      <link>https://news.google.com/rss/articles/CBMi456</link>
      <pubDate>Wed, 16 Sep 2026 00:15:00 GMT</pubDate>
      <description>President signed the updated initiative into law.</description>
    </item>
  </channel>
</rss>"""

SAMPLE_TRENDS_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:ht="https://trends.google.com/trending/rss" version="2.0">
  <channel>
    <title>Daily Search Trends</title>
    <item>
      <title>Nepal earthquake 2026</title>
      <ht:approx_traffic>500K+</ht:approx_traffic>
      <link>https://trends.google.com/trending/rss?geo=US</link>
      <pubDate>Wed, 16 Sep 2026 02:00:00 GMT</pubDate>
      <ht:news_item>
        <ht:news_item_title>Rescue operations underway in Nepal after massive tremor</ht:news_item_title>
        <ht:news_item_snippet>Emergency services report tremors felt across Kathmandu valley.</ht:news_item_snippet>
      </ht:news_item>
    </item>
  </channel>
</rss>"""


def test_clean_html():
    assert clean_html("<b>Hello</b> <a href='#'>world</a>!") == "Hello world !"
    assert clean_html("") == ""
    assert clean_html(None) == ""


def test_parse_google_news_rss():
    items = parse_google_news_rss(SAMPLE_NEWS_RSS, category="politics", max_items=10)
    assert len(items) == 2
    first = items[0]
    assert first.title == "Senate Judiciary Committee Debates AI Regulations"
    assert first.source == "Washington Post"
    assert first.category == "politics"
    assert "Senators questioned tech CEOs" in first.snippet
    assert "Full coverage" in first.snippet
    assert "<a" not in first.snippet

    second = items[1]
    assert second.title == "White House Announces Infrastructure Grants"
    assert second.source == "Reuters"


def test_parse_google_trends_rss():
    items = parse_google_trends_rss(SAMPLE_TRENDS_RSS, max_items=10)
    assert len(items) == 1
    item = items[0]
    assert item.title == "Nepal earthquake 2026"
    assert item.approx_traffic == "500K+"
    assert item.source == "Google Trends"
    assert item.category == "trending_spike"
    assert "Rescue operations underway" in item.snippet


def test_parse_empty_or_malformed_feeds():
    assert parse_google_news_rss(b"") == []
    assert parse_google_news_rss(b"not-xml") == []
    assert parse_google_trends_rss(b"") == []
    assert parse_google_trends_rss(b"<broken>") == []


def test_collect_raw_us_trends_mocked(monkeypatch):
    def fake_fetch(url, timeout=12.0):
        if "trends.google.com" in url:
            return SAMPLE_TRENDS_RSS
        return SAMPLE_NEWS_RSS

    monkeypatch.setattr("clippyme.integrations.trend_sources.fetch_url_content", fake_fetch)

    items = collect_raw_us_trends(categories=["politics"], include_google_trends=True)
    assert len(items) == 3  # 2 news + 1 trend
    titles = [i.title for i in items]
    assert "Nepal earthquake 2026" in titles
    assert "Senate Judiciary Committee Debates AI Regulations" in titles
