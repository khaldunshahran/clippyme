"""Generative AI Shorts / UGC Video Creator for ClippyMe.

Generates complete marketing videos with AI actors, voiceover, b-roll, and subtitles
from a URL or product description.
"""
from clippyme.ugc.research import research_product_online, scrape_product_website
from clippyme.ugc.scripting import generate_ugc_scripts

__all__ = [
    "research_product_online",
    "scrape_product_website",
    "generate_ugc_scripts",
]
