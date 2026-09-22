"""Tests for clippyme.domain.channel_service."""
import os
import tempfile
import pytest

from clippyme.domain.channel_service import (
    load_channels,
    save_channels,
    get_channel,
    create_channel,
    update_channel,
    delete_channel,
    match_channel_for_category,
    DEFAULT_CHANNELS,
)


def test_load_channels_seeds_defaults_if_empty():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "channels.json")
        channels = load_channels(file_path)
        assert len(channels) == len(DEFAULT_CHANNELS)
        assert channels[0]["id"] == "ch_us_politics"
        assert os.path.exists(file_path)


def test_create_and_get_channel():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "channels.json")
        created = create_channel(
            {
                "name": "Tech Breakthroughs",
                "description": "AI & robotics discoveries",
                "niches": ["tech", "ai", "science"],
                "banner_handle": "@TechBreakthroughs",
                "sub_preset": "hormozi_bold",
            },
            filepath=file_path,
        )
        assert created["id"].startswith("ch_tech_breakthroughs")
        assert created["name"] == "Tech Breakthroughs"
        assert created["banner_handle"] == "@TechBreakthroughs"

        fetched = get_channel(created["id"], filepath=file_path)
        assert fetched is not None
        assert fetched["name"] == "Tech Breakthroughs"


def test_update_and_delete_channel():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "channels.json")
        load_channels(file_path)

        updated = update_channel(
            "ch_us_politics",
            {"banner_handle": "@NewPoliticsHandle", "reframe_mode": "subject"},
            filepath=file_path,
        )
        assert updated is not None
        assert updated["banner_handle"] == "@NewPoliticsHandle"
        assert updated["reframe_mode"] == "subject"

        # Delete
        assert delete_channel("ch_us_politics", filepath=file_path) is True
        assert get_channel("ch_us_politics", filepath=file_path) is None
        assert delete_channel("nonexistent", filepath=file_path) is False


def test_match_channel_for_category():
    with tempfile.TemporaryDirectory() as td:
        file_path = os.path.join(td, "channels.json")
        load_channels(file_path)

        pol = match_channel_for_category("politics", filepath=file_path)
        assert pol is not None
        assert pol["id"] == "ch_us_politics"

        world = match_channel_for_category("breaking_world", filepath=file_path)
        assert world is not None
        assert world["id"] == "ch_breaking_world"

        pop = match_channel_for_category("entertainment", filepath=file_path)
        assert pop is not None
        assert pop["id"] == "ch_pop_entertainment"
