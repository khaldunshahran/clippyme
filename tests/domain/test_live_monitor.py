"""Tests for clippyme.domain.live_monitor pure helpers + config validation.

No curl_cffi / ffmpeg / event loop needed — LiveMonitor.__init__ is cheap and
the helpers under test are pure.
"""
import os
from datetime import date, datetime, timezone

import pytest

from clippyme.domain.errors import ConflictError, ValidationError
from clippyme.domain.live_monitor import (
    LiveMonitorRegistry,
    SharedGapScheduler,
    allocate_clip_filename,
    backfill_windows,
    build_backfill_cmd,
    build_monitor_compose,
    remaining_prelive,
    render_template,
    monitor_id_for,
    should_process_segment,
    validate_monitor_config,
    validate_monitor_partial_update,
    _hhmmss,
    _SNAPSHOT_CONFIG_FIELDS,
)
from clippyme.integrations.twitch_client import find_live_vod


# --- allocate_clip_filename (title-named, continuous collision counter) ----

def test_auto_title_unique_names_no_counter_bump():
    e = set()
    f1, c1 = allocate_clip_filename("{title}", {"title": "Litigio shock"}, e, 0)
    e.add(f1)
    f2, c2 = allocate_clip_filename("{title}", {"title": "Tradimento choc"}, e, c1)
    assert f1 == "Litigio shock.mp4"
    assert f2 == "Tradimento choc.mp4"
    assert c2 == 0                      # distinct titles → no numbering


def test_constant_custom_title_gets_continuous_counter():
    e, c = set(), 0
    names = []
    for _ in range(3):
        f, c = allocate_clip_filename("Clip", {"title": "x"}, e, c)
        e.add(f); names.append(f)
    assert names == ["Clip.mp4", "Clip_1.mp4", "Clip_2.mp4"]
    assert c == 2                        # continuous, never reset


def test_empty_template_falls_back_to_auto_title():
    f, _ = allocate_clip_filename("", {"video_title_for_youtube_short": "Ciao: mondo?"}, set(), 0)
    assert f == "Ciao mondo.mp4"


# --- build_monitor_compose (monitor auto-publish recipe) -------------------

def test_build_monitor_compose_defaults():
    clip = {"viral_hook_text": "Wait for it", "title": "T"}
    recipe = build_monitor_compose("kick", "grenbaud", clip, None)
    # smartcut is opt-in per monitor → off unless the config asks for it
    assert recipe["toggles"] == {
        "hook": True, "subtitles": True, "banner": True, "smartcut": False}
    assert recipe["hook_params"]["position"] == "top"
    assert recipe["hook_params"]["text"] == "Wait for it"
    # subtitles below the banner, left-aligned
    assert recipe["subtitle_params"] == {"position": "bottom", "align": "left"}
    # banner auto-injected from the monitor's platform + channel
    assert recipe["banner_params"]["platform"] == "kick"
    assert recipe["banner_params"]["handle"] == "grenbaud"
    assert recipe["banner_params"]["enabled"] is True


def test_build_monitor_compose_no_hook_text_disables_hook():
    recipe = build_monitor_compose("twitch", "chan", {}, None)
    assert recipe["toggles"]["hook"] is False


def test_build_monitor_compose_smart_cut_opt_in():
    recipe = build_monitor_compose("kick", "grenbaud", {"viral_hook_text": "x"},
                                   None, smart_cut=True)
    assert recipe["toggles"]["smartcut"] is True


def test_build_monitor_compose_banner_override_disables():
    recipe = build_monitor_compose("kick", "grenbaud", {"viral_hook_text": "x"},
                                   {"banner": {"enabled": False}})
    assert recipe["toggles"]["banner"] is False
    assert recipe["banner_params"] == {}


def test_build_monitor_compose_subtitle_override_merges():
    recipe = build_monitor_compose("kick", "grenbaud", {"viral_hook_text": "x"},
                                   {"subtitle_params": {"align": "center"}})
    assert recipe["subtitle_params"]["align"] == "center"
    assert recipe["subtitle_params"]["position"] == "bottom"  # default kept


def test_validate_monitor_config_carries_banner_and_compose():
    cfg = validate_monitor_config({
        "platform": "kick", "mode": "live", "slug": "grenbaud",
        "platforms": [{"platform": "tiktok", "accountId": "acc"}],
        "banner": {"enabled": False},
        "compose": {"subtitle_params": {"align": "center"}},
    })
    assert cfg["banner"] == {"enabled": False}
    assert cfg["compose"] == {"subtitle_params": {"align": "center"}}


def test_validate_monitor_config_smart_cut_and_zoom():
    base = {
        "platform": "kick", "mode": "live", "slug": "grenbaud",
        "platforms": [{"platform": "tiktok", "accountId": "acc"}],
    }
    cfg = validate_monitor_config(base)
    assert cfg["smart_cut"] is False      # opt-in: an extra render per clip
    assert cfg["letterbox_zoom"] == 0.0   # whole frame between the bars

    cfg = validate_monitor_config({**base, "smart_cut": True, "letterbox_zoom": 10})
    assert cfg["smart_cut"] is True
    assert cfg["letterbox_zoom"] == pytest.approx(0.10)

    # A garbage zoom must not stop a monitor from starting — it reads as off.
    assert validate_monitor_config({**base, "letterbox_zoom": "abc"})["letterbox_zoom"] == 0.0


# --- should_process_segment ------------------------------------------------

def test_should_process_segment_floor():
    assert should_process_segment(1800) is True
    assert should_process_segment(300) is True      # exactly 5 min → keep
    assert should_process_segment(299) is False     # just under → drop
    assert should_process_segment(0) is False
    assert should_process_segment(None) is False


# --- remaining_prelive -------------------------------------------------------

def test_remaining_prelive_none_started_at_falls_back_to_full_window():
    now = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    assert remaining_prelive(1800, None, now) == 1800


def test_remaining_prelive_fresh_stream_returns_partial():
    now = datetime(2026, 7, 21, 12, 10, tzinfo=timezone.utc)  # 10 min in
    started = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    assert remaining_prelive(1800, started, now) == 1200  # 30 - 10 min left


def test_remaining_prelive_already_old_stream_returns_zero():
    now = datetime(2026, 7, 21, 14, 0, tzinfo=timezone.utc)  # 2h in
    started = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    assert remaining_prelive(1800, started, now) == 0


def test_remaining_prelive_exactly_at_boundary():
    now = datetime(2026, 7, 21, 12, 30, tzinfo=timezone.utc)
    started = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    assert remaining_prelive(1800, started, now) == 0


# --- render_template -------------------------------------------------------

def test_render_template_fills_placeholders():
    clip = {"title": "My Clip", "viral_hook_text": "Wait for it"}
    assert render_template("{title} — {hook}", clip) == "My Clip — Wait for it"


def test_render_template_title_falls_back_to_youtube_short():
    # Gemini stores the viral title under video_title_for_youtube_short;
    # {title} must resolve it, not render an empty title (only the hashtag).
    clip = {"video_title_for_youtube_short": "Litigio shock"}
    assert render_template("{title} #tag", clip) == "Litigio shock #tag"


def test_render_template_empty_and_bad():
    assert render_template("", {}) == ""
    # Unknown placeholder → falls back to raw template, never raises.
    assert render_template("{nope}", {"title": "x"}) == "{nope}"


# --- SharedGapScheduler ----------------------------------------------------

def test_shared_gap_scheduler_enforces_spacing_across_picks():
    sched = SharedGapScheduler(min_gap_seconds=900)
    day = date(2026, 7, 21)  # a Tuesday with wide prime-time windows
    now = datetime(2026, 7, 21, 7, 0)
    slots = [sched.find_slot(day, [], now=now) for _ in range(5)]
    # Every pair of monitor-picked slots respects the 15-minute minimum gap.
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            assert abs((slots[i] - slots[j]).total_seconds()) >= 900
    assert len(sched.picked_slots) == 5


def test_shared_gap_scheduler_merges_external_occupied():
    sched = SharedGapScheduler(min_gap_seconds=900)
    day = date(2026, 7, 21)
    now = datetime(2026, 7, 21, 7, 0)
    external = datetime(2026, 7, 21, 10, 0)
    slot = sched.find_slot(day, [external], now=now)
    assert abs((slot - external).total_seconds()) >= 900


# --- validate_monitor_config -----------------------------------------------

def _base_cfg(**over):
    cfg = {"slug": "somechannel", "platforms": [{"platform": "tiktok", "accountId": "acc1"}]}
    cfg.update(over)
    return cfg


def test_validate_config_defaults():
    cfg = validate_monitor_config(_base_cfg())
    assert cfg["slug"] == "somechannel"
    assert cfg["segment_seconds"] == 1800
    assert cfg["prelive_skip_seconds"] == 1800
    assert cfg["min_gap_seconds"] == 900
    assert cfg["poll_interval"] == 60
    assert cfg["loop"] is False
    assert cfg["timezone"] == "Europe/Rome"


def test_validate_config_lowercases_and_bounds():
    cfg = validate_monitor_config(_base_cfg(
        slug="MixedCase", segment_seconds=99999, poll_interval=1, min_gap_seconds=-5))
    assert cfg["slug"] == "mixedcase"
    assert cfg["segment_seconds"] == 3600   # clamped to max
    assert cfg["poll_interval"] == 30       # clamped to min
    assert cfg["min_gap_seconds"] == 0      # clamped to floor


@pytest.mark.parametrize("bad", ["", "has space", "bad/slug", "UPPER!", "x" * 65])
def test_validate_config_rejects_bad_slug(bad):
    with pytest.raises(ValidationError):
        validate_monitor_config(_base_cfg(slug=bad))


def test_validate_config_requires_platforms():
    with pytest.raises(ValidationError):
        validate_monitor_config({"slug": "chan", "platforms": []})


def test_validate_config_rejects_incomplete_platform():
    with pytest.raises(ValidationError):
        validate_monitor_config({"slug": "chan", "platforms": [{"platform": "tiktok"}]})


def test_validate_config_custom_timezone_passthrough():
    cfg = validate_monitor_config(_base_cfg(timezone="America/New_York"))
    assert cfg["timezone"] == "America/New_York"


def test_validate_config_instructions_trimmed_and_capped():
    cfg = validate_monitor_config(_base_cfg(instructions="  find the funniest bits  "))
    assert cfg["instructions"] == "find the funniest bits"
    from clippyme.domain.job_results import MAX_INSTRUCTIONS_LEN
    cfg = validate_monitor_config(_base_cfg(instructions="x" * (MAX_INSTRUCTIONS_LEN + 1000)))
    assert len(cfg["instructions"]) == MAX_INSTRUCTIONS_LEN


def test_validate_config_instructions_defaults_empty():
    cfg = validate_monitor_config(_base_cfg())
    assert cfg["instructions"] == ""


# --- catchup (backfill vs live_only) ----------------------------------------

def test_validate_config_catchup_defaults_backfill():
    cfg = validate_monitor_config(_base_cfg())
    assert cfg["catchup"] == "backfill"


def test_validate_config_catchup_accepts_live_only():
    cfg = validate_monitor_config(_base_cfg(catchup="live_only"))
    assert cfg["catchup"] == "live_only"


def test_validate_config_catchup_rejects_bad_value():
    with pytest.raises(ValidationError):
        validate_monitor_config(_base_cfg(catchup="rewind"))


# --- multi-platform config -------------------------------------------------

def test_validate_config_platform_mode_defaults():
    cfg = validate_monitor_config(_base_cfg())
    assert cfg["platform"] == "kick"
    assert cfg["mode"] == "live"
    assert cfg["channel"] == "somechannel"
    assert cfg["poll_interval"] == 60  # live default


def test_validate_config_vod_poll_default():
    cfg = validate_monitor_config(_base_cfg(platform="youtube", mode="vod", slug="@somebody"))
    assert cfg["poll_interval"] == 600  # vod default
    assert cfg["channel"] == "@somebody"


def test_validate_config_youtube_live_accepted():
    cfg = validate_monitor_config(_base_cfg(platform="youtube", mode="live", slug="@x"))
    assert cfg["platform"] == "youtube"
    assert cfg["mode"] == "live"
    assert cfg["poll_interval"] == 60  # live default
    assert cfg["channel"] == "@x"
    assert cfg["stream_quality"] == "best"


def test_validate_config_stream_quality_custom_and_partial_update():
    cfg = validate_monitor_config(_base_cfg(stream_quality="1080p60,720p,best"))
    assert cfg["stream_quality"] == "1080p60,720p,best"
    updated = validate_monitor_partial_update({"stream_quality": "720p"}, cfg)
    assert updated["stream_quality"] == "720p"


def test_validate_config_bad_platform_or_mode():
    with pytest.raises(ValidationError):
        validate_monitor_config(_base_cfg(platform="rumble"))
    with pytest.raises(ValidationError):
        validate_monitor_config(_base_cfg(mode="clip"))


@pytest.mark.parametrize("chan", ["@handle", "UCabcdefghijklmnopqrstuv", "https://youtube.com/@x"])
def test_validate_config_youtube_channel_forms(chan):
    cfg = validate_monitor_config(_base_cfg(platform="youtube", mode="vod", slug=chan))
    assert cfg["channel"] == chan


def test_validate_config_youtube_rejects_junk_channel():
    with pytest.raises(ValidationError):
        validate_monitor_config(_base_cfg(platform="youtube", mode="vod", slug="not a handle"))


# --- validate_monitor_partial_update (runtime config updates) --------------

def _running_cfg(**over):
    return validate_monitor_config(_base_cfg(**over))


def test_partial_update_rejects_empty():
    with pytest.raises(ValidationError):
        validate_monitor_partial_update({}, _running_cfg())


def test_partial_update_rejects_unknown_field():
    with pytest.raises(ValidationError):
        validate_monitor_partial_update({"min_gap_seconds": 60, "channel": "other"}, _running_cfg())


@pytest.mark.parametrize(
    "field", ["platform", "mode", "channel", "slug", "loop", "publisher_mode", "catchup"])
def test_partial_update_rejects_identity_lifecycle_fields(field):
    with pytest.raises(ValidationError):
        validate_monitor_partial_update({field: "x"}, _running_cfg())


def test_partial_update_allows_updatable_fields_and_merges():
    current = _running_cfg(slug="grenbaud")
    new_cfg = validate_monitor_partial_update(
        {"min_gap_seconds": 120, "instructions": "focus on jokes"}, current)
    # unchanged fields carry over from current_cfg
    assert new_cfg["channel"] == "grenbaud"
    assert new_cfg["platform"] == "kick"
    # updated fields applied
    assert new_cfg["min_gap_seconds"] == 120
    assert new_cfg["instructions"] == "focus on jokes"


def test_partial_update_clamps_numeric_knobs_like_full_validation():
    current = _running_cfg()
    new_cfg = validate_monitor_partial_update({"segment_seconds": 99999}, current)
    assert new_cfg["segment_seconds"] == 3600  # clamped to max, same as validate_monitor_config


def test_partial_update_allows_banner_and_compose():
    current = _running_cfg()
    new_cfg = validate_monitor_partial_update(
        {"banner": {"enabled": False}, "compose": {"toggles": {"hook": False}}}, current)
    assert new_cfg["banner"] == {"enabled": False}
    assert new_cfg["compose"] == {"toggles": {"hook": False}}


def test_delete_after_publish_defaults_on():
    assert validate_monitor_config(_base_cfg())["delete_after_publish"] is True


def test_delete_after_publish_is_updatable_and_snapshotted():
    assert "delete_after_publish" in _SNAPSHOT_CONFIG_FIELDS
    current = _running_cfg()
    assert current["delete_after_publish"] is True
    new_cfg = validate_monitor_partial_update({"delete_after_publish": False}, current)
    assert new_cfg["delete_after_publish"] is False


def test_delete_after_publish_rejects_non_bool():
    with pytest.raises(ValidationError):
        validate_monitor_partial_update({"delete_after_publish": "yes"}, _running_cfg())


# --- LiveMonitor.update_config / LiveMonitorRegistry.update_config ---------

def test_monitor_update_config_swaps_cfg_and_persists(tmp_path):
    from clippyme.domain.live_monitor import LiveMonitor

    persisted = []
    mon = LiveMonitor(id="kick:foo", jobs={}, job_queue=None, output_dir=str(tmp_path),
                      on_state_change=lambda: persisted.append(True))
    mon.cfg = _running_cfg(slug="foo")

    result = mon.update_config({"min_gap_seconds": 42})

    assert mon.cfg["min_gap_seconds"] == 42
    assert result["min_gap_seconds"] == 42
    assert persisted  # _persist() called

    # returned dict is snapshot-shaped: only _SNAPSHOT_CONFIG_FIELDS keys
    from clippyme.domain.live_monitor import _SNAPSHOT_CONFIG_FIELDS
    assert set(result) == set(_SNAPSHOT_CONFIG_FIELDS)


def test_monitor_status_includes_config_allow_list(tmp_path):
    from clippyme.domain.live_monitor import _SNAPSHOT_CONFIG_FIELDS, LiveMonitor

    mon = LiveMonitor(id="kick:foo", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = _running_cfg(slug="foo")

    status = mon.status()

    assert "config" in status
    # Same allow-list snapshot() uses — no secrets by construction, and no
    # extra keys leaking through beyond what's explicitly allow-listed.
    assert set(status["config"]) == set(_SNAPSHOT_CONFIG_FIELDS)
    for key in _SNAPSHOT_CONFIG_FIELDS:
        assert status["config"][key] == mon.cfg.get(key)


def test_status_exposes_gemini_exhausted_at(tmp_path):
    from clippyme.domain.live_monitor import LiveMonitor

    mon = LiveMonitor(id="kick:foo", jobs={}, job_queue=None, output_dir=str(tmp_path))

    assert "gemini_exhausted_at" in mon.status()
    assert mon.status()["gemini_exhausted_at"] is None


def test_registry_update_config_not_found(tmp_path):
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path),
        state_path=str(tmp_path / "state.json"))
    with pytest.raises(__import__("clippyme.domain.errors", fromlist=["NotFoundError"]).NotFoundError):
        reg.update_config("kick:nope", {"min_gap_seconds": 60})


def test_registry_update_config_delegates_and_persists(tmp_path):
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path),
        state_path=str(tmp_path / "state.json"))

    class _FakeMon:
        id = "kick:foo"

        def update_config(self, partial):
            self.updated = partial
            return {"min_gap_seconds": partial["min_gap_seconds"]}

        def snapshot(self):
            return {"platform": "kick", "channel": "foo"}

    fake = _FakeMon()
    reg._monitors["kick:foo"] = fake

    result = reg.update_config("kick:foo", {"min_gap_seconds": 60})

    assert fake.updated == {"min_gap_seconds": 60}
    assert result == {"min_gap_seconds": 60}


def test_snapshot_includes_config_for_restart_survival(tmp_path):
    from clippyme.domain.live_monitor import LiveMonitor
    mon = LiveMonitor(id="kick:foo", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = _running_cfg(slug="foo", banner={"enabled": False}, compose={"toggles": {}})
    snap = mon.snapshot()
    assert snap["config"]["banner"] == {"enabled": False}
    assert snap["config"]["compose"] == {"toggles": {}}
    assert snap["config"]["min_gap_seconds"] == mon.cfg["min_gap_seconds"]


# --- global slot sharing across monitors -----------------------------------

def test_two_monitors_share_one_picked_slots_store():
    shared = []
    s1 = SharedGapScheduler(min_gap_seconds=900, picked_slots=shared)
    s2 = SharedGapScheduler(min_gap_seconds=900, picked_slots=shared)
    day = date(2026, 7, 21)
    now = datetime(2026, 7, 21, 7, 0)
    a = s1.find_slot(day, [], now=now)
    b = s2.find_slot(day, [], now=now)  # different scheduler, same store
    assert abs((a - b).total_seconds()) >= 900
    assert len(shared) == 2


# --- registry --------------------------------------------------------------

def test_registry_rejects_duplicate(tmp_path):
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path),
        state_path=str(tmp_path / "state.json"))

    class _FakeRunning:
        def is_running(self):
            return True
    reg._monitors["kick:foo"] = _FakeRunning()

    with pytest.raises(ConflictError):
        reg.start(_base_cfg(platform="kick", mode="live", slug="foo"))


def test_registry_stop_retires_monitor_but_keeps_snapshot(tmp_path):
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path),
        state_path=str(tmp_path / "state.json"))

    class _FakeMon:
        id = "kick:foo"

        async def stop(self):
            return {"id": self.id, "running": False}

        def status(self):
            return {"id": self.id, "running": False}

        def snapshot(self):
            return {"platform": "kick", "channel": "foo", "seen_ids": ["v1"]}

    reg._monitors["kick:foo"] = _FakeMon()
    import asyncio
    asyncio.run(reg.stop("kick:foo"))
    # Gone from the visible list…
    assert reg.status()["monitors"] == []
    # …but the guard snapshot survives for a future restart of the channel.
    assert reg._snapshots["kick:foo"]["seen_ids"] == ["v1"]


def test_loop_monitor_snapshot_marks_resume_on_start(tmp_path, monkeypatch):
    from clippyme.domain import live_monitor as lm
    from clippyme.storage import config_store

    monkeypatch.setattr(config_store, "load_persistent_config", lambda: {"GEMINI_API_KEY": "g"})
    monkeypatch.setattr(config_store, "load_zernio_config", lambda: {"timezone": "UTC", "api_key": "z"})
    monkeypatch.setattr(lm.LiveMonitor, "_make_strategy", lambda self, cfg, pc: object())

    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path),
        state_path=str(tmp_path / "state.json"))

    reg.start(_base_cfg(platform="kick", mode="live", slug="foo", loop=True, platforms=[{"platform": "tiktok", "accountId": "a1"}]))

    assert reg._monitors["kick:foo"].snapshot()["resume_on_start"] is True
    assert reg._monitors["kick:foo"].status()["resume_on_start"] is True


def test_registry_explicit_stop_disables_resume_on_start(tmp_path):
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path),
        state_path=str(tmp_path / "state.json"))

    class _FakeMon:
        id = "kick:foo"
        resume_on_start = True

        async def stop(self):
            return {"id": self.id, "running": False}

        def status(self):
            return {"id": self.id, "running": False, "resume_on_start": self.resume_on_start}

        def snapshot(self):
            return {
                "platform": "kick", "mode": "live", "channel": "foo",
                "config": {"platform": "kick", "mode": "live", "channel": "foo", "slug": "foo", "loop": True},
                "seen_ids": ["v1"], "published": ["/clips/a.mp4"],
                "manual_queued": {"/clips/b.mp4": "entry-1"},
                "segments_captured": 2, "clips_published": 3,
                "covered_elapsed": 42, "resume_on_start": self.resume_on_start,
            }

    reg._monitors["kick:foo"] = _FakeMon()

    import asyncio
    asyncio.run(reg.stop("kick:foo"))

    snap = reg._snapshots["kick:foo"]
    assert snap["resume_on_start"] is False
    assert snap["seen_ids"] == ["v1"]
    assert snap["published"] == ["/clips/a.mp4"]
    assert snap["manual_queued"] == {"/clips/b.mp4": "entry-1"}
    assert snap["segments_captured"] == 2
    assert snap["clips_published"] == 3
    assert snap["covered_elapsed"] == 42


def test_registry_shutdown_preserves_resume_on_start(tmp_path):
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path),
        state_path=str(tmp_path / "state.json"))

    class _FakeMon:
        id = "kick:foo"

        async def stop(self):
            return {"id": self.id, "running": False}

        def status(self):
            return {"id": self.id, "running": False}

        def snapshot(self):
            return {
                "platform": "kick", "mode": "live", "channel": "foo",
                "config": {"platform": "kick", "mode": "live", "channel": "foo", "slug": "foo", "loop": True},
                "resume_on_start": True, "seen_ids": ["v1"],
            }

    reg._monitors["kick:foo"] = _FakeMon()

    import asyncio
    asyncio.run(reg.shutdown())

    assert reg._snapshots["kick:foo"]["resume_on_start"] is True
    assert reg.status()["monitors"] == []


def test_registry_auto_resume_starts_marked_snapshots_and_preserves_guards(tmp_path, monkeypatch):
    from clippyme.domain import live_monitor as lm
    from clippyme.storage import config_store

    state = tmp_path / "state.json"
    state.write_text(__import__("json").dumps({
        "monitors": {
            "kick:foo": {
                "platform": "kick", "mode": "live", "channel": "foo",
                "config": {
                    "platform": "kick", "mode": "live", "channel": "foo", "slug": "foo",
                    "platforms": [{"platform": "tiktok", "accountId": "a1"}], "loop": True,
                    "segment_seconds": 120, "prelive_skip_seconds": 30,
                    "min_gap_seconds": 60, "poll_interval": 45, "timezone": "UTC",
                },
                "resume_on_start": True,
                "seen_ids": ["vod1"], "published": ["/clips/a.mp4"],
                "segments_captured": 4, "clips_published": 5,
                "covered_elapsed": 600, "covered_stream_start": "stream-start",
            }
        }
    }), encoding="utf-8")
    monkeypatch.setattr(config_store, "load_persistent_config", lambda: {"GEMINI_API_KEY": "fresh"})
    monkeypatch.setattr(config_store, "load_zernio_config", lambda: {"timezone": "UTC", "api_key": "z"})
    monkeypatch.setattr(lm.LiveMonitor, "_make_strategy", lambda self, cfg, pc: object())

    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path), state_path=str(state))

    import asyncio

    # NOTE: is_running() reflects a real asyncio.Task, which only stays
    # pending while its owning loop is running. asyncio.run() drains/cancels
    # every pending task before it returns, so the running-task check and the
    # shutdown must share ONE asyncio.run() call (one live loop), not two.
    async def _scenario():
        await reg.auto_resume()

        mon = reg._monitors["kick:foo"]
        assert mon.is_running()
        assert mon._gemini_key == "fresh"
        assert mon._seen_ids == {"vod1"}
        assert mon._published == {"/clips/a.mp4"}
        assert mon.segments_captured == 4
        assert mon.clips_published == 5
        assert mon._covered_elapsed == 600
        assert mon._covered_stream_start == "stream-start"

        await reg.shutdown()

    asyncio.run(_scenario())


def test_registry_auto_resume_failure_is_visible_in_status(tmp_path, monkeypatch):
    from clippyme.storage import config_store

    # Isolate from a real/leaked GEMINI_API_KEY env var (start()'s fallback):
    # this test asserts the specific "not configured" failure path.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    state = tmp_path / "state.json"
    state.write_text(__import__("json").dumps({
        "monitors": {
            "kick:foo": {
                "platform": "kick", "mode": "live", "channel": "foo",
                "config": {
                    "platform": "kick", "mode": "live", "channel": "foo", "slug": "foo",
                    "platforms": [{"platform": "tiktok", "accountId": "a1"}], "loop": True,
                },
                "resume_on_start": True,
            }
        }
    }), encoding="utf-8")
    monkeypatch.setattr(config_store, "load_persistent_config", lambda: {})
    monkeypatch.setattr(config_store, "load_zernio_config", lambda: {"timezone": "UTC"})
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path), state_path=str(state))

    import asyncio
    result = asyncio.run(reg.auto_resume())

    assert result["resumed"] == []
    assert result["failed"][0]["id"] == "kick:foo"
    status = reg.status()["monitors"]
    assert status[0]["id"] == "kick:foo"
    assert status[0]["running"] is False
    assert status[0]["state"] == "auto_resume_failed"
    assert "Gemini API key not configured" in status[0]["last_error"]
    assert reg._snapshots["kick:foo"]["resume_on_start"] is True


def test_registry_migrates_legacy_state(tmp_path):
    import json
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "slug": "oldchan", "published": ["/a.mp4"],
        "picked_slots": ["2026-07-21T10:00:00"],
    }), encoding="utf-8")
    reg = LiveMonitorRegistry(
        jobs={}, job_queue=None, output_dir=str(tmp_path), state_path=str(state))
    assert "kick:oldchan" in reg._snapshots
    assert reg._snapshots["kick:oldchan"]["published"] == ["/a.mp4"]
    assert len(reg._picked_slots) == 1


# --- _hhmmss ---------------------------------------------------------------

def test_hhmmss():
    assert _hhmmss(1800) == "00:30:00"
    assert _hhmmss(3661) == "01:01:01"
    assert _hhmmss(0) == "00:00:00"


# --- backfill_windows ------------------------------------------------------

def test_backfill_windows_already_live_two_hours():
    # 2h elapsed, skip first 30 min, 30-min segments → 3 full windows.
    windows = backfill_windows(7200, 1800, 1800)
    assert windows == [(1800, 3600), (3600, 5400), (5400, 7200)]


def test_backfill_windows_drops_short_trailing_chunk():
    # 5000s elapsed, skip 1800, 1800 segs → [1800-3600, 3600-5400→clamped 5000
    # (1400s < 300? no, 1400>=300 keep)]. Use a genuinely short tail instead.
    windows = backfill_windows(3700, 1800, 1800)  # tail 3600-3700 = 100s < 300
    assert windows == [(1800, 3600)]


def test_backfill_windows_nothing_to_recover():
    assert backfill_windows(1800, 1800, 1800) == []   # elapsed == skip
    assert backfill_windows(1000, 1800, 1800) == []   # elapsed < skip


def test_backfill_windows_clamps_bad_input():
    assert backfill_windows(-100, 1800, 1800) == []
    assert backfill_windows(7200, 1800, 0) == []      # zero segment
    assert backfill_windows("x", 1800, 1800) == []    # non-numeric


# --- build_backfill_cmd ----------------------------------------------------

def test_build_backfill_cmd_download_sections_format():
    cmd = build_backfill_cmd("https://www.twitch.tv/videos/42", 1800, 3600, "/out.mp4")
    assert "-m" in cmd and "yt_dlp" in cmd
    i = cmd.index("--download-sections")
    assert cmd[i + 1] == "*00:30:00-01:00:00"
    assert cmd[cmd.index("-o") + 1] == "/out.mp4"
    assert "--force-overwrites" in cmd and "-q" in cmd


# --- find_live_vod (twitch in-progress archive VOD) ------------------------

def test_find_live_vod_matches_stream_id():
    videos = {"data": [
        {"id": "v1", "stream_id": "999", "url": "https://twitch.tv/videos/v1"},
        {"id": "v2", "stream_id": "111"},
    ]}
    assert find_live_vod(videos, "999") == "https://twitch.tv/videos/v1"
    # falls back to a constructed url when the object omits it
    assert find_live_vod(videos, "111") == "https://www.twitch.tv/videos/v2"


def test_find_live_vod_no_match_or_malformed():
    assert find_live_vod({"data": [{"id": "v1", "stream_id": "1"}]}, "2") is None
    assert find_live_vod(None, "1") is None
    assert find_live_vod({}, "1") is None
    assert find_live_vod({"data": [{"id": "v1"}]}, "1") is None  # no stream_id
    assert find_live_vod({"data": [{"stream_id": "1"}]}, None) is None  # no stream id


# --- effective_backfill_start (restart mid-stream coverage guard) ----------

def test_effective_backfill_start_same_stream_uses_coverage():
    from clippyme.domain.live_monitor import effective_backfill_start
    # Same stream session → prior coverage floors the backfill start.
    assert effective_backfill_start(1800, 40000, "2026-07-20T10:00:00+00:00",
                                    "2026-07-20T10:00:00+00:00") == 40000
    # Coverage below prelive skip → prelive wins.
    assert effective_backfill_start(1800, 600, "s", "s") == 1800


def test_effective_backfill_start_new_stream_ignores_coverage():
    from clippyme.domain.live_monitor import effective_backfill_start
    assert effective_backfill_start(1800, 40000, "2026-07-20T10:00:00+00:00",
                                    "2026-07-21T09:00:00+00:00") == 1800
    assert effective_backfill_start(1800, 40000, None, "x") == 1800
    assert effective_backfill_start(1800, 40000, "x", None) == 1800


def test_snapshot_restore_roundtrips_coverage(tmp_path):
    from clippyme.domain.live_monitor import LiveMonitor
    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None,
                      output_dir=str(tmp_path))
    mon._covered_elapsed = 12345
    mon._covered_stream_start = "2026-07-20T10:00:00+00:00"
    snap = mon.snapshot()
    mon2 = LiveMonitor(id="kick:chan", jobs={}, job_queue=None,
                       output_dir=str(tmp_path))
    mon2.restore(snap)
    assert mon2._covered_elapsed == 12345
    assert mon2._covered_stream_start == "2026-07-20T10:00:00+00:00"


# --- catchup="live_only" never recovers pre-start footage -------------------

def test_schedule_backfill_live_only_skips_windows_and_tasks(tmp_path):
    """live_only must never queue missed windows, a kick VOD baseline, or a
    Twitch in-progress-VOD backfill task — only bookkeep 'now' as covered."""
    import asyncio

    from clippyme.domain.live_monitor import LiveMonitor

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = _running_cfg(catchup="live_only")
    mon.platform = "kick"
    started_at = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)

    def boom(*a, **k):
        raise AssertionError("backfill fetch_vods must not be called in live_only")

    mon._strategy = type("S", (), {"fetch_vods": boom})()

    asyncio.run(mon._schedule_backfill(started_at))

    assert mon._missed_windows == []
    assert mon._vod_baseline_ids == set()
    assert mon.backfill_pending == 0
    assert mon._covered_stream_start == started_at.isoformat()
    assert mon._covered_elapsed > 0  # "now", not the prelive-skip offset
    # no backfill task was ever scheduled
    assert not mon._publish_tasks


def test_schedule_backfill_backfill_mode_still_queues_kick_windows(tmp_path):
    """Unchanged default behaviour: catchup='backfill' still queues windows."""
    import asyncio

    from clippyme.domain.live_monitor import LiveMonitor

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = _running_cfg()  # default catchup="backfill"
    mon.platform = "kick"
    from datetime import timedelta
    started_at = datetime.now(timezone.utc) - timedelta(hours=2)

    mon._strategy = type("S", (), {"fetch_vods": lambda self: []})()

    asyncio.run(mon._schedule_backfill(started_at))

    assert mon._missed_windows  # windows queued, unlike live_only
    assert mon.backfill_pending == len(mon._missed_windows)


def test_recover_kick_backfill_noop_in_live_only(tmp_path):
    """Defence-in-depth: even if _missed_windows were somehow non-empty,
    _recover_kick_backfill must not submit a job when catchup='live_only'."""
    import asyncio

    from clippyme.domain.live_monitor import LiveMonitor

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = _running_cfg(catchup="live_only")
    mon._missed_windows = [(0, 100)]

    def boom(*a, **k):
        raise AssertionError("must not run when catchup='live_only'")

    mon._backfill_windows = boom

    asyncio.run(mon._recover_kick_backfill())  # returns immediately, no error


def test_backfill_from_vod_noop_in_live_only(tmp_path):
    import asyncio

    from clippyme.domain.live_monitor import LiveMonitor

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = _running_cfg(catchup="live_only")

    def boom(*a, **k):
        raise AssertionError("must not run when catchup='live_only'")

    mon._backfill_windows = boom

    asyncio.run(mon._backfill_from_vod([(0, 100)]))


# --- _publish_one: 429 backoff + retry -------------------------------------

def test_publish_one_retries_on_429_then_succeeds(tmp_path, monkeypatch):
    """A Zernio 429 must be retried after backoff, not drop the clip."""
    import asyncio

    from clippyme.domain import live_monitor as lm
    from clippyme.domain.live_monitor import LiveMonitor
    from clippyme.integrations import social_publisher as sp
    from clippyme.integrations.social_publisher import ZernioError

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")

    monkeypatch.setattr(lm, "PUBLISH_429_BACKOFF_SECONDS", 0)
    calls = []

    def fake_publish_clip(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise ZernioError("Zernio POST /posts → HTTP 429",
                              status_code=429, body="rate limited")

    monkeypatch.setattr(sp, "publish_clip", fake_publish_clip)

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None,
                      output_dir=str(tmp_path))
    mon.cfg = {"title_template": "{title}", "caption_template": "{hook}",
               "platforms": [{"platform": "tiktok", "accountId": "a"}],
               "timezone": "Europe/Rome"}
    mon._zernio_key = "sk_test"
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "viral_hook_text": "H"}
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(job_dir / "c.mp4")}

    asyncio.run(mon._publish_one(entry))

    assert len(calls) == 3
    assert mon.clips_published == 1


def test_publish_one_rolls_start_date_on_daily_limit(tmp_path, monkeypatch):
    """A 'Daily limit reached' 429 must reschedule on the next day, not sleep-retry."""
    import asyncio
    from datetime import date, timedelta

    from clippyme.domain.live_monitor import LiveMonitor
    from clippyme.integrations import social_publisher as sp
    from clippyme.integrations.social_publisher import ZernioError

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")

    calls = []

    def fake_publish_clip(**kwargs):
        calls.append(kwargs.get("start_date"))
        if len(calls) < 3:
            raise ZernioError(
                "Zernio POST /posts → HTTP 429", status_code=429,
                body='{"error":"Daily limit reached for this account: 5/5 posts today."}')

    monkeypatch.setattr(sp, "publish_clip", fake_publish_clip)

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None,
                      output_dir=str(tmp_path))
    mon.cfg = {"title_template": "{title}", "caption_template": "{hook}",
               "platforms": [{"platform": "tiktok", "accountId": "a"}],
               "timezone": "Europe/Rome"}
    mon._zernio_key = "sk_test"
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "viral_hook_text": "H"}
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(job_dir / "c.mp4")}

    asyncio.run(mon._publish_one(entry))

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    day_after = (date.today() + timedelta(days=2)).isoformat()
    assert calls == [None, tomorrow, day_after]
    assert mon.clips_published == 1


def test_publish_one_non_429_fails_without_retry(tmp_path, monkeypatch):
    import asyncio

    from clippyme.domain.live_monitor import LiveMonitor
    from clippyme.integrations import social_publisher as sp
    from clippyme.integrations.social_publisher import ZernioError

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")

    calls = []

    def fake_publish_clip(**kwargs):
        calls.append(kwargs)
        raise ZernioError("Zernio POST /posts → HTTP 500", status_code=500, body="boom")

    monkeypatch.setattr(sp, "publish_clip", fake_publish_clip)

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None,
                      output_dir=str(tmp_path))
    mon.cfg = {"title_template": "{title}", "caption_template": "{hook}",
               "platforms": [{"platform": "tiktok", "accountId": "a"}],
               "timezone": "Europe/Rome"}
    mon._zernio_key = "sk_test"
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "viral_hook_text": "H"}
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(job_dir / "c.mp4")}

    asyncio.run(mon._publish_one(entry))

    assert len(calls) == 1
    assert mon.clips_published == 0
    assert "HTTP 500" in (mon.last_error or "")


# --- pause / resume auto-publish -------------------------------------------

def _publishing_monitor(tmp_path, monkeypatch, *, jobs=None):
    """A LiveMonitor wired with a fake publish_clip (records calls, succeeds)."""
    from clippyme.domain.live_monitor import LiveMonitor
    from clippyme.integrations import social_publisher as sp

    calls = []
    monkeypatch.setattr(sp, "publish_clip", lambda **kw: calls.append(kw))
    mon = LiveMonitor(id="kick:chan", jobs=jobs if jobs is not None else {},
                      job_queue=None, output_dir=str(tmp_path))
    mon.cfg = {"title_template": "{title}", "caption_template": "{hook}",
               "platforms": [{"platform": "tiktok", "accountId": "a"}],
               "timezone": "Europe/Rome"}
    mon._zernio_key = "sk_test"

    # Skip the real ffmpeg compose pass — publish/delete behaviour is what these
    # tests exercise; publish the raw clip path directly.
    async def _no_compose(job_id, clip, base_path=None):
        return base_path
    mon._compose_for_publish = _no_compose
    return mon, calls


def test_paused_publish_queues_instead_of_publishing(tmp_path, monkeypatch):
    """publishing_enabled=False → clip goes to pending, publish_clip NOT called,
    and the flag + pending round-trip through snapshot/restore."""
    import asyncio

    from clippyme.domain.live_monitor import LiveMonitor

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")

    mon, calls = _publishing_monitor(tmp_path, monkeypatch)
    mon.publishing_enabled = False
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "viral_hook_text": "H"}
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(job_dir / "c.mp4")}

    asyncio.run(mon._publish_one(entry))

    assert calls == []                       # nothing published while paused
    assert len(mon._pending_publish) == 1
    assert mon.clips_published == 0

    snap = mon.snapshot()
    assert snap["publishing_enabled"] is False
    assert snap["pending_publish"] == [entry]

    fresh = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    fresh.restore(snap)
    assert fresh.publishing_enabled is False
    assert fresh._pending_publish == [entry]


def test_resume_drains_pending_in_order_with_spacing(tmp_path, monkeypatch):
    """Resume drains queued clips through the publish path, in order, spacing
    consecutive publishes."""
    import asyncio

    from clippyme.domain import live_monitor as lm

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "a.mp4").write_bytes(b"x")
    (job_dir / "b.mp4").write_bytes(b"x")

    sleeps = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(lm.asyncio, "sleep", fake_sleep)

    mon, calls = _publishing_monitor(tmp_path, monkeypatch)
    mon.publishing_enabled = False
    for name in ("a.mp4", "b.mp4"):
        asyncio.run(mon._publish_one({
            "job_id": "job1",
            "clip": {"video_url": f"/videos/job1/{name}", "title": name},
            "composed_path": str(job_dir / name)}))
    assert calls == [] and len(mon._pending_publish) == 2

    mon.publishing_enabled = True
    asyncio.run(mon._drain_pending())

    published_paths = [os.path.basename(c["clip_path"]) for c in calls]
    assert published_paths == ["a.mp4", "b.mp4"]           # order preserved
    assert lm.PUBLISH_SPACING_SECONDS in sleeps             # spacing applied
    assert mon._pending_publish == []
    assert mon.clips_published == 2


def test_drain_recomposes_missing_composed_path_and_survives_failure(tmp_path, monkeypatch):
    """A restored pending entry whose composed_path vanished is recomposed on
    drain; a recompose that raises re-queues that entry (retried, not lost) and
    never aborts draining the other pending clips."""
    import asyncio

    from clippyme.domain import live_monitor as lm

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "b.mp4").write_bytes(b"x")           # the good sibling
    recomposed = job_dir / "a_recomposed.mp4"
    recomposed.write_bytes(b"x")

    async def fake_sleep(secs):
        pass
    monkeypatch.setattr(lm.asyncio, "sleep", fake_sleep)

    mon, calls = _publishing_monitor(tmp_path, monkeypatch)

    compose_calls = []

    async def spy_compose(job_id, clip, base_path=None):
        compose_calls.append(clip)
        if len(compose_calls) == 1:                 # first recompose fails
            raise RuntimeError("compose boom")
        return str(recomposed)                      # retry succeeds
    mon._compose_for_publish = spy_compose

    a = {"job_id": "job1",
         "clip": {"video_url": "/videos/job1/a.mp4", "title": "A"},
         "composed_path": str(job_dir / "missing.mp4")}   # gone → triggers recompose
    b = {"job_id": "job1",
         "clip": {"video_url": "/videos/job1/b.mp4", "title": "B"},
         "composed_path": str(job_dir / "b.mp4")}
    mon.publishing_enabled = True
    mon._pending_publish = [a, b]

    asyncio.run(mon._drain_pending())

    assert compose_calls                             # recompose was invoked
    assert mon._pending_publish == []                # everything drained, nothing lost
    published = {os.path.basename(c["clip_path"]) for c in calls}
    assert published == {"b.mp4", "a_recomposed.mp4"}  # sibling published + entry retried
    assert mon.clips_published == 2


def test_successful_publish_deletes_clip_files_and_empty_dir(tmp_path, monkeypatch):
    """A published clip's artifacts are removed; when it was the last clip the
    whole job dir goes away. The consolidated per-monitor composed file is
    also removed (delete_after_publish default True)."""
    import asyncio

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")
    (job_dir / "source_c.mp4").write_bytes(b"x")
    (job_dir / "c_cover.jpg").write_bytes(b"x")
    # Job-dir composed file is now title-based (composed_clip_basename): the
    # clip's title "T" → "T.mp4", not the legacy composed_clip_0.mp4.
    (job_dir / "composed_T_clip_1.mp4").write_bytes(b"x")
    (job_dir / "run_metadata.json").write_text(
        '{"status": "completed", "shorts": [{"video_url": "/videos/job1/c.mp4"}]}')
    monitor_dir = tmp_path / "monitor_kick_chan"
    monitor_dir.mkdir()
    composed = monitor_dir / "Title.mp4"
    composed.write_bytes(b"x")

    jobs = {"job1": {"status": "completed"}}
    mon, calls = _publishing_monitor(tmp_path, monkeypatch, jobs=jobs)
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "original_index": 0}
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(composed)}

    asyncio.run(mon._publish_one(entry))

    assert len(calls) == 1
    assert mon.clips_published == 1
    assert not job_dir.exists()          # last clip → job dir removed
    assert not composed.exists()         # consolidated file removed too


def test_successful_publish_keeps_dir_and_marks_metadata_when_clips_remain(tmp_path, monkeypatch):
    """Deleting one clip marks its metadata entry and leaves siblings intact."""
    import asyncio
    import json

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")
    (job_dir / "d.mp4").write_bytes(b"x")   # sibling clip stays
    meta = job_dir / "run_metadata.json"
    meta.write_text(json.dumps({"shorts": [
        {"video_url": "/videos/job1/c.mp4"},
        {"video_url": "/videos/job1/d.mp4"},
    ]}))

    jobs = {"job1": {"status": "completed"}}
    mon, calls = _publishing_monitor(tmp_path, monkeypatch, jobs=jobs)
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "original_index": 0}
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(job_dir / "c.mp4")}

    asyncio.run(mon._publish_one(entry))

    assert job_dir.exists()
    assert not (job_dir / "c.mp4").exists()
    assert (job_dir / "d.mp4").exists()
    data = json.loads(meta.read_text())
    assert data["shorts"][0].get("deleted_after_publish") is True
    assert "deleted_after_publish" not in data["shorts"][1]


def test_delete_after_publish_false_keeps_artifacts_but_marks_published(tmp_path, monkeypatch):
    """delete_after_publish=False: clip is recorded as published (no
    re-publish) but its raw artifacts + metadata are left untouched."""
    import asyncio
    import json

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")
    (job_dir / "source_c.mp4").write_bytes(b"x")
    meta = job_dir / "run_metadata.json"
    meta.write_text(json.dumps({"shorts": [{"video_url": "/videos/job1/c.mp4"}]}))
    monitor_dir = tmp_path / "monitor_kick_chan"
    monitor_dir.mkdir()
    composed = monitor_dir / "Title.mp4"
    composed.write_bytes(b"x")

    jobs = {"job1": {"status": "completed"}}
    mon, calls = _publishing_monitor(tmp_path, monkeypatch, jobs=jobs)
    mon.cfg["delete_after_publish"] = False
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "original_index": 0}
    clip_path = os.path.join(str(tmp_path), "job1", "c.mp4")
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(composed)}

    asyncio.run(mon._publish_one(entry))

    assert len(calls) == 1
    assert mon.clips_published == 1
    assert clip_path in mon._published        # dedupe still recorded
    assert (job_dir / "c.mp4").exists()        # artifacts kept
    assert (job_dir / "source_c.mp4").exists()
    assert composed.exists()                   # consolidated file kept too
    data = json.loads(meta.read_text())
    assert "deleted_after_publish" not in data["shorts"][0]

    # Re-publishing the same entry is a no-op (dedupe holds even without deletion).
    calls.clear()
    asyncio.run(mon._publish_one(entry))
    assert calls == []


def test_deletion_failure_does_not_raise_and_clip_stays_published(tmp_path, monkeypatch):
    """os.remove blowing up must not surface — the publish already succeeded."""
    import asyncio

    from clippyme.domain import live_monitor as lm

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")

    # RuntimeError escapes _safe_remove's OSError guard → exercises the outer
    # best-effort try/except in _delete_clip_artifacts.
    def boom(path):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(lm.os, "remove", boom)

    mon, calls = _publishing_monitor(tmp_path, monkeypatch)
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "original_index": 0}
    entry = {"job_id": "job1", "clip": clip, "composed_path": str(job_dir / "c.mp4")}

    asyncio.run(mon._publish_one(entry))   # must not raise

    assert len(calls) == 1
    assert mon.clips_published == 1


def test_registry_set_publishing_unknown_id_raises_not_found(tmp_path):
    from clippyme.domain.errors import NotFoundError
    from clippyme.domain.live_monitor import LiveMonitorRegistry

    reg = LiveMonitorRegistry(jobs={}, job_queue=None, output_dir=str(tmp_path),
                              state_path=str(tmp_path / "state.json"))
    with pytest.raises(NotFoundError):
        reg.set_publishing("kick:nope", False)


def test_restored_monitor_with_pending_and_enabled_drains_on_start(tmp_path, monkeypatch):
    """Regression for I2: a snapshot taken mid-drain (publishing_enabled=True,
    non-empty pending) must not sit stuck after restart until someone toggles
    pause/resume — start() must kick the existing drain helper itself."""
    import asyncio

    from clippyme.domain import live_monitor as lm
    from clippyme.storage import config_store

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    (job_dir / "c.mp4").write_bytes(b"x")

    monkeypatch.setattr(config_store, "load_persistent_config", lambda: {"GEMINI_API_KEY": "g"})
    monkeypatch.setattr(config_store, "load_zernio_config", lambda: {"timezone": "UTC", "api_key": "z"})
    monkeypatch.setattr(lm.LiveMonitor, "_make_strategy", lambda self, cfg, pc: object())

    mon, calls = _publishing_monitor(tmp_path, monkeypatch)

    # Simulate a snapshot restored mid-drain: enabled, with a queued clip —
    # exactly what LiveMonitor.restore() would produce from disk.
    clip = {"video_url": "/videos/job1/c.mp4", "title": "T", "original_index": 0}
    mon.publishing_enabled = True
    mon._pending_publish = [
        {"job_id": "job1", "clip": clip, "composed_path": str(job_dir / "c.mp4")}]

    # Stub out the real live/vod loop entirely — this test only cares that
    # start() schedules the existing _drain_pending() helper, not that the
    # monitor loop runs.
    async def fake_run():
        return None
    mon._run = fake_run

    cfg = validate_monitor_config(_base_cfg(
        platform="kick", mode="live", slug="chan",
        platforms=[{"platform": "tiktok", "accountId": "a"}]))

    async def go():
        mon.start(cfg)
        # Let the scheduled drain task (and the stubbed _run task) complete.
        for _ in range(20):
            await asyncio.sleep(0)
        if mon._publish_tasks:
            await asyncio.gather(*mon._publish_tasks)

    asyncio.run(go())

    assert len(calls) == 1
    assert mon._pending_publish == []
    assert mon.clips_published == 1
    assert mon._draining is False   # guard released, not left stuck


def test_validate_config_malformed_max_clips_uses_default():
    cfg = validate_monitor_config(_base_cfg(max_clips="not-an-int"))
    assert cfg["max_clips"] == 5


def test_zero_max_clips_survives_as_uncapped(tmp_path):
    from clippyme.domain.live_monitor import LiveMonitor

    cfg = validate_monitor_config(_base_cfg(max_clips=0))
    assert cfg["max_clips"] == 0        # 0 = no cap, never clamped back up to 1

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon._gemini_key = "k"
    mon.cfg = cfg
    _, _, env = mon._new_job_dir()
    # The pipeline reads CLIPPYME_MAX_CLIPS=0 as "keep every clip".
    assert env["CLIPPYME_MAX_CLIPS"] == "0"


def test_validate_config_clip_selection_defaults_and_bounds():
    cfg = validate_monitor_config(_base_cfg())
    assert cfg["clip_selection"] == "fixed" and cfg["min_viral_score"] == 70
    auto = validate_monitor_config(_base_cfg(clip_selection="AUTO", min_viral_score=300))
    assert auto["clip_selection"] == "auto" and auto["min_viral_score"] == 100
    with pytest.raises(ValidationError):
        validate_monitor_config(_base_cfg(clip_selection="sometimes"))


def test_clip_selection_env_only_set_in_auto_mode(tmp_path):
    from clippyme.domain.live_monitor import LiveMonitor

    mon = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon._gemini_key = "k"
    mon.cfg = validate_monitor_config(_base_cfg(max_clips=8))
    _, _, env = mon._new_job_dir()
    assert env["CLIPPYME_MAX_CLIPS"] == "8"
    assert "CLIPPYME_MIN_VIRAL_SCORE" not in env

    mon.cfg = validate_monitor_config(
        _base_cfg(max_clips=8, clip_selection="auto", min_viral_score=82))
    _, _, env = mon._new_job_dir()
    assert env["CLIPPYME_MAX_CLIPS"] == "8"          # still the ceiling
    assert env["CLIPPYME_MIN_VIRAL_SCORE"] == "82"   # plus the quality floor


def test_monitor_snapshot_restores_pending_backfill(tmp_path):
    import json
    from clippyme.domain.live_monitor import LiveMonitor
    original = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    original._missed_windows = [(1800, 3600), (3600, 5400)]
    original._vod_baseline_ids = {"old-vod"}
    original._backfill_baseline_ready = True
    snap = json.loads(json.dumps(original.snapshot()))
    restored = LiveMonitor(id="kick:chan", jobs={}, job_queue=None, output_dir=str(tmp_path))
    restored.restore(snap)
    assert restored._missed_windows == [(1800, 3600), (3600, 5400)]
    assert restored.backfill_pending == 2
    assert restored._vod_baseline_ids == {"old-vod"}
    assert restored._backfill_baseline_ready is True



def test_youtube_monitor_id_is_stable_and_path_safe():
    channel = "https://www.youtube.com/@ExampleCreator"
    first = monitor_id_for("youtube", channel)
    assert first == monitor_id_for("youtube", channel)
    assert first.startswith("youtube:")
    assert "/" not in first
    assert len(first) == len("youtube:") + 20


def test_non_youtube_monitor_id_remains_human_readable():
    assert monitor_id_for("kick", "creator") == "kick:creator"


def test_vod_start_offset_skips_prelive_only_where_a_prelive_exists():
    from types import SimpleNamespace

    from clippyme.domain.live_monitor import LiveMonitor

    cfg = {"prelive_skip_seconds": 1800}
    # A Kick/Twitch VOD is the recording of a live: same waiting screen at the head.
    kick = SimpleNamespace(platform="kick", cfg=cfg)
    assert LiveMonitor._vod_start_offset(kick) == 1800
    # A YouTube upload has no prelive — trimming would eat real content.
    yt = SimpleNamespace(platform="youtube", cfg=cfg)
    assert LiveMonitor._vod_start_offset(yt) == 0.0


def test_youtube_strategy_get_live_state_and_capture_args(monkeypatch, tmp_path):
    from clippyme.domain.live_monitor import YoutubeStrategy

    strat = YoutubeStrategy("@TestCreator")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        "clippyme.integrations.youtube_feed.check_youtube_live",
        lambda chan: (True, "https://example.com/live.m3u8", now),
    )
    is_live, stream_url, started_at = strat.get_live_state()
    assert is_live is True
    assert stream_url == "https://example.com/live.m3u8"
    assert started_at == now

    # Direct URL capture args (ffmpeg copy)
    args = strat.capture_args(str(tmp_path / "seg.mp4"), 1800, "https://example.com/live.m3u8")
    assert args[0] == "ffmpeg"
    assert "https://example.com/live.m3u8" in args
    assert "-c" in args and "copy" in args

    # Streamlink capture args fallback
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/streamlink" if cmd == "streamlink" else None)
    monkeypatch.setattr(
        "clippyme.integrations.youtube_feed.get_youtube_live_stream_url",
        lambda url, quality="best": None,
    )
    args_sl = strat.capture_args(str(tmp_path / "seg.mp4"), 1800, "https://youtube.com/@TestCreator/live", quality="720p")
    assert args_sl is not None
    assert args_sl[0] == "streamlink"
    assert "720p" in args_sl


def test_strategy_quality_passed_to_capture_args(monkeypatch, tmp_path):
    from clippyme.domain.live_monitor import KickStrategy, TwitchStrategy

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/streamlink" if cmd == "streamlink" else None)

    kick = KickStrategy("streamer")
    args_k = kick.capture_args(str(tmp_path / "seg.mp4"), 1800, None, quality="1080p60,best")
    assert "1080p60,best" in args_k

    client = type("TwitchClientMock", (), {})()
    twitch = TwitchStrategy("streamer", client)
    args_t = twitch.capture_args(str(tmp_path / "seg.mp4"), 1800, None, quality="720p60,best")
    assert "720p60,best" in args_t


def test_live_monitor_status_telemetry(tmp_path):
    import time
    from clippyme.domain.live_monitor import LiveMonitor

    seg_file = tmp_path / "test_seg.mp4"
    seg_file.write_bytes(b"X" * 1024)

    mon = LiveMonitor(id="kick:test", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.state = "capturing"
    mon._active_segment_path = str(seg_file)
    mon._active_segment_start = time.time() - 10

    st = mon.status()
    assert st["current_segment_bytes"] == 1024
    assert st["current_segment_seconds"] >= 9


def test_live_monitor_anti_flap_check_reconnects():
    import asyncio
    from clippyme.domain.live_monitor import LiveMonitor

    async def _run():
        mon = LiveMonitor(id="kick:test", jobs={}, job_queue=None, output_dir="output")
        mon.state = "capturing"

        call_count = [0]
        def mock_get_live():
            call_count[0] += 1
            if call_count[0] >= 2:
                return True, "url", None
            return False, None, None

        mon._strategy = type("StrategyMock", (), {"get_live_state": staticmethod(mock_get_live)})()
        mon._interruptible_sleep = lambda secs: asyncio.sleep(0.001)

        res = await mon._anti_flap_check()
        assert res is True
        assert mon.state == "capturing"

    asyncio.run(_run())


def test_live_monitor_anti_flap_check_confirms_offline():
    import asyncio
    from clippyme.domain.live_monitor import LiveMonitor

    async def _run():
        mon = LiveMonitor(id="kick:test", jobs={}, job_queue=None, output_dir="output")
        mon.state = "capturing"

        mon._strategy = type("StrategyMock", (), {"get_live_state": staticmethod(lambda: (False, None, None))})()
        mon._interruptible_sleep = lambda secs: asyncio.sleep(0.001)

        res = await mon._anti_flap_check()
        assert res is False

    asyncio.run(_run())


def test_live_monitor_require_preview_config_and_status(tmp_path):
    from clippyme.domain.live_monitor import LiveMonitor, validate_monitor_config

    cfg = validate_monitor_config({
        "platform": "kick",
        "channel": "streamer",
        "platforms": [{"platform": "tiktok", "accountId": "acc1"}],
        "require_preview": True,
    })
    assert cfg["require_preview"] is True

    mon = LiveMonitor(id="kick:streamer", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = cfg
    status = mon.status()
    assert status["require_preview"] is True
    assert status["pending_clips"] == []


def test_live_monitor_pending_clips_flow(tmp_path, monkeypatch):
    import asyncio
    from clippyme.domain.live_monitor import LiveMonitor

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    clip_file = job_dir / "clip1.mp4"
    clip_file.write_bytes(b"dummy")

    published_calls = []
    def mock_publish_clip(**kw):
        published_calls.append(kw)

    from clippyme.integrations import social_publisher as sp
    monkeypatch.setattr(sp, "publish_clip", mock_publish_clip)

    mon = LiveMonitor(id="kick:streamer", jobs={}, job_queue=None, output_dir=str(tmp_path))
    mon.cfg = {
        "channel": "streamer",
        "platforms": [{"platform": "tiktok", "accountId": "acc1"}],
        "require_preview": True,
        "delete_after_publish": False,
        "timezone": "Europe/Rome",
    }
    mon._zernio_key = "sk_test"

    clip = {
        "title": "Insane Play",
        "video_title_for_youtube_short": "Insane Play",
        "video_description": "Watch this insane play right now! #gaming #viral",
        "hashtags": ["#gaming", "#viral"],
        "speaker_name": "Streamer",
        "start": 10.0,
        "end": 45.0,
        "viral_score": 95,
        "video_url": "/videos/job1/clip1.mp4",
    }

    # Queue an entry as consolidated
    entry = {
        "id": "clip_abc",
        "job_id": "job1",
        "clip": clip,
        "composed_path": str(clip_file),
        "filename": "clip1.mp4",
        "title": "Insane Play",
        "caption": "Watch this insane play right now! #gaming #viral",
        "speaker_name": "Streamer",
        "hashtags": ["#gaming", "#viral"],
        "viral_score": 95,
        "duration": 35.0,
        "status": "pending_preview",
    }
    mon._pending_publish.append(entry)

    # 1. get_pending_clips returns formatted preview data
    pending = mon.get_pending_clips()
    assert len(pending) == 1
    assert pending[0]["id"] == "clip_abc"
    assert pending[0]["title"] == "Insane Play"
    assert pending[0]["viral_score"] == 95
    assert pending[0]["duration"] == 35.0
    assert "/videos/monitor_kick_streamer/clip1.mp4" in pending[0]["video_url"]

    # 2. publish_pending_clip publishes the clip with optional overrides
    res = asyncio.run(mon.publish_pending_clip("clip_abc", overrides={
        "title": "Updated Viral Title",
        "caption": "Custom Caption #epic",
        "schedule_mode": "manual",
        "scheduled_for": "2026-09-07T12:00:00Z",
    }))
    assert res["success"] is True
    assert len(mon._pending_publish) == 0
    assert len(published_calls) == 1
    assert published_calls[0]["title"] == "Updated Viral Title"
    assert published_calls[0]["caption"] == "Custom Caption #epic"
    assert published_calls[0]["schedule_mode"] == "manual"
    assert published_calls[0]["scheduled_for"] == "2026-09-07T12:00:00Z"
    assert mon.clips_published == 1

    # 3. dismiss_pending_clip removes from queue and cleans file
    clip_file2 = job_dir / "clip2.mp4"
    clip_file2.write_bytes(b"dummy2")
    entry2 = {
        "id": "clip_xyz",
        "job_id": "job1",
        "clip": clip,
        "composed_path": str(clip_file2),
        "filename": "clip2.mp4",
    }
    mon._pending_publish.append(entry2)
    assert len(mon.get_pending_clips()) == 1
    dismiss_res = mon.dismiss_pending_clip("clip_xyz")
    assert dismiss_res["success"] is True
    assert len(mon.get_pending_clips()) == 0
    assert not clip_file2.exists()


