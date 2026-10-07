"""Editorial benchmark foundation (Task 07, rev3).

Addresses review2 five groups:
1. Real media_qa.inspect_clip API with expected_duration/aspect.
2. Manifest/review paths protected; output collisions rejected.
3. Per-run frozen identities (source_fingerprints); clip fingerprints
   preserved; non-tautological pairing.
4. Zero-candidate sources in denominators; complete coverage.
5. Strict validation; reproducible synthetic demo.
"""
from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

GENRES = (
    "interview/podcast",
    "reaction/comedy",
    "gaming/stream",
    "tutorial/screen",
    "sport/action",
    "performance/visual",
)

MANIFEST_VERSION = 1


class BenchmarkError(Exception):
    pass


def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        # Huge ints cause OverflowError on float conversion.
        try:
            f = float(v)
        except OverflowError:
            return False
        return math.isfinite(f)
    return isinstance(v, float) and math.isfinite(v)


def _resolve_asset(asset_root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise BenchmarkError(f"invalid asset path: {rel!r}")
    root = asset_root.resolve()
    p = (root / rel).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise BenchmarkError(f"asset traversal rejected: {rel}")
    if not p.is_file():
        raise BenchmarkError(f"asset missing: {rel}")
    return p


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def load_manifest(path: str | Path) -> dict:
    p = Path(path).resolve()
    with open(p, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise BenchmarkError(f"manifest JSON invalid: {e}")
    if not isinstance(data, dict):
        raise BenchmarkError("manifest root must be object")
    if data.get("version") != MANIFEST_VERSION:
        raise BenchmarkError("manifest version must be 1")

    asset_root = (p.parent / data.get("asset_root", ".")).resolve()
    if not asset_root.is_dir():
        raise BenchmarkError("asset_root not a directory")

    sources_raw = data.get("sources")
    if not isinstance(sources_raw, list):
        raise BenchmarkError("sources must be list")
    sources = {}
    for s in sources_raw:
        if not isinstance(s, dict):
            raise BenchmarkError("source must be object")
        sid = s.get("id")
        if not isinstance(sid, str) or not sid or sid in sources:
            raise BenchmarkError(f"invalid source id: {sid!r}")
        genre = s.get("genre")
        if genre not in GENRES:
            raise BenchmarkError(f"source {sid}: invalid genre")
        dur = s.get("duration")
        if not _is_finite_number(dur) or float(dur) <= 0:
            raise BenchmarkError(f"source {sid}: invalid duration")
        # synthetic must be bool, not truthy string.
        syn = s.get("synthetic", False)
        if not isinstance(syn, bool):
            raise BenchmarkError(
                f"source {sid}: synthetic must be boolean")
        ap = _resolve_asset(asset_root, s.get("file", ""))
        fp = _fingerprint(ap)
        if "fingerprint" in s and s["fingerprint"] != fp:
            raise BenchmarkError(f"source {sid}: stale fingerprint")
        # Validate typed fields.
        commit = s.get("commit", "")
        if not isinstance(commit, str):
            raise BenchmarkError(f"source {sid}: commit must be string")
        sources[sid] = {
            "id": sid, "genre": genre, "duration": float(dur),
            "file": s.get("file"), "_path": str(ap),
            "fingerprint": fp, "synthetic": syn,
        }

    runs_raw = data.get("runs")
    if not isinstance(runs_raw, list):
        raise BenchmarkError("runs must be list")
    runs = []
    run_ids = set()
    clip_ids = set()
    for r in runs_raw:
        if not isinstance(r, dict):
            raise BenchmarkError("run must be object")
        rid = r.get("run_id")
        if not isinstance(rid, str) or not rid or rid in run_ids:
            raise BenchmarkError(f"invalid run_id: {rid!r}")
        run_ids.add(rid)
        commit = r.get("commit", "")
        if not isinstance(commit, str):
            raise BenchmarkError(f"run {rid}: commit must be string")
        settings = r.get("settings", {})
        if not isinstance(settings, dict):
            raise BenchmarkError(f"run {rid}: settings must be object")
        # Explicit source membership (including zero-candidate).
        # Required, unique, immutable.
        source_ids = r.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise BenchmarkError(
                f"run {rid}: source_ids required (non-empty list)")
        if len(source_ids) != len(set(source_ids)):
            raise BenchmarkError(
                f"run {rid}: duplicate source_ids")
        for sid in source_ids:
            if sid not in sources:
                raise BenchmarkError(f"run {rid}: unknown source {sid!r}")
        # Per-run frozen source fingerprints: required, exact keys.
        frozen = r.get("source_fingerprints")
        if not isinstance(frozen, dict) or not frozen:
            raise BenchmarkError(
                f"run {rid}: source_fingerprints required (frozen)")
        if set(frozen.keys()) != set(source_ids):
            raise BenchmarkError(
                f"run {rid}: source_fingerprints keys must match source_ids")
        for sid, fp in frozen.items():
            if not isinstance(fp, str) or not fp.startswith("sha256:"):
                raise BenchmarkError(f"run {rid}: invalid fingerprint")
            if fp != sources[sid]["fingerprint"]:
                raise BenchmarkError(
                    f"run {rid}: source {sid} fingerprint mismatch "
                    f"(stale run or changed source)")
        clips_raw = r.get("clips", [])
        if not isinstance(clips_raw, list):
            raise BenchmarkError(f"run {rid}: clips must be list")
        clips = []
        for c in clips_raw:
            if not isinstance(c, dict):
                raise BenchmarkError("clip must be object")
            cid = c.get("clip_id")
            if not isinstance(cid, str) or not cid or cid in clip_ids:
                raise BenchmarkError(f"invalid clip_id: {cid!r}")
            clip_ids.add(cid)
            sid = c.get("source_id")
            if sid not in sources:
                raise BenchmarkError(f"clip {cid}: unknown source")
            st, en = c.get("start"), c.get("end")
            if not (_is_finite_number(st) and _is_finite_number(en)):
                raise BenchmarkError(f"clip {cid}: invalid interval")
            st, en = float(st), float(en)
            if not (0 <= st < en <= sources[sid]["duration"]):
                raise BenchmarkError(f"clip {cid}: interval out of range")
            clip_file = c.get("file")
            if not isinstance(clip_file, str) or not clip_file:
                raise BenchmarkError(f"clip {cid}: missing file")
            cp = _resolve_asset(asset_root, clip_file)
            cfp = _fingerprint(cp)
            if "fingerprint" in c and c["fingerprint"] != cfp:
                raise BenchmarkError(f"clip {cid}: stale fingerprint")
            mid = c.get("moment_id")
            if mid is not None and (not isinstance(mid, str) or not mid):
                raise BenchmarkError(f"clip {cid}: invalid moment_id")
            if sid not in source_ids:
                raise BenchmarkError(
                    f"clip {cid}: source {sid} not in run {rid} membership")
            clips.append({
                "clip_id": cid, "source_id": sid,
                "start": st, "end": en, "moment_id": mid,
                "file": clip_file, "_clip_path": str(cp),
                "fingerprint": cfp,
            })
        runs.append({
            "run_id": rid, "commit": commit, "settings": settings,
            "source_ids": source_ids,
            "source_fingerprints": dict(frozen),
            "clips": clips,
        })

    return {"version": MANIFEST_VERSION,
            "asset_root": str(asset_root),
            "manifest_path": str(p),
            "sources": sources, "runs": runs}


def _parse_bool_strict(v: Any, field: str, cid: str,
                       from_json: bool = False) -> bool | None:
    # Unknown: None, "", omitted. Invalid: anything else.
    # JSON: only actual booleans accepted; strings rejected.
    # CSV: documented tokens "true"/"false"/"1"/"0" allowed.
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if from_json:
        # JSON strings are never valid booleans.
        raise BenchmarkError(
            f"clip {cid}: invalid boolean for {field}: {v!r} "
            f"(JSON requires true/false, not strings)")
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in ("true", "1", "yes"):
            return True
        if lv in ("false", "0", "no"):
            return False
    raise BenchmarkError(
        f"clip {cid}: invalid boolean for {field}: {v!r}")


def load_reviews(path: str | Path) -> dict[str, dict]:
    p = Path(path).resolve()
    if p.suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            fe = r.get("framing_errors", "")
            r["framing_errors"] = (
                [x.strip() for x in fe.split(",") if x.strip()]
                if isinstance(fe, str) and fe else
                fe if isinstance(fe, list) else None)
    else:
        with open(p, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise BenchmarkError(f"reviews JSON invalid: {e}")
        if not isinstance(data, dict):
            raise BenchmarkError("reviews root must be object")
        rows = data.get("reviews", [])
        if not isinstance(rows, list):
            raise BenchmarkError("reviews must be list")

    reviews = {}
    from_json = p.suffix != ".csv"
    for r in rows:
        if not isinstance(r, dict):
            raise BenchmarkError("review must be object")
        cid = r.get("clip_id")
        if not isinstance(cid, str) or not cid or cid in reviews:
            raise BenchmarkError(f"invalid clip_id: {cid!r}")
        fe = r.get("framing_errors")
        # None/omitted -> unknown (None), not [].
        if fe is None or fe == "":
            fe_val = None
        elif isinstance(fe, list):
            if not all(isinstance(x, str) for x in fe):
                raise BenchmarkError(f"clip {cid}: framing_errors elements must be strings")
            fe_val = fe
        else:
            raise BenchmarkError(
                f"clip {cid}: framing_errors must be list or null")
        notes = r.get("notes", "")
        if not isinstance(notes, str):
            raise BenchmarkError(f"clip {cid}: notes must be string")
        reviews[cid] = {
            "clip_id": cid,
            "opening_understandable": _parse_bool_strict(
                r.get("opening_understandable"), "opening", cid, from_json),
            "payoff_delivered": _parse_bool_strict(
                r.get("payoff_delivered"), "payoff", cid, from_json),
            "ending_natural": _parse_bool_strict(
                r.get("ending_natural"), "ending", cid, from_json),
            "framing_errors": fe_val,  # None = unknown
            "hook_obstruction": _parse_bool_strict(
                r.get("hook_obstruction"), "hook_obstruction", cid, from_json),
            "caption_obstruction": _parse_bool_strict(
                r.get("caption_obstruction"), "caption_obstruction", cid, from_json),
            "accepted": _parse_bool_strict(r.get("accepted"), "accepted", cid, from_json),
            "notes": notes,
        }
    return {"reviews": reviews, "reviews_path": str(p)}


def _strict_json(obj: Any) -> str:
    def _check(o):
        if isinstance(o, float) and not math.isfinite(o):
            raise BenchmarkError("non-finite in output")
        if isinstance(o, dict):
            for v in o.values():
                _check(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _check(v)
    _check(obj)
    return json.dumps(obj, indent=2, sort_keys=True)


def _media_check(path: str, expected_duration: float | None,
                 expected_aspect: float | None) -> dict:
    """Real media_qa.inspect_clip with correct API."""
    try:
        from clippyme.pipeline.media_qa import inspect_clip
        r = inspect_clip(
            path,
            expected_duration=expected_duration,
            expected_aspect=expected_aspect)
        # Real schema: ok (bool), critical (bool), issues (list),
        # warnings (list), metrics (dict).
        return {
            "ok": bool(r.get("ok")),
            "critical": bool(r.get("critical")),
            "issues": list(r.get("issues", [])),
            "warnings": list(r.get("warnings", [])),
            "errors": list(r.get("issues", [])),
            "invocation_error": None,
            "duration": r.get("metrics", {}).get("duration"),
            "width": r.get("metrics", {}).get("width"),
            "height": r.get("metrics", {}).get("height"),
        }
    except TypeError as e:
        return {"ok": False, "critical": True,
                "issues": [], "warnings": [],
                "errors": [], "invocation_error": str(e),
                "duration": None, "width": None, "height": None}
    except Exception as e:
        return {"ok": False, "critical": True,
                "issues": [], "warnings": [],
                "errors": [], "invocation_error": str(e),
                "duration": None, "width": None, "height": None}


def run_benchmark(manifest: dict, reviews_data: dict,
                  check_media: bool = False) -> dict:
    reviews = reviews_data["reviews"] if isinstance(
        reviews_data, dict) and "reviews" in reviews_data else reviews_data

    sources = manifest["sources"]
    runs = manifest["runs"]
    all_cids = {c["clip_id"] for r in runs for c in r["clips"]}
    for cid in reviews:
        if cid not in all_cids:
            raise BenchmarkError(f"unknown review clip_id: {cid!r}")

    media_results = {}
    if check_media:
        for sid, s in sources.items():
            # Expected aspect from source? Use None; duration validated.
            media_results[f"source:{sid}"] = _media_check(
                s["_path"], expected_duration=s["duration"],
                expected_aspect=None)
        for r in runs:
            for c in r["clips"]:
                exp_dur = c["end"] - c["start"]
                media_results[f"clip:{c['clip_id']}"] = _media_check(
                    c["_clip_path"], expected_duration=exp_dur,
                    expected_aspect=None)

    run_stats = []
    for r in runs:
        # Explicit source membership (including zero-candidate).
        run_sources = r["source_ids"]
        run_source_hours = sum(
            sources[sid]["duration"] for sid in run_sources) / 3600.0

        clips = r["clips"]
        candidate = len(clips)
        reviewed = accepted = 0
        moments = set()
        labeled_accepted = 0
        # Judgments.
        j_known = {"opening": 0, "payoff": 0, "ending": 0,
                   "framing": 0, "hook": 0, "caption": 0}
        j_ok = {"opening": 0, "payoff": 0, "ending": 0}
        j_defects = {"framing": 0, "hook": 0, "caption": 0}

        for c in clips:
            rev = reviews.get(c["clip_id"])
            if rev is None:
                continue
            acc = rev.get("accepted")
            if acc is not None:
                reviewed += 1
                if acc:
                    accepted += 1
                    mid = c.get("moment_id")
                    if mid:
                        moments.add((c["source_id"], mid))
                        labeled_accepted += 1
            # Coverage: known vs unknown.
            if rev.get("opening_understandable") is not None:
                j_known["opening"] += 1
                if rev["opening_understandable"]:
                    j_ok["opening"] += 1
            if rev.get("payoff_delivered") is not None:
                j_known["payoff"] += 1
                if rev["payoff_delivered"]:
                    j_ok["payoff"] += 1
            if rev.get("ending_natural") is not None:
                j_known["ending"] += 1
                if rev["ending_natural"]:
                    j_ok["ending"] += 1
            fe = rev.get("framing_errors")
            if fe is not None:  # None = unknown, [] = known no defects
                j_known["framing"] += 1
                if fe:
                    j_defects["framing"] += 1
            ho = rev.get("hook_obstruction")
            if ho is not None:
                j_known["hook"] += 1
                if ho:
                    j_defects["hook"] += 1
            co = rev.get("caption_obstruction")
            if co is not None:
                j_known["caption"] += 1
                if co:
                    j_defects["caption"] += 1

        # Distinct moments.
        if accepted == 0:
            distinct = 0 if (reviewed == candidate and candidate > 0) else None
        elif labeled_accepted == accepted:
            distinct = len(moments)
        else:
            distinct = None

        rate = (accepted / reviewed) if reviewed else None
        aph = (accepted / run_source_hours) if run_source_hours else None

        # Per-run frozen fingerprints (required, validated at load).
        run_fps = r["source_fingerprints"]

        run_stats.append({
            "run_id": r["run_id"], "commit": r["commit"],
            "settings": r["settings"],
            "source_ids": run_sources,
            "source_fingerprints": run_fps,
            "candidate_count": candidate,
            "reviewed_count": reviewed,
            "accepted_count": accepted,
            "acceptance_rate": rate,
            "distinct_accepted_moments": distinct,
            "moment_label_coverage": (
                labeled_accepted / accepted if accepted else None),
            "source_hours": round(run_source_hours, 4),
            "accepted_per_source_hour": (
                round(aph, 2) if aph is not None else None),
            "review_completeness": (
                reviewed / candidate if candidate else None),
            "judgments": {
                "opening": {"known": j_known["opening"], "ok": j_ok["opening"]},
                "payoff": {"known": j_known["payoff"], "ok": j_ok["payoff"]},
                "ending": {"known": j_known["ending"], "ok": j_ok["ending"]},
                "framing": {"known": j_known["framing"],
                            "defects": j_defects["framing"]},
                "framing_known": j_known["framing"],
                "hook_obstruction": {"known": j_known["hook"],
                                     "defects": j_defects["hook"]},
                "caption_obstruction": {"known": j_known["caption"],
                                        "defects": j_defects["caption"]},
            },
            "clips": [
                {"clip_id": c["clip_id"], "source_id": c["source_id"],
                 "start": c["start"], "end": c["end"],
                 "moment_id": c.get("moment_id"),
                 "fingerprint": c["fingerprint"]}
                for c in clips],
        })

    # Genres (all six).
    genre_stats = {}
    for g in GENRES:
        g_items = []
        for r in runs:
            for c in r["clips"]:
                if sources[c["source_id"]]["genre"] == g:
                    g_items.append((r["run_id"], c))
        reviewed = sum(
            1 for _, c in g_items
            if reviews.get(c["clip_id"], {}).get("accepted") is not None)
        accepted = sum(
            1 for _, c in g_items
            if reviews.get(c["clip_id"], {}).get("accepted") is True)
        # Distinct: consistent rule with per-run.
        moments = set()
        labeled = 0
        for _, c in g_items:
            if reviews.get(c["clip_id"], {}).get("accepted") is True:
                mid = c.get("moment_id")
                if mid:
                    moments.add((c["source_id"], mid))
                    labeled += 1
        if accepted == 0:
            g_distinct = 0 if (reviewed == len(g_items) and g_items) else None
        elif labeled == accepted:
            g_distinct = len(moments)
        else:
            g_distinct = None
        # Per-run/per-genre from declared membership.
        by_run = {}
        for r in runs:
            rid = r["run_id"]
            # Sources of this genre in this run's membership.
            g_sources = [sid for sid in r["source_ids"]
                         if sources[sid]["genre"] == g]
            if not g_sources:
                continue
            g_hours = sum(sources[sid]["duration"]
                          for sid in g_sources) / 3600.0
            d = by_run[rid] = {
                "candidate": 0, "reviewed": 0, "accepted": 0,
                "source_count": len(g_sources),
                "source_hours": g_hours,
            }
        for rid, c in g_items:
            d = by_run[rid]
            d["candidate"] += 1
            acc = reviews.get(c["clip_id"], {}).get("accepted")
            if acc is not None:
                d["reviewed"] += 1
                if acc:
                    d["accepted"] += 1
        # Add rates to each cell.
        for rid, d in by_run.items():
            d["acceptance_rate"] = (
                d["accepted"] / d["reviewed"] if d["reviewed"] else None)
            d["review_completeness"] = (
                d["reviewed"] / d["candidate"] if d["candidate"] else None)
            d["accepted_per_source_hour"] = (
                round(d["accepted"] / d["source_hours"], 2)
                if d["source_hours"] else None)
        genre_stats[g] = {
            "candidate_count": len(g_items),
            "reviewed_count": reviewed,
            "accepted_count": accepted,
            "acceptance_rate": (accepted / reviewed) if reviewed else None,
            "distinct_accepted_moments": g_distinct,
            "by_run": by_run,
        }

    # Paired: use per-run frozen fingerprints (not tautological).
    paired = {}
    for sid in sources:
        runs_for = {}
        for r in runs:
            if sid in r["source_ids"]:
                # Frozen fingerprint for this run+source.
                fp = r["source_fingerprints"].get(
                    sid, sources[sid]["fingerprint"])
                cids = [c["clip_id"] for c in r["clips"]
                        if c["source_id"] == sid]
                runs_for[r["run_id"]] = {
                    "clip_ids": cids, "commit": r["commit"],
                    "source_fingerprint": fp,
                }
        if runs_for:
            fps = {v["source_fingerprint"] for v in runs_for.values()}
            # Also compare against current.
            current_fp = sources[sid]["fingerprint"]
            all_match = len(fps) == 1 and current_fp in fps
            paired[sid] = {
                "runs": runs_for,
                "fingerprints_compatible": all_match,
                "current_fingerprint": current_fp,
                "missing_runs": [r["run_id"] for r in runs
                                 if r["run_id"] not in runs_for],
            }

    return {
        "version": MANIFEST_VERSION,
        "run_stats": run_stats,
        "genre_stats": genre_stats,
        "paired_sources": paired,
        "media_checks": media_results,
        "note": ("Real-footage benchmark has not run; synthetic only."
                 if all(s.get("synthetic") for s in sources.values())
                 else "Includes real footage."),
    }


def format_report(stats: dict) -> str:
    lines = ["Editorial Benchmark Report", "=" * 30, ""]
    for rs in stats["run_stats"]:
        rate = rs["acceptance_rate"]
        rate_s = f"{rate:.1%}" if rate is not None else "unknown"
        dm = rs["distinct_accepted_moments"]
        dm_s = str(dm) if dm is not None else "unknown"
        comp = rs["review_completeness"]
        comp_s = f"{comp:.0%}" if comp is not None else "n/a"
        aph = rs["accepted_per_source_hour"]
        aph_s = str(aph) if aph is not None else "unknown"
        lines.append(f"Run {rs['run_id']} ({rs['commit']}):")
        lines.append(f"  Sources: {rs['source_ids']} "
                     f"({rs['source_hours']}h)")
        lines.append(f"  Candidates: {rs['candidate_count']}, Reviewed: "
                     f"{rs['reviewed_count']} ({comp_s}), Accepted: "
                     f"{rs['accepted_count']} ({rate_s})")
        lines.append(f"  Distinct moments: {dm_s}, Accepted/hour: {aph_s}")
        j = rs["judgments"]
        lines.append(
            f"  Opening: {j['opening']['ok']}/{j['opening']['known']}, "
            f"Payoff: {j['payoff']['ok']}/{j['payoff']['known']}, "
            f"Ending: {j['ending']['ok']}/{j['ending']['known']}")
        lines.append(
            f"  Framing: {j['framing']['defects']} defects / "
            f"{j['framing']['known']} reviewed, "
            f"Hook obstruction: {j['hook_obstruction']['defects']}/"
            f"{j['hook_obstruction']['known']}, "
            f"Caption: {j['caption_obstruction']['defects']}/"
            f"{j['caption_obstruction']['known']}")
    lines.append("")
    lines.append("Per genre (all six):")
    for g, gs in stats["genre_stats"].items():
        rate = gs["acceptance_rate"]
        rate_s = f"{rate:.1%}" if rate is not None else "no data"
        dm = gs["distinct_accepted_moments"]
        dm_s = str(dm) if dm is not None else "unknown"
        lines.append(f"  {g}: {gs['accepted_count']}/"
                     f"{gs['reviewed_count']} ({rate_s}), distinct: {dm_s}")
    lines.append("")
    lines.append("Paired sources:")
    for sid, p in stats["paired_sources"].items():
        runs = ", ".join(p["runs"].keys())
        compat = "compatible" if p["fingerprints_compatible"] else "MISMATCH"
        missing = (f", missing: {p['missing_runs']}"
                   if p["missing_runs"] else "")
        lines.append(f"  {sid}: [{runs}] {compat}{missing}")
    if stats["media_checks"]:
        lines.append("")
        lines.append("Media checks (objective):")
        for k, v in stats["media_checks"].items():
            s = "ok" if v["ok"] else "FAIL"
            lines.append(f"  {k}: {s}")
    lines.append("")
    lines.append(stats["note"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Editorial benchmark.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--json-out")
    ap.add_argument("--text-out")
    ap.add_argument("--check-media", action="store_true")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    reviews_path = Path(args.reviews).resolve()

    try:
        manifest = load_manifest(manifest_path)
        reviews_data = load_reviews(reviews_path)
        stats = run_benchmark(manifest, reviews_data,
                              check_media=args.check_media)
        json_str = _strict_json(stats)
        text_str = format_report(stats)

        # Protect inputs and reject collisions.
        protected = {manifest_path, reviews_path,
                     Path(manifest["asset_root"]).resolve()}
        for s in manifest["sources"].values():
            protected.add(Path(s["_path"]).resolve())
        for r in manifest["runs"]:
            for c in r["clips"]:
                protected.add(Path(c["_clip_path"]).resolve())

        def _check_out(p):
            if not p:
                return
            rp = Path(p).resolve()
            # Real containment: reject if inside asset root.
            asset_root = Path(manifest["asset_root"]).resolve()
            try:
                rp.relative_to(asset_root)
                raise BenchmarkError(f"output inside asset root: {p}")
            except ValueError:
                pass
            if rp in protected:
                raise BenchmarkError(f"output collides with input: {p}")
            # Identical destinations.
            if args.json_out and args.text_out:
                if Path(args.json_out).resolve() == Path(
                        args.text_out).resolve():
                    raise BenchmarkError("json-out and text-out identical")
            # Parent must exist or be creatable.
            parent = rp.parent
            if not parent.exists():
                raise BenchmarkError(f"output parent missing: {parent}")

        if args.json_out:
            _check_out(args.json_out)
        if args.text_out:
            _check_out(args.text_out)
    except BenchmarkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: I/O failure: {e}", file=sys.stderr)
        return 1

    # Write outputs atomically: backup existing, track created.
    # On failure, restore backups and delete created files.
    backups = {}  # path -> original bytes (for existing files)
    created = []  # paths we created (did not exist before)
    try:
        if args.json_out:
            jp = Path(args.json_out)
            if jp.exists():
                backups[jp] = jp.read_bytes()
            else:
                created.append(jp)
            jp.write_text(json_str, encoding="utf-8")
        if args.text_out:
            tp = Path(args.text_out)
            if tp.exists():
                backups[tp] = tp.read_bytes()
            else:
                created.append(tp)
            tp.write_text(text_str, encoding="utf-8")
    except OSError as e:
        # Restore existing files, delete newly created.
        for p, data in backups.items():
            try:
                p.write_bytes(data)
            except OSError:
                pass
        for p in created:
            try:
                p.unlink()
            except OSError:
                pass
        print(f"error: write failed: {e}", file=sys.stderr)
        return 1

    if not args.json_out:
        print(json_str)
    if not args.text_out and not args.json_out:
        print("\n" + text_str)
    elif args.text_out and not args.json_out:
        # Text was staged and published; also print if no json_out?
        # No, text_out means write to file only.
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
