"""Editorial benchmark foundation (Task 07, rev2).

Records what clips were produced, how many distinct useful moments humans
accept, and their editorial/framing/text defects. Human judgments are kept
separate from automated checks; unreviewed stays unknown, invalid is an
error.

Owns only this file. Does not modify discovery, models, prompts, renderers,
frontend, or existing test manifests. Reuses media_qa.inspect_clip for
objective media checks without changing its policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
    """Clear failure for malformed records (never silently counted)."""


def _is_finite_number(v: Any) -> bool:
    # Strict: bool is not numeric.
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float)) and math.isfinite(v)


def _resolve_asset(asset_root: Path, rel: str) -> Path:
    """Resolve asset with real path containment (not string prefix)."""
    if not rel or not isinstance(rel, str):
        raise BenchmarkError(f"invalid asset path: {rel!r}")
    # Resolve both (handles symlinks, .., absolute).
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


def _check_output_collision(path: Path, manifest: dict):
    """Reject output paths colliding with inputs or projects."""
    p = Path(path).resolve()
    protected = set()
    root = Path(manifest["asset_root"]).resolve()
    protected.add(root)
    for s in manifest["sources"].values():
        protected.add(Path(s["_path"]).resolve())
        # Clip assets.
    for r in manifest["runs"]:
        for c in r["clips"]:
            if "_clip_path" in c:
                protected.add(Path(c["_clip_path"]).resolve())
    # Also protect manifest/review files themselves (caller's responsibility
    # to pass them; we check the output path isn't inside asset root).
    if str(p).startswith(str(root)):
        # Allow output inside asset root only if not colliding with assets.
        # Actually, safest: reject any output inside asset root.
        raise BenchmarkError(
            f"output path inside asset root (collision risk): {path}")
    for prot in protected:
        if p == prot:
            raise BenchmarkError(f"output collides with input: {path}")


def load_manifest(path: str | Path) -> dict:
    """Load and validate versioned input manifest."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise BenchmarkError(f"manifest JSON invalid: {e}")

    if not isinstance(data, dict):
        raise BenchmarkError(
            f"manifest root must be object, got {type(data).__name__}")
    if data.get("version") != MANIFEST_VERSION:
        raise BenchmarkError(
            f"manifest version must be {MANIFEST_VERSION}")

    asset_root = Path(p.parent) / data.get("asset_root", ".")
    if not asset_root.is_dir():
        raise BenchmarkError("asset_root not a directory")

    # Sources.
    sources = {}
    for s in data.get("sources", []):
        if not isinstance(s, dict):
            raise BenchmarkError("source record must be object")
        sid = s.get("id")
        if not isinstance(sid, str) or not sid or sid in sources:
            raise BenchmarkError(f"duplicate or invalid source id: {sid!r}")
        genre = s.get("genre")
        if genre not in GENRES:
            raise BenchmarkError(f"source {sid}: invalid genre {genre!r}")
        dur = s.get("duration")
        if not _is_finite_number(dur) or dur <= 0:
            raise BenchmarkError(f"source {sid}: invalid duration")
        ap = _resolve_asset(asset_root, s.get("file", ""))
        fp = _fingerprint(ap)
        if "fingerprint" in s and s["fingerprint"] != fp:
            raise BenchmarkError(f"source {sid}: fingerprint mismatch")
        sources[sid] = {
            "id": sid, "genre": genre, "duration": float(dur),
            "file": s.get("file"), "_path": str(ap), "fingerprint": fp,
            "synthetic": bool(s.get("synthetic", False)),
        }

    # Runs (duplicate run IDs rejected).
    runs = []
    run_ids = set()
    clip_ids = set()
    for r in data.get("runs", []):
        if not isinstance(r, dict):
            raise BenchmarkError("run record must be object")
        rid = r.get("run_id")
        if not isinstance(rid, str) or not rid or rid in run_ids:
            raise BenchmarkError(f"duplicate or invalid run_id: {rid!r}")
        run_ids.add(rid)
        clips = []
        for c in r.get("clips", []):
            if not isinstance(c, dict):
                raise BenchmarkError("clip record must be object")
            cid = c.get("clip_id")
            if not isinstance(cid, str) or not cid or cid in clip_ids:
                raise BenchmarkError(
                    f"duplicate or invalid clip_id: {cid!r}")
            clip_ids.add(cid)
            sid = c.get("source_id")
            if sid not in sources:
                raise BenchmarkError(f"clip {cid}: unknown source {sid!r}")
            st, en = c.get("start"), c.get("end")
            if not (_is_finite_number(st) and _is_finite_number(en)):
                raise BenchmarkError(f"clip {cid}: invalid interval")
            if not (0 <= st < en <= sources[sid]["duration"]):
                raise BenchmarkError(f"clip {cid}: interval out of range")
            # Clip asset (rendered file) is required.
            clip_file = c.get("file")
            if not clip_file:
                raise BenchmarkError(f"clip {cid}: missing file")
            cp = _resolve_asset(asset_root, clip_file)
            cfp = _fingerprint(cp)
            if "fingerprint" in c and c["fingerprint"] != cfp:
                raise BenchmarkError(f"clip {cid}: fingerprint mismatch")
            mid = c.get("moment_id")
            if mid is not None and (not isinstance(mid, str) or not mid):
                raise BenchmarkError(f"clip {cid}: invalid moment_id")
            clips.append({
                "clip_id": cid, "source_id": sid,
                "start": float(st), "end": float(en),
                "moment_id": mid, "file": clip_file,
                "_clip_path": str(cp), "fingerprint": cfp,
            })
        runs.append({
            "run_id": rid, "commit": r.get("commit", ""),
            "settings": r.get("settings", {}),
            "clips": clips,
        })

    return {"version": MANIFEST_VERSION, "asset_root": str(asset_root),
            "sources": sources, "runs": runs}


def _parse_bool_strict(v: Any, field: str, cid: str) -> bool | None:
    """Strict boolean parsing. Unknown is None/blank/omitted; invalid errors."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in ("true", "1", "yes"):
            return True
        if lv in ("false", "0", "no"):
            return False
    raise BenchmarkError(
        f"clip {cid}: invalid boolean for {field}: {v!r}")


def load_reviews(path: str | Path) -> dict[str, dict]:
    """Load human reviews. Unknown review IDs are validated later."""
    p = Path(path)
    if p.suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Parse framing_errors as comma-separated list.
        for r in rows:
            fe = r.get("framing_errors", "")
            r["framing_errors"] = (
                [x.strip() for x in fe.split(",") if x.strip()]
                if fe else [])
    else:
        with open(p, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise BenchmarkError(f"reviews JSON invalid: {e}")
        if not isinstance(data, dict):
            raise BenchmarkError("reviews root must be object")
        rows = data.get("reviews", [])

    reviews = {}
    for r in rows:
        if not isinstance(r, dict):
            raise BenchmarkError("review record must be object")
        cid = r.get("clip_id")
        if not isinstance(cid, str) or not cid or cid in reviews:
            raise BenchmarkError(f"duplicate or invalid clip_id: {cid!r}")
        fe = r.get("framing_errors", [])
        if isinstance(fe, str):
            # Should have been parsed for CSV; for JSON, reject opaque string.
            raise BenchmarkError(
                f"clip {cid}: framing_errors must be list, not string")
        if not isinstance(fe, list):
            raise BenchmarkError(f"clip {cid}: invalid framing_errors")
        notes = r.get("notes", "")
        if not isinstance(notes, str):
            raise BenchmarkError(f"clip {cid}: notes must be string")
        reviews[cid] = {
            "clip_id": cid,
            "opening_understandable": _parse_bool_strict(
                r.get("opening_understandable"), "opening_understandable", cid),
            "payoff_delivered": _parse_bool_strict(
                r.get("payoff_delivered"), "payoff_delivered", cid),
            "ending_natural": _parse_bool_strict(
                r.get("ending_natural"), "ending_natural", cid),
            "framing_errors": fe,
            "hook_obstruction": _parse_bool_strict(
                r.get("hook_obstruction"), "hook_obstruction", cid),
            "caption_obstruction": _parse_bool_strict(
                r.get("caption_obstruction"), "caption_obstruction", cid),
            "accepted": _parse_bool_strict(r.get("accepted"), "accepted", cid),
            "notes": notes,
        }
    return reviews


def _strict_json(obj: Any) -> str:
    def _check(o):
        if isinstance(o, float) and not math.isfinite(o):
            raise BenchmarkError("non-finite number in output")
        if isinstance(o, dict):
            for v in o.values():
                _check(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _check(v)
    _check(obj)
    return json.dumps(obj, indent=2, sort_keys=True)


def _media_check(path: str) -> dict:
    """Objective media check via media_qa (separate from human ratings)."""
    try:
        from clippyme.pipeline.media_qa import inspect_clip
        report = inspect_clip(path)
        return {
            "valid": bool(report.get("ok", False)),
            "duration": report.get("duration"),
            "errors": report.get("errors", []),
        }
    except Exception as e:
        return {"valid": False, "duration": None,
                "errors": [f"media_qa failed: {e}"]}


def run_benchmark(manifest: dict, reviews: dict[str, dict],
                  check_media: bool = False) -> dict:
    """Compute stats. Orphan reviews rejected; unknown stays unknown."""
    sources = manifest["sources"]
    runs = manifest["runs"]

    # Reject unknown review IDs.
    all_cids = {c["clip_id"] for r in runs for c in r["clips"]}
    for cid in reviews:
        if cid not in all_cids:
            raise BenchmarkError(f"review for unknown clip_id: {cid!r}")

    # Media checks (objective, separate).
    media_results = {}
    if check_media:
        for sid, s in sources.items():
            media_results[f"source:{sid}"] = _media_check(s["_path"])
        for r in runs:
            for c in r["clips"]:
                media_results[f"clip:{c['clip_id']}"] = _media_check(
                    c["_clip_path"])

    run_stats = []
    for r in runs:
        clips = r["clips"]
        # Source membership for this run.
        run_sources = {c["source_id"] for c in clips}
        run_source_hours = sum(
            sources[sid]["duration"] for sid in run_sources) / 3600.0

        candidate = len(clips)
        reviewed = 0
        accepted = 0
        # Moments scoped by (source_id, moment_id).
        moments = set()
        labeled_accepted = 0
        # Human judgment tallies.
        opening_known = opening_ok = 0
        payoff_known = payoff_ok = 0
        ending_known = ending_ok = 0
        framing_defects = 0

        for c in clips:
            rev = reviews.get(c["clip_id"])
            if rev is None:
                continue
            # Robust to minimal dicts (probes) vs normalized (load_reviews).
            acc = rev.get("accepted")
            if acc is not None:
                reviewed += 1
                if acc:
                    accepted += 1
                    mid = c.get("moment_id")
                    if mid:
                        moments.add((c["source_id"], mid))
                        labeled_accepted += 1
            # Judgment coverage (known denominators).
            ou = rev.get("opening_understandable")
            if ou is not None:
                opening_known += 1
                if ou:
                    opening_ok += 1
            pd = rev.get("payoff_delivered")
            if pd is not None:
                payoff_known += 1
                if pd:
                    payoff_ok += 1
            en = rev.get("ending_natural")
            if en is not None:
                ending_known += 1
                if en:
                    ending_ok += 1
            if rev.get("framing_errors"):
                framing_defects += 1

        # Distinct moments: unknown if partial labels.
        if accepted == 0:
            distinct = 0 if reviewed == candidate else None
            # Known zero only if fully reviewed.
        elif labeled_accepted == accepted:
            distinct = len(moments)
        else:
            distinct = None  # partial labels -> unknown

        rate = (accepted / reviewed) if reviewed else None
        aph = (accepted / run_source_hours) if run_source_hours else None

        run_stats.append({
            "run_id": r["run_id"],
            "commit": r["commit"],
            "settings": r["settings"],
            "candidate_count": candidate,
            "reviewed_count": reviewed,
            "accepted_count": accepted,
            "acceptance_rate": rate,
            "distinct_accepted_moments": distinct,
            "moment_label_coverage": (
                labeled_accepted / accepted if accepted else None),
            "source_hours": round(run_source_hours, 3),
            "accepted_per_source_hour": (
                round(aph, 2) if aph is not None else None),
            "review_completeness": (
                reviewed / candidate if candidate else None),
            "judgments": {
                "opening": {"known": opening_known, "ok": opening_ok},
                "payoff": {"known": payoff_known, "ok": payoff_ok},
                "ending": {"known": ending_known, "ok": ending_ok},
                "framing_defects": framing_defects,
            },
            "source_fingerprints": {
                sid: sources[sid]["fingerprint"] for sid in run_sources},
        })

    # Per-genre (all six, even missing).
    genre_stats = {}
    for g in GENRES:
        g_clips = []
        for r in runs:
            for c in r["clips"]:
                if sources[c["source_id"]]["genre"] == g:
                    g_clips.append((r["run_id"], c))
        reviewed = sum(
            1 for _, c in g_clips
            if reviews.get(c["clip_id"], {}).get("accepted") is not None)
        accepted = sum(
            1 for _, c in g_clips
            if reviews.get(c["clip_id"], {}).get("accepted") is True)
        moments = set(
            (c["source_id"], c["moment_id"]) for _, c in g_clips
            if c.get("moment_id") and
            reviews.get(c["clip_id"], {}).get("accepted") is True)
        # Per-run/per-genre.
        by_run = {}
        for rid, c in g_clips:
            by_run.setdefault(rid, {"candidate": 0, "accepted": 0})
            by_run[rid]["candidate"] += 1
            if reviews.get(c["clip_id"], {}).get("accepted") is True:
                by_run[rid]["accepted"] += 1
        genre_stats[g] = {
            "candidate_count": len(g_clips),
            "reviewed_count": reviewed,
            "accepted_count": accepted,
            "acceptance_rate": (accepted / reviewed) if reviewed else None,
            "distinct_accepted_moments": len(moments) if moments else None,
            "by_run": by_run,
        }

    # Paired comparison by source.
    paired = {}
    for sid in sources:
        runs_for_source = {}
        for r in runs:
            cids = [c["clip_id"] for c in r["clips"]
                    if c["source_id"] == sid]
            if cids:
                runs_for_source[r["run_id"]] = {
                    "clip_ids": cids,
                    "commit": r["commit"],
                    "fingerprint": sources[sid]["fingerprint"],
                }
        if runs_for_source:
            # Check fingerprint compatibility.
            fps = {v["fingerprint"] for v in runs_for_source.values()}
            paired[sid] = {
                "runs": runs_for_source,
                "fingerprint_compatible": len(fps) == 1,
                "missing_runs": [
                    r["run_id"] for r in runs
                    if r["run_id"] not in runs_for_source],
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
        cov = rs["moment_label_coverage"]
        cov_s = f"{cov:.0%}" if cov is not None else "n/a"
        comp = rs["review_completeness"]
        comp_s = f"{comp:.0%}" if comp is not None else "unknown"
        aph = rs["accepted_per_source_hour"]
        aph_s = str(aph) if aph is not None else "unknown"
        lines.append(f"Run {rs['run_id']} ({rs['commit']}):")
        lines.append(f"  Candidates: {rs['candidate_count']}, "
                     f"Reviewed: {rs['reviewed_count']} ({comp_s}), "
                     f"Accepted: {rs['accepted_count']} ({rate_s})")
        lines.append(f"  Distinct moments: {dm_s} (label coverage {cov_s})")
        lines.append(f"  Source hours: {rs['source_hours']}, "
                     f"Accepted/hour: {aph_s}")
        j = rs["judgments"]
        lines.append(f"  Judgments: opening {j['opening']['ok']}/"
                     f"{j['opening']['known']}, payoff {j['payoff']['ok']}/"
                     f"{j['payoff']['known']}, ending {j['ending']['ok']}/"
                     f"{j['ending']['known']}, framing defects: "
                     f"{j['framing_defects']}")
    lines.append("")
    lines.append("Per genre (all six):")
    for g, gs in stats["genre_stats"].items():
        rate = gs["acceptance_rate"]
        rate_s = f"{rate:.1%}" if rate is not None else "no data"
        lines.append(f"  {g}: {gs['accepted_count']}/"
                     f"{gs['reviewed_count']} ({rate_s}), "
                     f"candidates: {gs['candidate_count']}")
    lines.append("")
    lines.append("Paired sources:")
    for sid, p in stats["paired_sources"].items():
        runs = ", ".join(p["runs"].keys())
        compat = "compatible" if p["fingerprint_compatible"] else "MISMATCH"
        missing = f", missing: {p['missing_runs']}" if p["missing_runs"] else ""
        lines.append(f"  {sid}: [{runs}] fingerprints {compat}{missing}")
    if stats["media_checks"]:
        lines.append("")
        lines.append("Media checks (objective, separate from human):")
        for k, v in stats["media_checks"].items():
            status = "ok" if v["valid"] else "FAIL"
            lines.append(f"  {k}: {status}")
    lines.append("")
    lines.append(stats["note"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Editorial benchmark: human acceptance of clips.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--json-out", help="Write strict JSON report")
    ap.add_argument("--text-out", help="Write readable report")
    ap.add_argument("--check-media", action="store_true",
                    help="Run media_qa checks (objective)")
    args = ap.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        reviews = load_reviews(args.reviews)
        stats = run_benchmark(manifest, reviews,
                              check_media=args.check_media)
        json_str = _strict_json(stats)
        text_str = format_report(stats)
        # Collision checks before writing.
        if args.json_out:
            _check_output_collision(args.json_out, manifest)
        if args.text_out:
            _check_output_collision(args.text_out, manifest)
    except BenchmarkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json_out:
        Path(args.json_out).write_text(json_str, encoding="utf-8")
    else:
        print(json_str)
    if args.text_out:
        Path(args.text_out).write_text(text_str, encoding="utf-8")
    elif not args.json_out:
        print("\n" + text_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
