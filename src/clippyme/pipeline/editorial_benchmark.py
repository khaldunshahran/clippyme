"""Editorial benchmark foundation (Task 07).

Records what clips were produced, how many distinct useful moments humans
accept, and their editorial/framing/text defects. Human judgments are kept
separate from automated checks; unreviewed stays unknown.

Owns only this file. Does not modify discovery, models, prompts, renderers,
frontend, or existing test manifests. Reuses media_qa for objective checks
without changing its policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Six intended genre groups.
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
    import math
    return isinstance(v, (int, float)) and math.isfinite(v)


def _resolve_asset(asset_root: Path, rel: str) -> Path:
    """Resolve asset within root; reject traversal and missing."""
    p = (asset_root / rel).resolve()
    root = asset_root.resolve()
    if not str(p).startswith(str(root)):
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
    """Load and validate versioned input manifest."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise BenchmarkError(f"manifest JSON invalid: {e}")

    if data.get("version") != MANIFEST_VERSION:
        raise BenchmarkError(
            f"manifest version must be {MANIFEST_VERSION}, "
            f"got {data.get('version')}")

    asset_root = Path(p.parent) / data.get("asset_root", ".")
    if not asset_root.is_dir():
        raise BenchmarkError(f"asset_root not a directory: {asset_root}")

    # Validate sources.
    sources = {}
    for s in data.get("sources", []):
        sid = s.get("id")
        if not sid or sid in sources:
            raise BenchmarkError(f"duplicate or missing source id: {sid}")
        if s.get("genre") not in GENRES:
            raise BenchmarkError(
                f"source {sid}: genre must be one of {GENRES}")
        if not _is_finite_number(s.get("duration")) or s["duration"] <= 0:
            raise BenchmarkError(f"source {sid}: invalid duration")
        # Resolve and fingerprint.
        ap = _resolve_asset(asset_root, s.get("file", ""))
        fp = _fingerprint(ap)
        if "fingerprint" in s and s["fingerprint"] != fp:
            raise BenchmarkError(
                f"source {sid}: fingerprint mismatch (stale asset)")
        sources[sid] = {**s, "_path": str(ap), "_fingerprint": fp}

    # Validate runs and clips.
    runs = []
    clip_ids = set()
    for r in data.get("runs", []):
        rid = r.get("run_id")
        if not rid:
            raise BenchmarkError("run missing run_id")
        clips = []
        for c in r.get("clips", []):
            cid = c.get("clip_id")
            if not cid or cid in clip_ids:
                raise BenchmarkError(f"duplicate or missing clip_id: {cid}")
            clip_ids.add(cid)
            sid = c.get("source_id")
            if sid not in sources:
                raise BenchmarkError(
                    f"clip {cid}: unknown source_id {sid}")
            # Interval validation.
            st, en = c.get("start"), c.get("end")
            if not (_is_finite_number(st) and _is_finite_number(en)):
                raise BenchmarkError(f"clip {cid}: invalid interval")
            if not (0 <= st < en <= sources[sid]["duration"]):
                raise BenchmarkError(
                    f"clip {cid}: interval [{st},{en}) outside "
                    f"source duration {sources[sid]['duration']}")
            clips.append(c)
        runs.append({**r, "clips": clips})

    return {"version": MANIFEST_VERSION, "asset_root": str(asset_root),
            "sources": sources, "runs": runs}


def load_reviews(path: str | Path) -> dict[str, dict]:
    """Load human review records (JSON or CSV). Returns clip_id -> review."""
    p = Path(path)
    reviews = {}
    if p.suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        with open(p, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise BenchmarkError(f"reviews JSON invalid: {e}")
        rows = data.get("reviews", [])

    for r in rows:
        cid = r.get("clip_id")
        if not cid or cid in reviews:
            raise BenchmarkError(f"duplicate or missing clip_id: {cid}")
        # Normalize booleans; None/empty stays unknown.
        def _bool(v):
            if v in (True, "true", "True", "1", 1):
                return True
            if v in (False, "false", "False", "0", 0):
                return False
            return None  # unknown
        reviews[cid] = {
            "clip_id": cid,
            "opening_understandable": _bool(r.get("opening_understandable")),
            "payoff_delivered": _bool(r.get("payoff_delivered")),
            "ending_natural": _bool(r.get("ending_natural")),
            "framing_errors": r.get("framing_errors") or [],
            "hook_obstruction": _bool(r.get("hook_obstruction")),
            "caption_obstruction": _bool(r.get("caption_obstruction")),
            "accepted": _bool(r.get("accepted")),
            "notes": r.get("notes") or "",
        }
    return reviews


def _strict_json(obj: Any) -> str:
    """Emit strict finite JSON (no NaN/Infinity)."""
    import math

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


def run_benchmark(manifest: dict, reviews: dict[str, dict]) -> dict:
    """Compute benchmark stats. Unknown stays unknown."""
    sources = manifest["sources"]
    runs = manifest["runs"]

    # Per-run stats.
    run_stats = []
    for r in runs:
        clips = r["clips"]
        candidate = len(clips)
        reviewed = 0
        accepted = 0
        moments = set()
        moment_labeled = 0
        for c in clips:
            rev = reviews.get(c["clip_id"])
            if rev is None:
                continue  # unreviewed -> unknown
            # Count as reviewed if accepted is not None.
            if rev["accepted"] is not None:
                reviewed += 1
                if rev["accepted"]:
                    accepted += 1
                    mid = c.get("moment_id")
                    if mid:
                        moments.add(mid)
                        moment_labeled += 1
        # Acceptance denominator: reviewed with known acceptance.
        denom = reviewed
        rate = (accepted / denom) if denom else None
        # Distinct moments: only if labeled, else unknown.
        distinct = len(moments) if moment_labeled else None
        run_stats.append({
            "run_id": r["run_id"],
            "candidate_count": candidate,
            "reviewed_count": reviewed,
            "accepted_count": accepted,
            "acceptance_rate": rate,
            "distinct_accepted_moments": distinct,
            "moment_label_coverage": (
                moment_labeled / accepted if accepted else None),
        })

    # Per-genre aggregate.
    genre_stats = {}
    for g in GENRES:
        g_clips = []
        for r in runs:
            for c in r["clips"]:
                if sources[c["source_id"]]["genre"] == g:
                    g_clips.append(c)
        if not g_clips:
            continue
        reviewed = sum(
            1 for c in g_clips
            if reviews.get(c["clip_id"], {}).get("accepted") is not None)
        accepted = sum(
            1 for c in g_clips
            if reviews.get(c["clip_id"], {}).get("accepted") is True)
        moments = set(
            c["moment_id"] for c in g_clips
            if c.get("moment_id") and
            reviews.get(c["clip_id"], {}).get("accepted") is True)
        genre_stats[g] = {
            "candidate_count": len(g_clips),
            "reviewed_count": reviewed,
            "accepted_count": accepted,
            "acceptance_rate": (accepted / reviewed) if reviewed else None,
            "distinct_accepted_moments": len(moments) if moments else None,
        }

    # Source hours for accepted-per-hour.
    total_source_hours = sum(
        s["duration"] for s in sources.values()) / 3600.0
    total_accepted = sum(rs["accepted_count"] for rs in run_stats)
    # Paired runs by source ID.
    paired = {}
    for r in runs:
        for c in r["clips"]:
            sid = c["source_id"]
            paired.setdefault(sid, {}).setdefault(
                r["run_id"], []).append(c["clip_id"])

    return {
        "version": MANIFEST_VERSION,
        "run_stats": run_stats,
        "genre_stats": genre_stats,
        "total_source_hours": round(total_source_hours, 3),
        "accepted_per_source_hour": (
            round(total_accepted / total_source_hours, 2)
            if total_source_hours else None),
        "paired_sources": paired,
        "note": ("Real-footage benchmark has not run; "
                 "synthetic fixtures only." if all(
                     s.get("synthetic") for s in sources.values())
                 else "Includes real footage."),
    }


def format_report(stats: dict) -> str:
    """Human-readable report."""
    lines = ["Editorial Benchmark Report", "=" * 30, ""]
    lines.append(f"Source hours: {stats['total_source_hours']}")
    aph = stats["accepted_per_source_hour"]
    lines.append(f"Accepted per source hour: {aph if aph is not None else 'unknown'}")
    lines.append("")
    lines.append("Per run:")
    for rs in stats["run_stats"]:
        rate = rs["acceptance_rate"]
        rate_s = f"{rate:.1%}" if rate is not None else "unknown"
        dm = rs["distinct_accepted_moments"]
        dm_s = str(dm) if dm is not None else "unknown (no labels)"
        lines.append(
            f"  {rs['run_id']}: {rs['accepted_count']}/{rs['reviewed_count']} "
            f"accepted ({rate_s}), distinct moments: {dm_s}")
    lines.append("")
    lines.append("Per genre:")
    for g, gs in stats["genre_stats"].items():
        rate = gs["acceptance_rate"]
        rate_s = f"{rate:.1%}" if rate is not None else "unknown"
        lines.append(
            f"  {g}: {gs['accepted_count']}/{gs['reviewed_count']} "
            f"({rate_s})")
    lines.append("")
    lines.append(stats["note"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Editorial benchmark: human acceptance of clips.")
    ap.add_argument("--manifest", required=True, help="Input manifest JSON")
    ap.add_argument("--reviews", required=True, help="Reviews JSON or CSV")
    ap.add_argument("--json-out", help="Write strict JSON report")
    ap.add_argument("--text-out", help="Write readable report")
    args = ap.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        reviews = load_reviews(args.reviews)
        stats = run_benchmark(manifest, reviews)
    except BenchmarkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    json_str = _strict_json(stats)
    text_str = format_report(stats)

    if args.json_out:
        Path(args.json_out).write_text(json_str, encoding="utf-8")
    else:
        print(json_str)
    if args.text_out:
        Path(args.text_out).write_text(text_str, encoding="utf-8")
    elif not args.json_out:
        # If JSON went to stdout, also show text.
        print("\n" + text_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
