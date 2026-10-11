"""Task 18 — typed Hyperframes finishing compiler (contract 4).

``compile_hyperframes_composition`` turns a validated ClipProject into a
deterministic, self-contained HTML composition:

- local immutable assets/fonts (copied per compile, sha256-recorded),
- escaped text (no markup/code from chat ever reaches the composition),
- bounded typed values (closed enums, numeric clamps),
- the PRODUCTION time mapping (``_map_words_to_timeline``) — no independent
  index/window heuristics,
- half-open caption/hook intervals, delayed seekable animation, render
  aspect/dimensions/FPS, word IDs + retained-word edits, timeline
  offsets/segment joins.

Preview and export consume the SAME compiled composition and clock: the
manifest digest is the preview/export identity.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field

from . import templates
from .cleanbase import CleanBaseError, CleanBaseUnavailable, resolve_clean_base

logger = logging.getLogger(__name__)

ENGINE_VERSION = "0.8.145"
COMPILER_VERSION = "nugget-finishing/1"

# Caption font sizes as fractions of render height (bounded, typed).
_CAPTION_SIZE_FRAC = {"small": 0.042, "medium": 0.052, "large": 0.064}
_HOOK_SIZE_FRAC = 0.058

# Phrase grouping bounds (deterministic presentation grouping; the underlying
# word timings always come from the production mapping).
_PHRASE_MAX_WORDS = 8
_PHRASE_MAX_SPAN = 4.0


class CompileError(Exception):
    """Compiler gate failure: invalid input, unsupported state, or bad asset."""


class FinishingDisabled(CompileError):
    """The project did not explicitly opt into the Hyperframes renderer."""


@dataclass(frozen=True)
class FinishingParams:
    """Typed, bounded finishing parameters (closed enums only)."""

    template_id: str
    caption_size: str = "medium"
    caption_position: str = "lower-third"

    def __post_init__(self):
        spec = templates.TEMPLATES.get(self.template_id)
        if spec is None:
            raise CompileError(
                f"unknown template {self.template_id!r}; "
                f"supported: {', '.join(templates.TEMPLATE_IDS)}"
            )
        for name, value in (("caption_size", self.caption_size),
                            ("caption_position", self.caption_position)):
            allowed = spec.param_enums.get(name, ())
            if value not in allowed:
                raise CompileError(
                    f"invalid {name} {value!r}; allowed: {', '.join(allowed)}"
                )

    @classmethod
    def from_dict(cls, template_id: str, params: dict | None) -> "FinishingParams":
        params = dict(params or {})
        unknown = set(params) - {"caption_size", "caption_position"}
        if unknown:
            raise CompileError(f"unknown finishing params: {sorted(unknown)}")
        try:
            return cls(
                template_id=template_id,
                caption_size=params.get("caption_size", "medium"),
                caption_position=params.get("caption_position", "lower-third"),
            )
        except TypeError as exc:
            raise CompileError(f"invalid finishing params: {exc}") from exc


@dataclass(frozen=True)
class CompiledComposition:
    """Result of a successful compile."""

    compile_id: str
    composition_dir: str  # abs path: index.html + base.mp4 + fonts
    # NOTE: the entry file MUST be index.html — the engine navigates to
    # {serverUrl}/index.html (hardcoded in frameCapture.js).
    html_path: str
    base_path: str
    digest: str  # sha256 of the canonical manifest: preview/export identity
    manifest: dict
    duration: float
    width: int
    height: int
    fps: int
    frames: int
    phrases: list = field(default_factory=list)
    hook: dict | None = None


def project_digest(project) -> str:
    """Canonical digest of the validated project (stable keying)."""
    canonical = project.model_dump_json()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def renderer_versions(worker_dir: str | None = None) -> dict:
    """Report pinned renderer versions; chromium None when not installed."""
    info = {"engine": ENGINE_VERSION, "compiler": COMPILER_VERSION,
            "chromium": None}
    try:
        base = worker_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "hyperframes-worker")
        cache = os.path.join(os.path.abspath(base), ".puppeteer-cache",
                             "chrome-headless-shell")
        if os.path.isdir(cache):
            versions = sorted(os.listdir(cache))
            if versions:
                info["chromium"] = versions[-1]
    except OSError:
        pass
    return info


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _configured_music_file(project) -> str | None:
    """Return the configured music file, or None for source-audio projects.

    ``AudioTrack.file`` is the "music file" field (null = no music); a
    legacy ``music`` attribute is honored too. Any non-empty value means the
    accepted pipeline would mix music the pilot cannot reproduce.
    """
    legacy = getattr(project, "music", None)
    if legacy:
        f = legacy if isinstance(legacy, str) else getattr(legacy, "file", None)
        if f and str(f).strip():
            return str(f)
    for track in getattr(project, "audio", None) or []:
        f = getattr(track, "file", None)
        if f and str(f).strip():
            return str(f)
    return None


def _group_phrases(mapped_words: list[dict]) -> list[dict]:
    """Deterministically group word cues into phrase cues for display.

    Grouping is presentation-only: every word keeps its production-mapped
    [start, end); a phrase spans [first.start, last.end).
    """
    phrases: list[dict] = []
    cur: list[dict] = []

    def flush():
        if cur:
            phrases.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "words": list(cur),
            })
            cur.clear()

    for w in mapped_words:
        if (cur and (len(cur) >= _PHRASE_MAX_WORDS
                     or w["start"] - cur[0]["start"] > _PHRASE_MAX_SPAN
                     or cur[-1]["word"].rstrip().endswith((".", "?", "!")))):
            flush()
        cur.append(w)
    flush()
    return phrases


def _select_hook(project, template_id: str, duration: float) -> dict | None:
    """Select the timed hook for the composition (half-open [start, end)).

    Preference: the grounded Task 15 opening selection when present, else the
    first hook overlay. Returns None when no hook text/timing is usable.
    """
    text, start, end = None, None, None
    if template_id == "timed-opening":
        opening = getattr(project, "opening", None)
        if opening is not None and opening.status in ("selected", "user_authored"):
            text = (opening.text or "").strip()
            start = opening.display_start
            end = opening.display_end
    if not text:
        for o in project.overlays or []:
            if o.type == "hook" and (o.text or "").strip():
                text = o.text.strip()
                start, end = o.start, o.end
                break
    if not text:
        return None
    try:
        s, e = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if not (s < e):
        return None
    s = max(0.0, s)
    e = min(duration, e)
    if not (s < e):
        return None
    return {"text": text, "start": round(s, 3), "end": round(e, 3)}


def _font_face_css(fonts: list[dict]) -> str:
    lines = []
    for f in fonts:
        lines.append(
            "@font-face { font-family:'%s'; src:url('%s'); font-display:block; }" % (
                f["family"], f["file"]))
    return "\n".join(lines)


# Task 18 review-02: preview postMessage bridge (contract 2 backend).
# The composition listens for {hf:1, type:'seek'|'play'|'pause', t?} and
# drives its own window.__hf.seek / the base <video> element (play() with
# audio, pause()). It posts {hf:1, type:'ready'} to the parent once __hf is
# installed and {hf:1, type:'timeupdate', t} during playback. Incoming
# messages are bound to ev.source === window.parent (shape-only checks
# accept unrelated windows); every message echoes the per-instance hfiid
# from the iframe URL so concurrently mounted previews cannot cross-update.
# Graphics are driven CONTINUOUSLY from the actual media clock: __hf.seek
# only positions graphics, so without this the video advances while
# captions/hooks stay frozen. Incoming messages carry no secrets (numeric t
# only). The worker engine keeps using window.__hf directly (untouched).
_BRIDGE_JS = """
(function () {
  var video = document.getElementById('hf-base');
  var D = 0;
  try { D = JSON.parse(document.getElementById('hf-data').textContent).duration; } catch (e) { D = 0; }
  var IID = null;
  try {
    var mm = /[?&]hfiid=([A-Za-z0-9_-]{1,64})/.exec(location.search || '');
    if (mm) { IID = mm[1]; }
  } catch (e) {}
  function post(msg) {
    try { window.parent.postMessage(msg, '*'); } catch (e) {}
  }
  function baseMsg(type, extra) {
    var msg = { hf: 1, type: type };
    if (IID) { msg.hfiid = IID; }
    if (extra) {
      for (var k in extra) {
        if (Object.prototype.hasOwnProperty.call(extra, k)) { msg[k] = extra[k]; }
      }
    }
    return msg;
  }
  function syncGraphics() {
    if (!video) { return; }
    try {
      if (window.__hf && typeof window.__hf.seek === 'function') {
        window.__hf.seek(video.currentTime);
      }
    } catch (e) {}
  }
  var tries = 0;
  (function waitReady() {
    if (window.__hf && typeof window.__hf.seek === 'function') {
      syncGraphics();
      post(baseMsg('ready'));
      return;
    }
    if (++tries < 200) { setTimeout(waitReady, 50); }
  })();
  function validT(v) {
    return typeof v === 'number' && isFinite(v) && v >= 0;
  }
  window.addEventListener('message', function (ev) {
    try { if (ev.source !== window.parent) { return; } } catch (e) { return; }
    var m = ev.data;
    if (!m || typeof m !== 'object' || m.hf !== 1) { return; }
    if (IID && m.hfiid !== IID) { return; }
    var t = validT(m.t) ? m.t : null;
    if (m.type === 'seek') {
      if (t === null) { return; }
      var ct = (D > 0) ? Math.min(Math.max(t, 0), D) : Math.max(t, 0);
      if (video) { try { video.currentTime = ct; } catch (e) {} }
      syncGraphics();
    } else if (m.type === 'play') {
      if (video) {
        try { video.muted = false; } catch (e) {}
        try {
          var pr = video.play();
          if (pr && typeof pr.catch === 'function') { pr.catch(function () {}); }
        } catch (e) {}
      }
    } else if (m.type === 'pause') {
      if (video) { try { video.pause(); } catch (e) {} }
      syncGraphics();
    }
  });
  if (video) {
    video.addEventListener('timeupdate', function () {
      syncGraphics();
      post(baseMsg('timeupdate', { t: video.currentTime }));
    });
    video.addEventListener('seeked', syncGraphics);
    video.addEventListener('play', syncGraphics);
    video.addEventListener('pause', syncGraphics);
    video.addEventListener('ended', syncGraphics);
  }
})();
"""


def _build_html(*, spec, params: FinishingParams, width: int, height: int,
                duration: float, phrases: list[dict], hook: dict | None,
                fonts: list[dict]) -> str:
    """Assemble the deterministic composition HTML (escaped text only)."""
    cap_px = {
        "sz_s": max(10, round(height * _CAPTION_SIZE_FRAC["small"])),
        "sz_m": max(10, round(height * _CAPTION_SIZE_FRAC["medium"])),
        "sz_l": max(10, round(height * _CAPTION_SIZE_FRAC["large"])),
    }
    hook_px = max(12, round(height * _HOOK_SIZE_FRAC))

    css = spec.css
    for token, val in (("__W__", str(width)), ("__H__", str(height)),
                       ("__CAPFONT__", spec.font_family),
                       ("__HOOKFONT__", spec.hook_font_family or spec.font_family),
                       ("__SZ_S__", str(cap_px["sz_s"])),
                       ("__SZ_M__", str(cap_px["sz_m"])),
                       ("__SZ_L__", str(cap_px["sz_l"])),
                       ("__HOOKSZ__", str(hook_px))):
        css = css.replace(token, val)

    size_cls = {"small": "sz-small", "medium": "sz-medium", "large": "sz-large"}[
        params.caption_size]
    pos_cls = {"lower-third": "pos-lower-third", "middle": "pos-middle"}[
        params.caption_position]

    if spec.id == "emphasized-word":
        words_html = []
        for ph in phrases:
            for w in ph["words"]:
                words_html.append(
                    '<span class="w %s" data-start="%.3f" data-end="%.3f">%s</span>' % (
                        size_cls, w["start"], w["end"], _escape(w["word"])))
        caps_html = '<div id="hf-caps" class="%s">%s</div>' % (
            pos_cls, " ".join(words_html))
    else:
        phrase_html = []
        for ph in phrases:
            words_html = " ".join(
                '<span class="w" data-start="%.3f" data-end="%.3f">%s</span>' % (
                    w["start"], w["end"], _escape(w["word"]))
                for w in ph["words"])
            phrase_html.append(
                '<div class="hf-phrase %s" data-start="%.3f" data-end="%.3f">%s</div>' % (
                    size_cls, ph["start"], ph["end"], words_html))
        caps_html = '<div id="hf-caps" class="%s">%s</div>' % (
            pos_cls, "".join(phrase_html))

    hook_html = ""
    if hook is not None:
        hook_html = (
            '<div id="hf-hook" data-start="%.3f" data-end="%.3f">%s</div>' % (
                hook["start"], hook["end"], _escape(hook["text"])))

    data_json = json.dumps({"duration": duration, "template": spec.id,
                            "template_version": spec.version},
                           separators=(",", ":"))

    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
%s
%s
</style>
</head>
<body>
<div id="stage" data-composition-id="nugget-hf" data-no-timeline data-width="%d" data-height="%d">
  <video id="hf-base" class="clip" data-start="0" data-duration="%.3f" data-track-index="0" src="base.mp4" muted playsinline></video>
  %s
  %s
</div>
<script type="application/json" id="hf-data">%s</script>
<script>
%s
</script>
<script>
%s
</script>
</body>
</html>
""" % (_font_face_css(fonts), css, width, height, duration,
       caps_html, hook_html, data_json, spec.seek_js, _BRIDGE_JS)


async def compile_hyperframes_composition(
    project,
    *,
    template_id: str,
    params: dict | None,
    job_dir: str,
    clip_index: int,
    acknowledged_version: int,
    renderer_enabled: bool,
    worker_dir: str | None = None,
) -> CompiledComposition:
    """Compile a validated ClipProject into a deterministic composition.

    Gates (each raises CompileError / subclass):
    - renderer explicitly enabled (default OFF) and project.finishing selects
      the hyperframes renderer,
    - known template + bounded typed params,
    - acknowledged_version matches project.version (stale -> reject),
    - no configured music (the pilot renders source audio only; configured
      music fails clearly instead of being silently dropped),
    - verified clean base available (framing trajectories and reframed_base
      lineage fail clearly),
    - local immutable fonts present (sha256 recorded),
    - composition duration matches the materialized base clock.
    """
    # 1. Explicit opt-in: default OFF, project must select the renderer.
    finishing = getattr(project, "finishing", None)
    finishing_renderer = getattr(finishing, "renderer", None) if finishing else None
    if not renderer_enabled or finishing_renderer != "hyperframes":
        raise FinishingDisabled(
            "hyperframes renderer is not enabled for this project "
            "(default OFF; apply a Hyperframes template to opt in)")

    # 2. Template + typed params.
    fp = FinishingParams.from_dict(template_id, params)
    spec = templates.get_template(fp.template_id)

    # 3. Version identity: compile only the acknowledged snapshot.
    if int(project.version) != int(acknowledged_version):
        raise CompileError(
            f"stale project version: acknowledged {acknowledged_version}, "
            f"project is v{project.version}")

    # 3b. Configured music: the pilot renders the source-audio base only and
    # would silently drop a configured music track (the accepted pipeline
    # mixes project.audio music in _mix_music). Fail clearly instead.
    music_file = _configured_music_file(project)
    if music_file:
        raise CleanBaseUnavailable(
            f"project has configured music ({music_file}); the Hyperframes "
            "pilot renders source audio only — use the legacy render path"
        )

    # 4. Verified clean base (contract 3).
    base = await resolve_clean_base(project, job_dir, clip_index)

    # 5. Production time mapping (contract 4): word IDs, retained-word edits,
    #    timeline offsets/segment joins, removed words dropped.
    from clippyme.domain.project_render import _map_words_to_timeline
    mapped_words, timeline_duration = _map_words_to_timeline(project)

    duration = float(base["duration"])
    if abs(duration - float(timeline_duration)) > 0.05 + 1.0 / max(1, project.render.fps):
        raise CompileError(
            f"base clock {duration:.3f}s disagrees with mapped timeline "
            f"{timeline_duration:.3f}s")
    fps = int(project.render.fps)
    width, height = int(project.render.width), int(project.render.height)
    frames = int(round(duration * fps))
    if frames < 1:
        raise CompileError("composition has no frames")

    phrases = _group_phrases(mapped_words)
    hook = _select_hook(project, fp.template_id, duration)
    if fp.template_id == "timed-opening" and hook is None:
        raise CompileError(
            "timed-opening needs a hook: no opening selection or hook overlay "
            "with usable text/timing on this project")

    # 6. Local immutable fonts.
    fonts = []
    repo_fonts = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "fonts")
    for family, fname in ((spec.font_family, spec.font_file),
                          (spec.hook_font_family, spec.hook_font_file)):
        if not family or not fname:
            continue
        if any(f["family"] == family for f in fonts):
            continue
        src = os.path.join(os.path.abspath(repo_fonts), fname)
        if not os.path.isfile(src):
            raise CompileError(f"template font missing: {fname}")
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        fonts.append({"family": family, "file": fname, "sha256": h.hexdigest(),
                      "src": src})

    # 7. Stage the immutable composition dir.
    compile_id = uuid.uuid4().hex[:16]
    comp_dir = os.path.join(os.path.abspath(job_dir), ".hf", compile_id)
    os.makedirs(comp_dir, exist_ok=False)
    shutil.copy2(base["path"], os.path.join(comp_dir, "base.mp4"))
    for f in fonts:
        shutil.copy2(f["src"], os.path.join(comp_dir, f["file"]))
    html_text = _build_html(
        spec=spec, params=fp, width=width, height=height, duration=duration,
        phrases=phrases, hook=hook,
        fonts=[{"family": f["family"], "file": f["file"]} for f in fonts])
    html_path = os.path.join(comp_dir, "index.html")
    with open(html_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html_text)

    # 8. Manifest: the preview/export identity.
    base_sha = hashlib.sha256()
    with open(os.path.join(comp_dir, "base.mp4"), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            base_sha.update(chunk)
    manifest = {
        "compiler": COMPILER_VERSION,
        "project_digest": project_digest(project),
        "project_version": int(project.version),
        "job_id": project.job_id,
        "clip_index": int(clip_index),
        "template": {"id": spec.id, "version": spec.version},
        "params": {"caption_size": fp.caption_size,
                   "caption_position": fp.caption_position},
        "render": {"width": width, "height": height, "fps": fps},
        "duration": duration,
        "frames": frames,
        "base": {"sha256": base_sha.hexdigest(),
                 "provenance": base["provenance"],
                 "duration": base["duration"]},
        "fonts": [{"family": f["family"], "file": f["file"],
                   "sha256": f["sha256"]} for f in fonts],
        "renderer": renderer_versions(worker_dir),
        "captions": {"phrases": len(phrases),
                     "words": sum(len(p["words"]) for p in phrases)},
        "hook": ({"present": True, "start": hook["start"], "end": hook["end"]}
                 if hook else {"present": False}),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with open(os.path.join(comp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return CompiledComposition(
        compile_id=compile_id,
        composition_dir=comp_dir,
        html_path=html_path,
        base_path=os.path.join(comp_dir, "base.mp4"),
        digest=digest,
        manifest=manifest,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        phrases=phrases,
        hook=hook,
    )
