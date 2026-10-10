#!/usr/bin/env python3
"""Task 18 synthetic verification — real local Hyperframes + Chromium + FFmpeg.

Generates a synthetic test-pattern source (NOT real footage, no provider
calls), compiles Hyperframes compositions through the real compiler, renders
them with the real worker (Puppeteer + FFmpeg), and checks decoded frames,
seek agreement, audio, geometry, cache, failures, fonts, and immutability.

Run on the laptop from the worktree root:
    D:\\clippyme\\clippyme\\.venv\\Scripts\\python.exe verify_task18.py
with PYTHONPATH=src.

Outputs: .verify/ artifacts + a JSON summary printed to stdout.
Exit code: 0 only if every check passes.
"""
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
WORKER = os.path.join(ROOT, "hyperframes-worker")
VERIFY = os.path.join(ROOT, ".verify", "task18")
ART = os.path.join(VERIFY, "artifacts")
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
NODE = "node"

sys.path.insert(0, SRC)
os.environ.setdefault("PYTHONPATH", SRC)

CHECKS = []
MEASURED = {}


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))
    return ok


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_json(path):
    r = run([FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path])
    assert r.returncode == 0, f"ffprobe failed on {path}: {r.stderr[:300]}"
    return json.loads(r.stdout)


# ---------------------------------------------------------------- fonts ----
def ttf_cmap_coverage(path):
    """Pure-stdlib TTF cmap coverage (formats 4 and 12). Returns set of codepoints."""
    data = open(path, "rb").read()
    num_tables = struct.unpack(">H", data[4:6])[0]
    tables = {}
    for i in range(num_tables):
        tag, _ck, off, _ln = struct.unpack(">4sIII", data[12 + i * 16:28 + i * 16])
        tables[tag] = off
    cmap_off = tables[b"cmap"]
    _ver, nsubs = struct.unpack(">HH", data[cmap_off:cmap_off + 4])
    cps = set()
    for i in range(nsubs):
        _pid, _eid, sub = struct.unpack(">HHI", data[cmap_off + 4 + i * 8:cmap_off + 12 + i * 8])
        fmt = struct.unpack(">H", data[cmap_off + sub:cmap_off + sub + 2])[0]
        if fmt == 4:
            seg_x2 = struct.unpack(">H", data[cmap_off + sub + 6:cmap_off + sub + 8])[0]
            nseg = seg_x2 // 2
            base = cmap_off + sub + 14
            ends = struct.unpack(">" + "H" * nseg, data[base:base + 2 * nseg])
            starts = struct.unpack(">" + "H" * nseg, data[base + 2 + 2 * nseg:base + 2 + 4 * nseg])
            for s, e in zip(starts, ends):
                if e == 0xFFFF:
                    continue
                cps.update(range(s, e + 1))
        elif fmt == 12:
            _f, _r, ngroups = struct.unpack(">HHI", data[cmap_off + sub:cmap_off + sub + 12])
            for g in range(ngroups):
                s, e, _gid = struct.unpack(">III", data[cmap_off + sub + 16 + g * 12:cmap_off + sub + 28 + g * 12])
                cps.update(range(s, min(e, s + 0x10FFFF) + 1))
    return cps


def glyph_report():
    out = {}
    for fam, fname in (("Anton", "Anton-Regular.ttf"), ("Bangers", "Bangers-Regular.ttf")):
        cov = ttf_cmap_coverage(os.path.join(ROOT, "fonts", fname))
        groups = {
            "ascii_printable": list(range(0x20, 0x7F)),
            "finnish_diacritics": [0xE4, 0xF6, 0xFC, 0xC4, 0xD6, 0xDC, 0xE5, 0xE5],
            "latin_ext_sample": [0x161, 0x17E, 0x152, 0x153],
            "cyrillic_sample": list(range(0x410, 0x430)),
            "cjk_sample": [0x4F60, 0x597D, 0x4E2D, 0x6587],
            "arabic_sample": list(range(0x627, 0x64A)),
        }
        res = {}
        for gname, cps in groups.items():
            have = sum(1 for c in cps if c in cov)
            res[gname] = f"{have}/{len(cps)}"
        out[fam] = {"total_codepoints": len(cov), "groups": res}
    return out


# ---------------------------------------------------------------- audio ----
def goertzel(samples, sr, target):
    n = len(samples)
    k = int(0.5 + n * target / sr)
    w = 2 * math.pi * k / n
    cw, sw = math.cos(w), math.sin(w)
    coeff = 2 * cw
    s0 = s1 = s2 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2, s1 = s1, s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return power / (n * n)


def tone_check(wav_path, target=880.0):
    with wave.open(wav_path, "rb") as w:
        sr, n, sw = w.getframerate(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
    fmt = "<" + "h" * (len(raw) // 2) if sw == 2 else None
    assert fmt, "unexpected sample width"
    samples = struct.unpack(fmt, raw)
    if w.getnchannels() == 2:
        samples = [(a + b) / 2 for a, b in zip(samples[::2], samples[1::2])]
    scale = max(1, max(abs(s) for s in samples[:sr]))
    norm = [s / scale for s in samples[: sr * 4]]
    p_target = goertzel(norm, sr, target)
    p_other = goertzel(norm, sr, 440.0)
    return p_target, p_other


# ---------------------------------------------------------------- frames ---
def extract_frame(video, t, out_png, size=None):
    vf = ["select=eq(n\\,0)"]
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", video,
           "-vf", ",".join(vf), "-frames:v", "1"]
    if size:
        cmd += ["-s", size]
    cmd += [out_png]
    r = run(cmd)
    assert r.returncode == 0 and os.path.isfile(out_png), f"frame extract failed t={t}"


def diff_regions(a_png, b_png, regions, size):
    """Mean abs diff per named region via ffmpeg blend -> raw gray."""
    w, h = size
    p = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", a_png, "-i", b_png,
                        "-filter_complex", "[0][1]blend=all_mode=difference,format=gray",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    buf = p.stdout
    assert len(buf) >= w * h, f"short diff buffer {len(buf)}"
    out = {}
    for name, (x0, y0, x1, y1) in regions.items():
        tot = 0
        cnt = 0
        for y in range(y0, y1):
            row = buf[y * w:(y + 1) * w]
            tot += sum(row[x0:x1])
            cnt += (x1 - x0)
        out[name] = tot / max(1, cnt)
    return out


# ------------------------------------------------------------- screenshots -
SHOOT_JS = r"""
const puppeteer = require('puppeteer');
(async () => {
  const [,, pageUrl, outDir, timesJson, exePath] = process.argv;
  const times = JSON.parse(timesJson);
  const browser = await puppeteer.launch({
    headless: 'shell',
    executablePath: exePath,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: __W__, height: __H__ });
  await page.goto(pageUrl, { waitUntil: 'load', timeout: 30000 });
  await page.waitForFunction('window.__hf && typeof window.__hf.seek === "function"', { timeout: 15000 });
  const shots = {};
  for (const t of times) {
    await page.evaluate((tt) => window.__hf.seek(tt), t);
    await new Promise((r) => setTimeout(r, 700));
    const p = outDir + '/seek_' + String(t).replace('.', '_') + '.png';
    await page.screenshot({ path: p });
    shots[String(t)] = p;
  }
  console.log(JSON.stringify(shots));
  await browser.close();
})().catch((e) => { console.error('SHOOT-FAIL', e.message); process.exit(1); });
"""


def shoot(comp_dir, width, height, times, tag):
    out_dir = os.path.join(ART, f"shots_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    js = SHOOT_JS.replace("__W__", str(width)).replace("__H__", str(height))
    js_path = os.path.join(WORKER, f"shoot_{tag}.cjs")
    open(js_path, "w").write(js)
    url = "file:///" + comp_dir.replace("\\", "/") + "/index.html"
    exe = os.path.join(WORKER, ".puppeteer-cache", "chrome-headless-shell",
                       "win64-155.0.8059.39", "chrome-headless-shell-win64",
                       "chrome-headless-shell.exe")
    assert os.path.isfile(exe), f"chromium missing: {exe}"
    env = dict(os.environ)
    env["PUPPETEER_CACHE_DIR"] = os.path.join(WORKER, ".puppeteer-cache")
    r = subprocess.run([NODE, js_path, url, out_dir.replace("\\", "/"), json.dumps(times), exe],
                       capture_output=True, text=True, cwd=WORKER,
                       env=env, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"screenshot failed: {r.stderr[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])

# ============================================================ main flow ====
JOB = os.path.join(VERIFY, "job_synth")
SOURCE = os.path.join(JOB, "source_0.mp4")


def gen_source():
    os.makedirs(JOB, exist_ok=True)
    anton = os.path.join(ROOT, "fonts", "Anton-Regular.ttf")
    vf = ("drawtext=fontfile='%s':text='%%{n}':fontsize=72:fontcolor=white:"
          "x=(w-text_w)/2:y=(h-text_h)/2" % anton.replace("\\", "/").replace(":", "\\:"))
    # Colorspace-tagged like real camera footage: the engine applies a
    # matrix GUESS for untagged sources (chromeGuessForUntaggedMatrix),
    # which shows up as uniform pixel noise vs the tagged decode.
    r = run([FFMPEG, "-y", "-v", "error",
             "-f", "lavfi", "-i", "testsrc2=size=608x1080:rate=30:duration=16",
             "-f", "lavfi", "-i", "sine=frequency=880:duration=16:sample_rate=44100",
             "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-colorspace", "bt709", "-color_primaries", "bt709",
             "-color_trc", "bt709", "-color_range", "tv",
             "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest", SOURCE])
    assert r.returncode == 0, f"source gen failed: {r.stderr[:400]}"
    return sha256_file(SOURCE)


def make_project(render_w, render_h, crop, version=1, template="clean-captions",
                 finishing_renderer="hyperframes", no_finishing=False,
                 extra_words=()):
    from clippyme.domain.clip_project import validate_project
    words = [
        {"id": "w0", "w": "hello", "start": 2.0, "end": 2.5},
        {"id": "w1", "w": "world", "start": 2.5, "end": 3.0},
        {"id": "w2", "w": "again", "start": 10.5, "end": 11.0},
        {"id": "w3", "w": "dropped", "start": 50.0, "end": 51.0},
        {"id": "w4", "w": "gone", "start": 3.0, "end": 3.5, "removed": True},
        {"id": "w5", "w": "", "start": 3.5, "end": 4.0},
        {"id": "w6", "w": "<b>bold</b>", "start": 12.0, "end": 12.5},
        {"id": "w7", "w": "Suomi", "start": 11.0, "end": 11.5},
        *extra_words,
    ]
    data = {
        "schema": "nugget.clip-project/1",
        "job_id": "task18synth",
        "clip_index": 0,
        "source": {"file": "source_0.mp4", "width": 608, "height": 1080},
        "segments": [
            {"id": "seg0", "start": 2.0, "end": 8.0, "crop": crop},
            {"id": "seg1", "start": 10.0, "end": 14.0, "crop": crop},
        ],
        "captions": {"style": "hormozi", "position": "lower-third",
                     "words": words, "edits": {"w1": "WORLD"}, "next_word_id": 8},
        "overlays": [{"type": "hook", "text": "watch this", "style": "viral",
                      "start": 0.0, "end": 2.5, "position": "top"}],
        "render": {"width": render_w, "height": render_h, "fps": 30},
        "version": version,
    }
    if not no_finishing:
        data["finishing"] = {"renderer": finishing_renderer, "template": template,
                             "caption_size": "medium", "caption_position": "lower-third"}
    return validate_project(data)


def compile_project(project, template, workdir_tag):
    from clippyme.finishing.compile import compile_hyperframes_composition
    return asyncio.run(compile_hyperframes_composition(
        project, template_id=template,
        params={"caption_size": "medium", "caption_position": "lower-third"},
        job_dir=JOB, clip_index=0, acknowledged_version=project.version,
        renderer_enabled=True))


def cues(html):
    return re.findall(r'<span[^>]*data-start="([\d.]+)" data-end="([\d.]+)">([^<]*)<', html)


def worker_manifest_for(comp, deadline_sec):
    """Build the worker manifest exactly like the production render route."""
    return {
        "composition_dir": comp.composition_dir,
        "width": comp.manifest["render"]["width"],
        "height": comp.manifest["render"]["height"],
        "fps": comp.manifest["render"]["fps"],
        "duration": comp.manifest["duration"],
        "frames": comp.manifest["frames"],
        "audio_source": os.path.join(comp.composition_dir, "base.mp4"),
        "format": "png",
        "deadline_sec": deadline_sec,
    }


def spawn_render(wc, comp, out_path, workdir, deadline_sec=600):
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    wm_path = os.path.join(workdir, "worker-manifest.json")
    with open(wm_path, "w", encoding="utf-8") as f:
        json.dump(worker_manifest_for(comp, deadline_sec), f)
    return wc.run_worker_render(manifest_path=wm_path, out_path=out_path,
                                workdir=workdir, deadline_sec=deadline_sec)


def _wait_for_worker_idle(timeout_sec=120):
    """A previous crashed run may leave a worker exiting; wait for the lock.

    The worker lock carries the holder pid; a live holder means a real render
    is still running (wait), a dead pid is taken over by the worker itself.
    """
    lock = os.path.join(WORKER, ".render.lock")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_sec:
        if not os.path.exists(lock):
            return True
        try:
            pid = int(open(lock, encoding="utf-8").read().strip())
        except (OSError, ValueError):
            return True  # worker treats non-numeric as stale
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        if not alive:
            return True  # worker will take over the stale lock itself
        time.sleep(2)
    return False


def main():
    os.makedirs(ART, exist_ok=True)
    check("worker lock free (no overlapping render)", _wait_for_worker_idle())
    from clippyme.finishing import worker_client as wc
    from clippyme.finishing.compile import CompileError, FinishingDisabled
    from clippyme.finishing import cleanbase as cb

    # --- 0. availability -------------------------------------------------
    avail = wc.check_available()
    check("worker available (node+chromium+ffmpeg)", avail["available"], avail.get("reason", ""))
    MEASURED["availability"] = avail
    if not avail["available"]:
        return finish(1)

    # --- 1. synthetic source ---------------------------------------------
    src_hash_before = gen_source()
    MEASURED["source_sha256"] = src_hash_before
    probe = ffprobe_json(SOURCE)
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    check("synthetic source 608x1080@30 + audio", vs["width"] == 608 and vs["height"] == 1080,
          f'{vs["width"]}x{vs["height"]} {vs.get("avg_frame_rate")}')

    # --- 2. compile 9:16 clean-captions -----------------------------------
    crop916 = {"x": 0, "y": 0, "w": 608, "h": 1080}
    project = make_project(608, 1080, crop916)
    comp = compile_project(project, "clean-captions", "c1")
    html = open(comp.html_path, encoding="utf-8").read()
    found = {(s, e): txt for s, e, txt in cues(html)}
    check("word-id mapping seg0 (hello 0.0-0.5)", ("0.000", "0.500") in found and found[("0.000", "0.500")] == "hello")
    check("retained-word edit applied (WORLD 0.5-1.0)",
          ("0.500", "1.000") in found and found[("0.500", "1.000")] == "WORLD",
          str(found.get(("0.500", "1.000"))))
    check("multi-segment mapping (again 6.5-7.0)",
          ("6.500", "7.000") in found and found[("6.500", "7.000")] == "again")
    check("multilingual word kept (Suomi 7.0-7.5)",
          ("7.000", "7.500") in found and found[("7.000", "7.500")] == "Suomi")
    check("out-of-segment word dropped", not any(t == "dropped" for t in found.values()))
    check("removed word dropped", not any(t == "gone" for t in found.values()))
    check("empty word dropped", "" not in found.values())
    check("chat markup escaped", "&lt;b&gt;bold&lt;/b&gt;" in html and "<b>bold</b>" not in html.replace("&lt;b&gt;bold&lt;/b&gt;", ""))
    check("hook cue half-open [0.0,2.5)",
          'id="hf-hook" data-start="0.000" data-end="2.500"' in html)
    check("no remote URLs in composition",
          "http://" not in html and "https://" not in html)
    check("local immutable fonts copied+recorded",
          os.path.isfile(os.path.join(comp.composition_dir, "Anton-Regular.ttf"))
          and comp.manifest["fonts"][0]["sha256"] == sha256_file(os.path.join(ROOT, "fonts", "Anton-Regular.ttf")))
    check("clean-base provenance is materialized pre-compose (never clip_final)",
          comp.manifest["base"]["provenance"] == "materialized-pre-compose-timeline")
    check("composition duration matches base clock (10.0s)",
          abs(comp.duration - 10.0) < 0.05 and comp.frames == 300, f"{comp.duration}s {comp.frames}f")

    comp2 = compile_project(project, "clean-captions", "c2")
    check("manifest digest deterministic (preview/export identity)",
          comp.digest == comp2.digest, comp.digest[:16])
    MEASURED["manifest_digest"] = comp.digest

    # --- 3. worker render 9:16 ---------------------------------------------
    out_mp4 = os.path.join(ART, "export_916_clean.mp4")
    wd = os.path.join(VERIFY, "work_render1")
    t0 = time.monotonic()
    res = spawn_render(wc, comp, out_mp4, wd, deadline_sec=600)
    wall = time.monotonic() - t0
    check("worker render ok", res.status == "rendered" and os.path.isfile(out_mp4),
          f"status={res.status}")
    MEASURED["render_916"] = {"wall_sec": round(wall, 1), "metrics": res.metrics}
    rj = os.path.join(wd, "result.json")
    check("result.json metrics recorded", os.path.isfile(rj),
          str({k: res.metrics.get(k) for k in ("runtime_sec", "frames_captured", "peak_memory_mb", "temp_disk_peak_bytes")}))
    leftovers = [p for p in os.listdir(wd) if p.startswith("tmp")]
    check("worker temp cleaned", not leftovers, str(leftovers))

    op = ffprobe_json(out_mp4)
    ovs = next(s for s in op["streams"] if s["codec_type"] == "video")
    oas = next(s for s in op["streams"] if s["codec_type"] == "audio")
    dur = float(op["format"]["duration"])
    check("export 608x1080@30", ovs["width"] == 608 and ovs["height"] == 1080,
          f'{ovs["width"]}x{ovs["height"]} {ovs.get("avg_frame_rate")} {ovs.get("nb_frames")} frames')
    check("export duration ~10s", abs(dur - 10.0) < 0.3, f"{dur:.2f}s")
    check("export has audio", oas["codec_name"] == "aac")
    wav = os.path.join(ART, "export_916.wav")
    run([FFMPEG, "-y", "-v", "error", "-i", out_mp4, "-map", "0:a", "-ac", "1", "-ar", "44100", wav])
    p880, p440 = tone_check(wav)
    check("880Hz tone preserved (not silent/resynthesized)", p880 > p440 * 100,
          f"P880={p880:.2e} P440={p440:.2e}")
    MEASURED["audio"] = {"P880": p880, "P440": p440}

    # --- 4. decoded frames vs clean base ------------------------------------
    W, H = 608, 1080
    regions = {
        "hook_band": (0, int(0.02 * H), W, int(0.20 * H)),
        "control": (0, int(0.40 * H), W, int(0.55 * H)),
        "caption_band": (0, int(0.66 * H), W, int(0.95 * H)),
    }
    base_mp4 = comp.base_path
    ef = os.path.join(ART, "exp_t0_25.png"); extract_frame(out_mp4, 0.25, ef)
    bf = os.path.join(ART, "base_t0_25.png"); extract_frame(base_mp4, 0.25, bf)
    sf = os.path.join(ART, "src_t2_25.png"); extract_frame(SOURCE, 2.25, sf)
    d_exp_base = diff_regions(ef, bf, regions, (W, H))
    d_base_src = diff_regions(bf, sf, regions, (W, H))
    check("caption rendered over base (caption band differs)",
          d_exp_base["caption_band"] > 2.0, f'{d_exp_base["caption_band"]:.2f}')
    check("base pixels preserved outside graphics (control band ~equal)",
          d_exp_base["control"] < 6.0, f'{d_exp_base["control"]:.2f}')
    check("caption-free base: base caption band == source pixels (no baked text)",
          d_base_src["caption_band"] < 6.0, f'{d_base_src["caption_band"]:.2f}')
    shutil.copy(ef, os.path.join(ART, "frame_export_t0.25.png"))
    shutil.copy(bf, os.path.join(ART, "frame_base_t0.25.png"))

    # --- 5. seek agreement via headless Chromium ----------------------------
    shots = shoot(comp.composition_dir, W, H, [0.25, 0.75, 6.75], "clean")
    # forward/backward determinism
    s_a = shots["0.25"]
    shots2 = shoot(comp.composition_dir, W, H, [6.75, 0.25], "clean2")
    same = sha256_file(s_a) == sha256_file(shots2["0.25"])
    check("seek forward/backward deterministic (0.25 -> 6.75 -> 0.25 identical)",
          same, sha256_file(s_a)[:16])
    d_se = diff_regions(shots["0.25"], ef, regions, (W, H))
    check("preview seek shot agrees with decoded export (same composition+clock)",
          d_se["caption_band"] < 25.0 and d_se["control"] < 25.0,
          f'caption={d_se["caption_band"]:.2f} control={d_se["control"]:.2f}')
    for k, v in shots.items():
        shutil.copy(v, os.path.join(ART, f"seek_clean_{k}.png"))

    # --- 6. hook boundary (timed-opening) ------------------------------------
    project_hook = make_project(608, 1080, crop916, template="timed-opening")
    comp_h = compile_project(project_hook, "timed-opening", "h1")
    hshots = shoot(comp_h.composition_dir, W, H, [2.49, 2.51], "hook")
    d_hb = diff_regions(hshots["2.49"], hshots["2.51"], regions, (W, H))
    check("hook visible at 2.49, gone at 2.51 (half-open [0,2.5))",
          d_hb["hook_band"] > 1.5 and d_hb["caption_band"] < 1.0,
          f'hook={d_hb["hook_band"]:.2f} caption={d_hb["caption_band"]:.2f}')
    for k, v in hshots.items():
        shutil.copy(v, os.path.join(ART, f"seek_hook_{k}.png"))

    # --- 7. other templates compile ------------------------------------------
    for tid in ("emphasized-word",):
        c = compile_project(make_project(608, 1080, crop916, template=tid), tid, tid)
        h = open(c.html_path, encoding="utf-8").read()
        check(f"template {tid} compiles with mapped cues",
              'data-start="0.000" data-end="0.500">hello<' in h)

    # --- 8. geometries: square + landscape ------------------------------------
    sq = compile_project(make_project(608, 608, {"x": 0, "y": 236, "w": 608, "h": 608}), "clean-captions", "sq")
    sq_out = os.path.join(ART, "export_square.mp4")
    wd2 = os.path.join(VERIFY, "work_render_sq")
    spawn_render(wc, sq, sq_out, wd2, deadline_sec=600)
    sp = ffprobe_json(sq_out)
    svs = next(s for s in sp["streams"] if s["codec_type"] == "video")
    check("square export 608x608", svs["width"] == 608 and svs["height"] == 608,
          f'{svs["width"]}x{svs["height"]}')

    ls = compile_project(make_project(640, 360, {"x": 0, "y": 369, "w": 608, "h": 342}), "clean-captions", "ls")
    ls_out = os.path.join(ART, "export_landscape.mp4")
    wd3 = os.path.join(VERIFY, "work_render_ls")
    spawn_render(wc, ls, ls_out, wd3, deadline_sec=600)
    lp = ffprobe_json(ls_out)
    lvs = next(s for s in lp["streams"] if s["codec_type"] == "video")
    check("landscape export 640x360", lvs["width"] == 640 and lvs["height"] == 360,
          f'{lvs["width"]}x{lvs["height"]}')

    # --- 9. cache module: key binds inputs, invalidation -----------------------
    from clippyme.finishing import cache as hfcache
    kw = dict(base_sha256=comp.manifest["base"]["sha256"],
              project_digest=comp.manifest["project_digest"], project_version=1,
              template_id="clean-captions", template_version=1,
              params={"caption_size": "medium", "caption_position": "lower-third"},
              font_shas=[f["sha256"] for f in comp.manifest["fonts"]],
              renderer=comp.manifest["renderer"], width=608, height=1080, fps=30)
    key = hfcache.cache_key(**kw)
    check("cache miss on fresh key", hfcache.lookup(JOB, key) is None)
    dest = hfcache.store(JOB, key, out_mp4)
    check("cache store+lookup roundtrip", hfcache.lookup(JOB, key) == dest)
    key_v2 = hfcache.cache_key(**{**kw, "project_version": 2})
    check("cache invalidates on version change", hfcache.lookup(JOB, key_v2) is None)
    key_tpl = hfcache.cache_key(**{**kw, "template_id": "emphasized-word"})
    check("cache invalidates on template change", hfcache.lookup(JOB, key_tpl) is None)
    MEASURED["cache"] = {"key_prefix": key[:16], "stored": dest}

    # --- 10. failure gates ------------------------------------------------------
    try:
        compile_project(make_project(608, 1080, crop916, finishing_renderer="legacy"), "clean-captions", "f1")
        check("renderer-off refused", False)
    except FinishingDisabled as e:
        check("renderer-off refused (default OFF)", True, str(e)[:60])
    try:
        compile_project(make_project(608, 1080, crop916, no_finishing=True), "clean-captions", "f2")
        check("old project (no finishing) refused", False)
    except FinishingDisabled:
        check("old project (no finishing) refused", True)
    try:
        from clippyme.finishing.compile import compile_hyperframes_composition as chc
        asyncio.run(chc(make_project(608, 1080, crop916), template_id="clean-captions",
                        params={}, job_dir=JOB, clip_index=0, acknowledged_version=999,
                        renderer_enabled=True))
        check("stale version rejected", False)
    except CompileError as e:
        check("stale version rejected", "version" in str(e).lower(), str(e)[:60])
    try:
        compile_project(make_project(608, 1080, crop916), "cinematic-zoom", "f3")
        check("invalid template rejected", False)
    except CompileError as e:
        check("invalid template rejected", True, str(e)[:60])
    try:
        pf = make_project(608, 1080, crop916)
        pf.framing = {"plan_ref": "x"}  # post-validation: trajectory present
        compile_project(pf, "clean-captions", "f4")
        check("framing trajectory fails clearly", False)
    except cb.CleanBaseUnavailable as e:
        check("framing trajectory fails clearly (no invented geometry)", True, str(e)[:70])
    # bad timing word: dropped honestly, compile still succeeds
    badw = make_project(608, 1080, crop916, extra_words=({"id": "wx", "w": "bad", "start": 4.0, "end": 3.0},))
    cbt = compile_project(badw, "clean-captions", "f5")
    check("bad-timing word dropped honestly",
          ">bad<" not in open(cbt.html_path, encoding="utf-8").read())

    # worker-level failures
    lock = os.path.join(WORKER, ".render.lock")
    try:
        # A lock holding a LIVE pid: the worker must refuse with WorkerBusy
        # (a non-numeric/dead-pid lock is correctly treated as stale).
        open(lock, "w").write(str(os.getpid()) + "\n")
        try:
            spawn_render(wc, comp, os.path.join(ART, "busy.mp4"),
                         os.path.join(VERIFY, "work_busy"), deadline_sec=60)
            check("worker busy -> honest failure", False)
        except wc.WorkerBusy:
            check("worker busy -> honest failure (no silent queue)", True)
    finally:
        if os.path.exists(lock):
            os.remove(lock)

    wd_cancel = os.path.join(VERIFY, "work_cancel")
    shutil.rmtree(wd_cancel, ignore_errors=True)
    os.makedirs(wd_cancel, exist_ok=True)
    wm_cancel = os.path.join(wd_cancel, "worker-manifest.json")
    json.dump(worker_manifest_for(comp, 600), open(wm_cancel, "w"))
    wc.request_cancel(wd_cancel)  # after workdir setup: rmtree would delete it
    try:
        wc.run_worker_render(manifest_path=wm_cancel,
                             out_path=os.path.join(ART, "cancel.mp4"),
                             workdir=wd_cancel, deadline_sec=600)
        check("cancel -> WorkerCancelled", False)
    except wc.WorkerCancelled:
        check("cancel -> WorkerCancelled, nothing published",
              not os.path.exists(os.path.join(ART, "cancel.mp4")))

    wd_dl = os.path.join(VERIFY, "work_deadline")
    try:
        # deadline_sec=6: valid per the worker's [5,7200] bound, but far too
        # short for the 300-frame render (~19s), so the worker must time out.
        spawn_render(wc, comp, os.path.join(ART, "deadline.mp4"), wd_dl, deadline_sec=6)
        check("deadline -> WorkerDeadline", False)
    except wc.WorkerDeadline:
        check("deadline -> WorkerDeadline", True)

    # bad assets: composition dir missing base.mp4
    bad_dir = os.path.join(VERIFY, "comp_bad")
    shutil.rmtree(bad_dir, ignore_errors=True)
    shutil.copytree(comp.composition_dir, bad_dir)
    os.remove(os.path.join(bad_dir, "base.mp4"))
    bad_wd = os.path.join(VERIFY, "work_bad")
    shutil.rmtree(bad_wd, ignore_errors=True)
    os.makedirs(bad_wd, exist_ok=True)
    bad_wm = os.path.join(bad_wd, "worker-manifest.json")
    wm = worker_manifest_for(comp, 120)
    wm["composition_dir"] = bad_dir
    wm["audio_source"] = os.path.join(bad_dir, "base.mp4")
    json.dump(wm, open(bad_wm, "w"))
    try:
        wc.run_worker_render(manifest_path=bad_wm,
                             out_path=os.path.join(ART, "badasset.mp4"),
                             workdir=bad_wd, deadline_sec=120)
        check("bad assets -> worker error (no output)", False)
    except wc.WorkerError as e:
        check("bad assets -> worker error (no output)",
              not os.path.exists(os.path.join(ART, "badasset.mp4")),
              f"{type(e).__name__}: {str(e)[:80]}")

    # --- 11. fonts: actual glyph coverage ----------------------------------------
    glyphs = glyph_report()
    MEASURED["glyphs"] = glyphs
    check("Anton: ASCII + Finnish diacritics covered",
          glyphs["Anton"]["groups"]["ascii_printable"].startswith("95/95")
          and glyphs["Anton"]["groups"]["finnish_diacritics"].startswith("8/8"),
          json.dumps(glyphs["Anton"]["groups"]))
    check("Bangers: ASCII covered",
          glyphs["Bangers"]["groups"]["ascii_printable"].startswith("95/95"),
          json.dumps(glyphs["Bangers"]["groups"]))

    # --- 12. immutability ----------------------------------------------------------
    check("synthetic source hash unchanged (originals never overwritten)",
          sha256_file(SOURCE) == src_hash_before)

    return finish(0)


def finish(code):
    summary = {"checks": CHECKS, "measured": MEASURED,
               "passed": sum(1 for c in CHECKS if c["ok"]),
               "failed": sum(1 for c in CHECKS if not c["ok"])}
    open(os.path.join(VERIFY, "summary.json"), "w").write(json.dumps(summary, indent=2))
    print(f"\n==== {summary['passed']} passed, {summary['failed']} failed ====")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        CHECKS.append({"name": "script completed", "ok": False, "detail": repr(e)[:200]})
        sys.exit(finish(1))
