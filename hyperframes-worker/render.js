#!/usr/bin/env node
/**
 * nugget-hyperframes-worker — bounded local Hyperframes render worker (Task 18 pilot).
 *
 * Renders ONE compiled composition (index.html + local assets, produced by
 * the typed Python compiler in src/clippyme/finishing/) to MP4 using the real
 * @hyperframes/engine (Puppeteer + headless Chromium capture) and FFmpeg.
 *
 * The engine drives the page's `window.__hf` seek protocol frame-by-frame and
 * injects pre-extracted base-video frames deterministically; frames are
 * encoded with the engine's chunkEncoder and muxed with the base audio track.
 *
 * Usage:
 *   node render.js --manifest <manifest.json> --out <output.mp4> --workdir <dir>
 *
 * The manifest is JSON written by the compiler:
 *   {
 *     "composition_dir": "<abs dir containing index.html + assets>",
 *     // NOTE: the engine navigates to {serverUrl}/index.html (hardcoded),
 *     // so the entry file must be named index.html and serverUrl below
 *     // is the file-server root, not the page URL.
 *     "width": 608, "height": 1080,
 *     "fps": 30, "duration": 12.5, "frames": 375,
 *     "audio_source": "<abs path to base mp4 carrying the audio track>" | null,
 *     "format": "png",
 *     "deadline_sec": 1200
 *   }
 *
 * Bounds (contract 6): duration <= 180s, frames <= 10800, request JSON <= 1MB.
 * One job at a time: an exclusive lockfile in the worker dir. Finite deadline.
 * Cancellation: the spawner creates <workdir>/CANCEL to request abort.
 * Temp cleanup always; atomic output install only after ffprobe validation.
 *
 * Exit codes: 0 ok | 2 bad input | 3 busy (lock held) | 4 deadline |
 *             5 cancelled | 6 render/encode/mux failure | 7 validation failure
 *
 * Result metrics are written to <workdir>/result.json on every terminal path
 * (except bad-input/busy): runtime_sec, frames_captured, peak memory (Chrome
 * RSS sampler), temp disk peak. Cache behavior is decided by the spawner.
 *
 * Notes:
 * - The browser is launched per session with the browser pool DISABLED, so one
 *   job owns exactly one browser and no cross-job state leaks.
 * - On Windows the engine resolves capture mode to "screenshot" (documented
 *   engine heuristic; atomic BeginFrame is Linux-only). Seeks still go through
 *   the deterministic `window.__hf.seek(t)` protocol.
 * - The file server binds an ephemeral localhost port (port 0): it never
 *   touches repo-owned 5175/5176 or any production/dev port.
 */
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

const ENGINE_VERSION = "0.8.145";
const MAX_DURATION_SEC = 180;
const MAX_FRAMES = 10800;
const MAX_MANIFEST_BYTES = 1024 * 1024;
const DEFAULT_DEADLINE_SEC = 1200;
const LOCK_NAME = ".render.lock";

const EXIT = { OK: 0, BAD_INPUT: 2, BUSY: 3, DEADLINE: 4, CANCELLED: 5, FAILED: 6, INVALID: 7 };

function writeResult(workdir, status, error, metrics) {
  try {
    fs.mkdirSync(workdir, { recursive: true });
    fs.writeFileSync(
      path.join(workdir, "result.json"),
      JSON.stringify({ status, ...(error ? { error } : {}), ...metrics }, null, 2)
    );
  } catch { /* best effort */ }
}

function fail(workdir, status, code, message, metrics) {
  writeResult(workdir, status, message, metrics || {});
  process.stderr.write(`[hf-worker] ${message}\n`);
  process.exit(code);
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--manifest") out.manifest = argv[++i];
    else if (a === "--out") out.out = argv[++i];
    else if (a === "--workdir") out.workdir = argv[++i];
    else if (a === "--help" || a === "-h") out.help = true;
  }
  return out;
}

function readManifest(p) {
  let st;
  try {
    st = fs.statSync(p);
  } catch {
    fail(null, "failed", EXIT.BAD_INPUT, `cannot stat manifest: ${p}`, {});
  }
  if (st.size > MAX_MANIFEST_BYTES) {
    fail(null, "failed", EXIT.BAD_INPUT, `manifest too large (${st.size} > ${MAX_MANIFEST_BYTES})`, {});
  }
  let m;
  try {
    m = JSON.parse(fs.readFileSync(p, "utf8"));
  } catch (e) {
    fail(null, "failed", EXIT.BAD_INPUT, `manifest is not valid JSON: ${e.message}`, {});
  }
  for (const k of ["composition_dir", "width", "height", "fps", "duration", "frames"]) {
    if (m[k] === undefined || m[k] === null) {
      fail(null, "failed", EXIT.BAD_INPUT, `manifest missing '${k}'`, {});
    }
  }
  const num = (v, name, lo, hi) => {
    if (typeof v === "boolean" || typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi) {
      fail(null, "failed", EXIT.BAD_INPUT, `manifest '${name}' out of bounds: ${JSON.stringify(v)}`, {});
    }
  };
  num(m.width, "width", 16, 4096);
  num(m.height, "height", 16, 4096);
  num(m.fps, "fps", 1, 120);
  num(m.duration, "duration", 0.05, MAX_DURATION_SEC);
  if (!Number.isInteger(m.frames) || m.frames < 1 || m.frames > MAX_FRAMES) {
    fail(null, "failed", EXIT.BAD_INPUT, `manifest 'frames' out of bounds: ${JSON.stringify(m.frames)}`, {});
  }
  if (Math.abs(m.frames - Math.round(m.duration * m.fps)) > 1) {
    fail(null, "failed", EXIT.BAD_INPUT, `manifest frames (${m.frames}) != round(duration*fps)`, {});
  }
  if (!fs.statSync(path.join(m.composition_dir, "index.html"), { throwIfNoEntry: false })?.isFile()) {
    fail(null, "failed", EXIT.BAD_INPUT, `index.html missing in ${m.composition_dir}`, {});
  }
  if (m.audio_source !== null && m.audio_source !== undefined) {
    if (typeof m.audio_source !== "string" ||
        !fs.statSync(m.audio_source, { throwIfNoEntry: false })?.isFile()) {
      fail(null, "failed", EXIT.BAD_INPUT, `audio_source missing: ${m.audio_source}`, {});
    }
  }
  m.deadline_sec = m.deadline_sec === undefined ? DEFAULT_DEADLINE_SEC : m.deadline_sec;
  num(m.deadline_sec, "deadline_sec", 5, 7200);
  if (m.format !== undefined && m.format !== "png" && m.format !== "jpeg") {
    fail(null, "failed", EXIT.BAD_INPUT, `manifest 'format' must be png|jpeg`, {});
  }
  m.format = m.format || "png";
  return m;
}

/** Exclusive one-job lock. Returns an unlock fn. Exits BUSY if held by a live pid. */
function acquireLock() {
  const lockPath = path.join(HERE, LOCK_NAME);
  const mine = `${process.pid}`;
  try {
    fs.writeFileSync(lockPath, mine, { flag: "wx" });
    return () => { try { fs.unlinkSync(lockPath); } catch {} };
  } catch (e) {
    if (e.code !== "EEXIST") fail(null, "failed", EXIT.FAILED, `lock error: ${e.message}`, {});
    let owner = "";
    try { owner = fs.readFileSync(lockPath, "utf8").trim(); } catch {}
    const pid = parseInt(owner, 10);
    let alive = Number.isInteger(pid) && pid > 0;
    if (alive) {
      try { process.kill(pid, 0); } catch { alive = false; }
    }
    if (alive) {
      fail(null, "failed", EXIT.BUSY, `another render job holds the worker lock (pid ${pid})`, {});
    }
    try { fs.unlinkSync(lockPath); } catch {}
    try {
      fs.writeFileSync(lockPath, mine, { flag: "wx" });
      process.stderr.write(`[hf-worker] took over stale lock (dead pid ${owner})\n`);
      return () => { try { fs.unlinkSync(lockPath); } catch {} };
    } catch (e2) {
      fail(null, "failed", EXIT.BUSY, `lost lock race: ${e2.message}`, {});
    }
  }
}

function ffprobeJson(bin, file) {
  const r = spawnSync(bin, ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", file],
    { encoding: "utf8", timeout: 60000 });
  if (r.status !== 0) throw new Error(`ffprobe failed: ${(r.stderr || "").slice(0, 400)}`);
  return JSON.parse(r.stdout || "{}");
}

function dirSizeBytes(dir) {
  let total = 0;
  const walk = (d) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p);
      else { try { total += fs.statSync(p).size; } catch {} }
    }
  };
  try { walk(dir); } catch {}
  return total;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.manifest || !args.out || !args.workdir) {
    process.stdout.write("usage: node render.js --manifest <json> --out <mp4> --workdir <dir>\n");
    process.exit(args.help ? 0 : EXIT.BAD_INPUT);
  }
  const t0 = Date.now();
  const workdir = path.resolve(args.workdir);
  const metrics = { engine_version: ENGINE_VERSION };
  let status = "failed";

  fs.mkdirSync(workdir, { recursive: true });
  // Validate the manifest BEFORE taking the lock: fail() exits without
  // cleanup, so a bad-input rejection must never leak a held lockfile.
  const manifest = readManifest(args.manifest);
  const unlock = acquireLock();
  metrics.manifest = {
    width: manifest.width, height: manifest.height, fps: manifest.fps,
    duration: manifest.duration, frames: manifest.frames,
  };

  const cancelled = () => {
    try {
      return fs.statSync(path.join(workdir, "CANCEL"), { throwIfNoEntry: false })?.isFile() === true;
    } catch { return false; }
  };

  let engine = null;
  let server = null;
  let session = null;
  let installedOut = null;
  const framesDir = path.join(workdir, "frames");

  const cleanup = async () => {
    try { if (session) await engine.closeCaptureSession(session); } catch {}
    session = null;
    try { if (server) server.close(); } catch {}
    server = null;
    try { fs.rmSync(framesDir, { recursive: true, force: true }); } catch {}
    try { unlock(); } catch {}
  };

  const deadlineTimer = setTimeout(() => {
    status = "deadline";
    metrics.runtime_sec = (Date.now() - t0) / 1000;
    cleanup().finally(() => {
      fail(workdir, status, EXIT.DEADLINE, `render exceeded deadline of ${manifest.deadline_sec}s`, metrics);
    });
  }, manifest.deadline_sec * 1000);
  // Don't let the timer alone keep the process alive after normal exit.
  deadlineTimer.unref?.();

  try {
    if (cancelled()) {
      status = "cancelled";
      metrics.runtime_sec = (Date.now() - t0) / 1000;
      await cleanup(); clearTimeout(deadlineTimer);
      fail(workdir, status, EXIT.CANCELLED, "cancel requested before start", metrics);
    }

    engine = await import("@hyperframes/engine");
    const {
      createFileServer, createCaptureSession, initializeSession, captureFrame,
      closeCaptureSession, encodeFramesFromDir, muxVideoWithAudio,
      getCapturePerfSummary, getCompositionDuration,
    } = engine;

    // 1. Serve the composition dir on an ephemeral localhost port.
    server = await createFileServer({ projectDir: manifest.composition_dir, port: 0 });
    // Engine contract: serverUrl is the file-server ROOT; the engine
    // navigates to {serverUrl}/index.html itself (frameCapture.js).
    const serverRoot = server.url.replace(/\/$/, "");
    process.stderr.write(`[hf-worker] serving ${manifest.composition_dir} at ${server.url}\n`);

    // 2. Capture session. The engine launches and owns one browser for this
    //    session; the pool is disabled so no cross-job browser state leaks.
    fs.mkdirSync(framesDir, { recursive: true });
    session = await createCaptureSession(serverRoot, framesDir, {
      width: manifest.width,
      height: manifest.height,
      fps: { num: manifest.fps, den: 1 },
      format: manifest.format,
      compositionDurationSeconds: manifest.duration,
    }, null, { enableBrowserPool: false });
    await initializeSession(session);
    metrics.capture_mode = session.captureMode || null;
    const hfDuration = await getCompositionDuration(session).catch(() => null);
    if (hfDuration !== null && Math.abs(hfDuration - manifest.duration) > 1 / manifest.fps + 1e-6) {
      throw new Error(
        `composition __hf.duration (${hfDuration}) disagrees with manifest (${manifest.duration})`);
    }
    if ((session.captureWarnings || []).length) {
      metrics.capture_warnings = session.captureWarnings.map(w =>
        (typeof w === "string" ? w : (w && w.code ? `${w.code}: ${w.message || ""}` : JSON.stringify(w))));
    }

    // 3. Frame loop with deadline + cancellation checks.
    let captured = 0;
    let tempPeak = 0;
    for (let i = 0; i < manifest.frames; i++) {
      if (cancelled()) {
        status = "cancelled";
        metrics.frames_captured = captured;
        metrics.runtime_sec = (Date.now() - t0) / 1000;
        await cleanup(); clearTimeout(deadlineTimer);
        fail(workdir, status, EXIT.CANCELLED, `cancel requested at frame ${i}/${manifest.frames}`, metrics);
      }
      await captureFrame(session, i, i / manifest.fps);
      captured++;
      if (captured % 30 === 0) {
        tempPeak = Math.max(tempPeak, dirSizeBytes(framesDir));
        process.stderr.write(`[hf-worker] frame ${captured}/${manifest.frames}\n`);
      }
    }
    tempPeak = Math.max(tempPeak, dirSizeBytes(framesDir));
    metrics.frames_captured = captured;
    metrics.temp_disk_peak_bytes = tempPeak;

    try {
      const perf = getCapturePerfSummary(session);
      metrics.capture_perf = {
        frames: perf?.frames ?? null,
        total_ms: perf?.totalMs ?? null,
      };
    } catch {}
    await closeCaptureSession(session);
    try {
      if (session.chromeMemory) {
        const mem = session.chromeMemory.stats();
        metrics.peak_memory_mb = {
          browser_rss_peak: mem.browserRssPeakMb ?? null,
          renderer_rss_peak: mem.rendererRssPeakMb ?? null,
          rss_last: mem.rssLastMb ?? null,
          samples: mem.samples ?? 0,
        };
      }
    } catch {}
    session = null;
    server.close(); server = null;

    // 4. Encode frames -> silent mp4 (real engine encoder), then mux base audio.
    const tmpVideo = path.join(workdir, "video.tmp.mp4");
    const ext = manifest.format === "jpeg" ? "jpg" : "png";
    const enc = await encodeFramesFromDir(framesDir, `frame_%06d.${ext}`, tmpVideo, {
      fps: { num: manifest.fps, den: 1 },
      width: manifest.width, height: manifest.height,
      codec: "h264", pixelFormat: "yuv420p",
    });
    if (!enc.success) throw new Error(`encode failed: ${enc.error || "unknown"}`);
    metrics.encode_ms = enc.durationMs ?? null;
    metrics.encode_frames = enc.framesEncoded ?? null;

    let finalTmp = tmpVideo;
    if (manifest.audio_source) {
      const muxed = path.join(workdir, "mux.tmp.mp4");
      const mux = await muxVideoWithAudio(tmpVideo, manifest.audio_source, muxed);
      if (!mux.success) throw new Error(`audio mux failed: ${mux.error || "unknown"}`);
      try { fs.unlinkSync(tmpVideo); } catch {}
      finalTmp = muxed;
      metrics.mux_ms = mux.durationMs ?? null;
    }

    // 5. Media validation BEFORE atomic install (contract 6).
    const ffprobe = process.env.FFPROBE_PATH || "ffprobe";
    let probe;
    try {
      probe = ffprobeJson(ffprobe, finalTmp);
    } catch (e) {
      try { fs.unlinkSync(finalTmp); } catch {}
      status = "failed";
      metrics.runtime_sec = (Date.now() - t0) / 1000;
      await cleanup(); clearTimeout(deadlineTimer);
      fail(workdir, status, EXIT.INVALID, `output validation failed: ${e.message}`, metrics);
    }
    const vstreams = (probe.streams || []).filter(s => s.codec_type === "video");
    const astreams = (probe.streams || []).filter(s => s.codec_type === "audio");
    const problems = [];
    if (vstreams.length < 1) problems.push("no video stream in output");
    const v = vstreams[0] || {};
    const outW = parseInt(v.width, 10), outH = parseInt(v.height, 10);
    if (outW !== manifest.width || outH !== manifest.height) {
      problems.push(`dimensions ${outW}x${outH} != ${manifest.width}x${manifest.height}`);
    }
    const outDur = parseFloat((probe.format && probe.format.duration) || "0");
    if (Math.abs(outDur - manifest.duration) > 0.5) {
      problems.push(`duration ${outDur} != ${manifest.duration}`);
    }
    if (manifest.audio_source && astreams.length < 1) {
      problems.push("expected audio stream missing after mux");
    }
    if (problems.length) {
      try { fs.unlinkSync(finalTmp); } catch {}
      status = "failed";
      metrics.runtime_sec = (Date.now() - t0) / 1000;
      await cleanup(); clearTimeout(deadlineTimer);
      fail(workdir, status, EXIT.INVALID, `output validation failed: ${problems.join("; ")}`, metrics);
    }
    metrics.validated = {
      width: outW, height: outH, duration: outDur,
      video_streams: vstreams.length, audio_streams: astreams.length,
      nb_frames: parseInt(v.nb_frames || "0", 10) || null,
    };

    // 6. Atomic install.
    const outPath = path.resolve(args.out);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.renameSync(finalTmp, outPath);
    installedOut = outPath;

    clearTimeout(deadlineTimer);
    status = "rendered";
    metrics.runtime_sec = (Date.now() - t0) / 1000;
    metrics.output = outPath;
    metrics.output_bytes = fs.statSync(outPath).size;
    writeResult(workdir, status, null, metrics);
    await cleanup();
    process.stdout.write(JSON.stringify({ status: "rendered", output: outPath }) + "\n");
    process.exit(EXIT.OK);
  } catch (e) {
    clearTimeout(deadlineTimer);
    if (installedOut) {
      // Never leave a half-installed output behind on the success path only;
      // here nothing was installed yet (install is the last step).
    }
    if (status !== "cancelled" && status !== "deadline") status = "failed";
    metrics.runtime_sec = (Date.now() - t0) / 1000;
    await cleanup();
    const code = status === "cancelled" ? EXIT.CANCELLED : EXIT.FAILED;
    fail(workdir, status, code, `render failed: ${e && e.message ? e.message : e}`, metrics);
  }
}

main();
