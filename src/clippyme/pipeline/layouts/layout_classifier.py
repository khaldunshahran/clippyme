"""Intelligent Layout Classifier using Gemini with 12-frame low-cost visual sampling.

Sends 12 downscaled (1024px) JPEG frames (~3k tokens) to classify the optimal
vertical framing strategy:
- "none" / "standard": single speaker or standard scene
- "split": two people in conversation / podcast
- "screencast": screen recording, spreadsheet, presentation
- "inset": gaming or stream VOD with corner webcam
"""
import json
import os

ENABLED = os.environ.get("AUTO_LAYOUT", "0").strip().lower() in ("1", "true", "yes")
SAMPLE_FRAMES = int(os.environ.get("LAYOUT_SAMPLE_FRAMES", "12"))
SAMPLE_WIDTH = int(os.environ.get("LAYOUT_SAMPLE_WIDTH", "1024"))

VALID_LAYOUTS = {"none", "standard", "split", "screencast", "inset"}

LAYOUT_PROMPT = """Analyze these sampled frames from one video scene.
Determine the most appropriate vertical (9:16) video framing layout among these options:
1. "split": A two-person podcast or dialogue where two main speakers sit apart in a wide shot.
2. "screencast": A screen recording, software demo, presentation, code editor, or spreadsheet where full-width screen content is essential.
3. "inset": A gaming stream or VOD where a small webcam facecam is pinned in a corner over full-screen gameplay/content.
4. "none": Standard talking head, vlog, monologue, or standard video (handled by standard smart active-speaker reframing).

Return strict JSON:
{
  "layout": "none" | "split" | "screencast" | "inset",
  "confidence": 0.0 to 1.0,
  "screen_evidence": true or false,
  "why": "concise explanation"
}

Rules:
- "screen_evidence" must be boolean true ONLY when you see positive,
  unambiguous screen content (code editor, slides, spreadsheet, software UI,
  gameplay) filling a substantial part of the frame. A talking head, wall
  poster, or corner face alone is NOT screen evidence; set it false.
- "screencast"/"inset" require "screen_evidence": true. Do not label a
  natural scene "screencast" just because a face is visible.
- "why" must be a nonempty concise explanation naming what you saw.
- The number of frames provided varies by scene length; judge only what
  is shown.
"""


def sample_frames(video_path, n=SAMPLE_FRAMES, width=SAMPLE_WIDTH):
    """Extract JPEG bytes for n frames distributed evenly across the video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    try:
        if total <= 0:
            return out
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / float(n)))
            ok, frame = cap.read()
            if not ok or frame.mean() < 16:
                continue
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (width, max(2, int(h * width / float(w)))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append(buf.tobytes())
    finally:
        cap.release()
    return out


def classify_layout(video_path, api_key=None, model_name=None):
    """Classify video layout using Gemini. Never raises: returns 'none' on any error."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "none"

    from clippyme.pipeline.gemini_service import get_auxiliary_gemini_model

    model_name = get_auxiliary_gemini_model(model_name)
    try:
        from google import genai
        from google.genai import types as genai_types

        frames = sample_frames(video_path)
        if not frames:
            return "none"

        client = genai.Client(api_key=api_key)
        parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]
        response = client.models.generate_content(
            model=model_name,
            contents=parts + [LAYOUT_PROMPT],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        if not response or not response.text:
            return "none"

        data = json.loads(response.text) or {}
        layout = str(data.get("layout", "none")).strip().lower()
        if layout == "standard":
            layout = "none"
        return layout if layout in VALID_LAYOUTS else "none"
    except Exception as e:
        print(f"[LayoutClassifier] Optional layout classification failed: {e}")
        return "none"


def is_auto_layout_enabled():
    """AUTO_LAYOUT as opt-in flag. Unset/empty/false -> False. Never raises."""
    try:
        return os.environ.get("AUTO_LAYOUT", "0").strip().lower() in ("1", "true", "yes")
    except Exception:
        return False


def pick_and_apply_layout(video_path, explicit_layout=None, api_key=None):
    """Determine the effective layout strategy.

    Explicit modes pass through unchanged. "auto" (or unset) consults the
    invocation-time AUTO_LAYOUT flag: when disabled, no classification runs
    and "auto" is returned. Never triggers paid classification while off.
    """
    if explicit_layout and explicit_layout not in ("auto", "none"):
        return explicit_layout

    if is_auto_layout_enabled():
        detected = classify_layout(video_path, api_key=api_key)
        return detected if detected != "none" else "auto"

    return "auto"


# --- Task 02: scene-local semantic classification ---------------------------

# Policy defaults (provisional, not calibrated). Conservative: failures
# retain the previous strategy.
MIN_SEMANTIC_CONFIDENCE = 0.85
MIN_SCENE_SAMPLES = 3
MIN_PRESENTER_DETECTIONS = 3
MIN_PRESENTER_COVERAGE = 0.60
MAX_PRESENTER_SPREAD_FRACTION = 0.10

# Budget bounds (per scene)
MAX_SCENE_SAMPLES = 8
CLASSIFY_TIMEOUT_S = 30

# Invocation-level budgets (per maybe_promote_screencast call)
MAX_SCENES_PER_INVOCATION = 32
MAX_REQUESTS_PER_INVOCATION = 16
INVOCATION_DEADLINE_S = 120.0
GEMINI_MAX_RETRIES = 2


def parse_layout_response(text):
    """Validate a provider JSON response into structured evidence.

    Strict contract:
    - layout in VALID_LAYOUTS ("standard" maps to "none")
    - confidence: finite number in [0, 1] (bool rejected)
    - screen_evidence: must be boolean. For "screencast"/"inset" it must be
      exactly True; False/missing/invalid never becomes positive from the
      label alone.
    - why/reason: nonempty string.
    Returns the evidence dict, or None when invalid. Never raises.
    """
    import json
    import math

    try:
        data = json.loads(text or "")
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    layout = str(data.get("layout", "")).strip().lower()
    if layout == "standard":
        layout = "none"
    if layout not in VALID_LAYOUTS:
        return None
    conf = data.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None
    if not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
        return None
    reason = data.get("why") or data.get("reason") or ""
    if not isinstance(reason, str) or not reason.strip():
        return None
    ev = data.get("screen_evidence")
    if layout in ("screencast", "inset"):
        if ev is not True:
            return None
        screen_evidence = True
    else:
        screen_evidence = ev is True
    return {
        "layout": layout,
        "confidence": float(conf),
        "screen_evidence": screen_evidence,
        "reason": reason.strip()[:500],
    }


def sample_scene_frames(video_path, start_frame, end_frame, n=MAX_SCENE_SAMPLES,
                        width=SAMPLE_WIDTH, _budget=None):
    """JPEG bytes for n frames within [start_frame, end_frame).

    Deduplicates indices; all samples lie inside the half-open scene
    bounds. n is clamped to [1, MAX_SCENE_SAMPLES]. Returns [] on
    unreadable video. Never raises.
    """
    import cv2

    try:
        n = max(1, min(int(n), MAX_SCENE_SAMPLES))
    except (TypeError, ValueError):
        n = MAX_SCENE_SAMPLES
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        try:
            s, e = int(start_frame), int(end_frame)
            if e <= s:
                return []
            import numpy as np
            idxs = sorted(set(int(round(x)) for x in np.linspace(s, e - 1, min(n, e - s))))
            idxs = [i for i in idxs if s <= i < e]
            out = []
            for f_idx in idxs:
                if _budget is not None and _budget.expired():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ok, frame = cap.read()
                if not ok or frame is None or frame.mean() < 16:
                    continue
                h, w = frame.shape[:2]
                scaled = cv2.resize(frame, (width, max(2, int(h * width / float(w)))),
                                    interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    out.append(buf.tobytes())
            return out
        finally:
            cap.release()
    except Exception:
        return []


def classify_scene(video_path, scene_id, start_frame, end_frame,
                   provider=None, api_key=None, model_name=None, _budget=None):
    """Classify one scene's layout. Never raises; returns evidence dict or None.

    provider: injectable callable(frames, prompt) -> response text.
    When None, uses the Gemini provider (requires AUTO_LAYOUT + api_key).
    _budget: optional shared budget; provider attempts respect remaining
      time/requests. Returns None on: disabled, no frames, provider error,
      invalid response, low confidence, budget exhausted.
    """
    if not is_auto_layout_enabled():
        return None
    if _budget is not None and _budget.expired():
        return None
    frames = sample_scene_frames(video_path, start_frame, end_frame,
                                 _budget=_budget)
    if len(frames) < MIN_SCENE_SAMPLES:
        return None
    # Recheck after sampling: injected provider must not start after
    # sampling exhausted the deadline.
    if _budget is not None and _budget.expired():
        return None
    try:
        if provider is not None:
            # Injected provider: consume one actual-call allowance here
            # (the SDK path consumes inside _gemini_classify).
            if _budget is not None and not _budget.take_attempt():
                return None
            text = provider(frames, LAYOUT_PROMPT)
        else:
            text = _gemini_classify(frames, api_key=api_key, model_name=model_name,
                                    _budget=_budget)
    except Exception:
        return None
    if not text:
        return None
    parsed = parse_layout_response(text)
    if parsed is None:
        return None
    if parsed["confidence"] < MIN_SEMANTIC_CONFIDENCE:
        return None
    parsed["scene_id"] = scene_id
    parsed["samples"] = len(frames)
    return parsed


def _gemini_classify(frames, api_key=None, model_name=None,
                     timeout_s=CLASSIFY_TIMEOUT_S, max_retries=GEMINI_MAX_RETRIES,
                     _budget=None):
    """Gemini provider with timeout and bounded retries.

    Shares the invocation budget: each actual SDK attempt consumes one
    request allowance via _budget.take_attempt(); per-attempt timeout is
    capped by remaining milliseconds; retries stop when budget is exhausted.
    SDK internal retries explicitly disabled via HttpRetryOptions(attempts=1).
    Every owned SDK client is closed exactly once on all exits (success,
    failure, budget exhaustion, setup/refresh errors) via finally.
    Returns response text or None. Never raises. No background threads.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    if _budget is not None and _budget.expired():
        return None
    try:
        from clippyme.pipeline.gemini_service import get_auxiliary_gemini_model
        from google import genai
        from google.genai import types as genai_types

        model_name = get_auxiliary_gemini_model(model_name)

        try:
            retry_opts = genai_types.HttpRetryOptions(attempts=1)
        except Exception as e:
            print(f"[LayoutClassifier] cannot configure SDK retry policy: {e}")
            return None

        def _attempt_timeout_ms():
            t_ms = int(timeout_s * 1000)
            if _budget is not None:
                rem_ms = int(_budget.remaining_s() * 1000)
                t_ms = min(t_ms, rem_ms)
            return max(0, t_ms)

        def _make_client(timeout_ms):
            http_opts = genai_types.HttpOptions(
                timeout=timeout_ms, retry_options=retry_opts)
            return genai.Client(api_key=api_key, http_options=http_opts)

        def _close_client(c):
            if c is None:
                return
            close = getattr(c, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        timeout_ms = _attempt_timeout_ms()
        if timeout_ms <= 0:
            return None
        try:
            client = _make_client(timeout_ms)
        except Exception as e:
            print(f"[LayoutClassifier] client creation failed: {e}")
            return None

        last_err = None
        try:
            # Part serialization is inside the owning try/finally: a setup
            # error closes the already-allocated client exactly once.
            try:
                parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg")
                         for b in frames]
            except Exception as e:
                print(f"[LayoutClassifier] part setup failed: {e}")
                return None
            for attempt in range(max_retries + 1):
                if _budget is not None:
                    if not _budget.take_attempt():
                        break
                # Refresh timeout from current remaining budget.
                try:
                    cur_ms = _attempt_timeout_ms()
                    if cur_ms <= 0:
                        break
                    if cur_ms != timeout_ms:
                        # Detach and close old before replacing. If the
                        # replacement allocation raises, the finally must
                        # not close the already-closed client a second time.
                        old_client, client = client, None
                        _close_client(old_client)
                        timeout_ms = cur_ms
                        client = _make_client(timeout_ms)
                except Exception as e:
                    print(f"[LayoutClassifier] timeout refresh failed: {e}")
                    break
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=parts + [LAYOUT_PROMPT],
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                        ),
                    )
                    if response and response.text:
                        return response.text
                    return None
                except Exception as e:
                    last_err = e
                    if attempt < max_retries:
                        if _budget is not None and _budget.expired():
                            break
                        import time
                        delay = min(2.0 ** attempt, 4.0)
                        if _budget is not None:
                            delay = min(delay, _budget.remaining_s())
                        if delay > 0:
                            time.sleep(delay)
                        else:
                            break
            if last_err is not None:
                print(f"[LayoutClassifier] scene classification failed: {last_err}")
            return None
        finally:
            # Close the current owned client on all exits.
            _close_client(client)
    except Exception as e:
        print(f"[LayoutClassifier] scene classification failed: {e}")
        return None
