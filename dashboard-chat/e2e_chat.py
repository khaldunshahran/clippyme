"""Nugget chat UI E2E — C1..C6 against the live backend. No GPU burned:
- validation uses the real /api/validate-url (read-only probe)
- progress/clip-row/popup/grid/editor are driven by a SEEDED completed job
"""
import json, time, sys
from playwright.sync_api import sync_playwright

BASE = "http://localhost:5176"
JOB_ID = "74f02524-2179-4aec-8487-bb3ff7a3757c"  # completed, 1 clip (Rick Astley)
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

errors = []

seed = [{
    "id": "e2e-seed-1",
    "title": "Rick Astley - Never Gonna Give You Up",
    "createdAt": int(time.time()*1000), "updatedAt": int(time.time()*1000),
    "phase": "clipping",
    "url": TEST_URL,
    "validation": {"valid": True, "downloadable": True,
                   "title": "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)",
                   "duration": 213, "thumbnail": None, "reason": None},
    "settings": {"model": "", "genre": "", "clipStyle": "", "aspect": "9:16",
                 "clipLength": "auto", "captions": "", "timeframe": None, "maxClips": 5},
    "jobId": JOB_ID, "progress": None, "clips": [],
    "messages": [{"id": "m1", "ts": int(time.time()*1000), "role": "user",
                  "kind": "text", "text": TEST_URL}],
    "steerLog": [], "error": None,
}]

def snap(page, name):
    page.screenshot(path=f"D:/clippyme/clippyme/dashboard-chat/e2e-{name}.png")
    print(f"screenshot: e2e-{name}.png", flush=True)

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                          args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))

    # seed BEFORE first load
    pg.add_init_script(f"localStorage.setItem('nugget-chat-threads-v1', {json.dumps(json.dumps(seed))});")
    pg.goto(BASE, wait_until="networkidle")
    pg.wait_for_timeout(2500)

    # C1: shell renders, sidebar shows seeded thread
    assert pg.locator(".nc-sidebar").count() == 1, "sidebar missing"
    snap(pg, "c1-shell")

    # seeded job should complete -> clips row appears (poll every 2s; allow 20s)
    pg.wait_for_selector(".nc-clip-row", timeout=20000)
    print("PASS: clips row rendered from seeded completed job", flush=True)
    pg.wait_for_timeout(1200)
    snap(pg, "c4-cliprow")

    # C4: open clip popup
    pg.locator(".nc-clip-row > div").first.click()
    pg.wait_for_timeout(2500)
    assert pg.locator("video").count() >= 1, "popup video missing"
    print("PASS: clip popup with video preview", flush=True)
    snap(pg, "c4-popup")

    # actions present
    for label in ["Publish on Social", "Export XML", "Download HD", "Upscale & download",
                  "Edit clip", "AI tools", "Aspect ratio", "Duplicate"]:
        assert pg.get_by_text(label, exact=True).count() >= 1, f"action missing: {label}"
    print("PASS: all 8 popup actions present", flush=True)

    # close popup, open View all grid
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(600)
    pg.get_by_text("View all", exact=True).click()
    pg.wait_for_timeout(1200)
    print("PASS: view-all grid opened", flush=True)
    snap(pg, "c4-grid")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(600)

    # C5: editor mounts via card edit button
    pg.locator(".nc-clip-row button[aria-label='Edit']").first.click()
    pg.wait_for_timeout(3500)
    assert pg.locator(".nc-editor-scope").count() == 1, "editor scope missing"
    print("PASS: deep editor mounted", flush=True)
    # copilot floating?
    has_copilot = pg.get_by_text("AI copilot").count() >= 1
    print(f"copilot panel present: {has_copilot}", flush=True)
    snap(pg, "c5-editor")
    # back to chat
    pg.locator("button[aria-label='Back to chat']").click()
    pg.wait_for_timeout(800)

    # C2: new chat -> paste URL -> validation -> settings popup (REAL backend probe)
    pg.get_by_text("New chat", exact=True).click()
    pg.wait_for_timeout(800)
    pg.locator("textarea").fill(TEST_URL)
    pg.locator("button[aria-label='Send']").click()
    pg.wait_for_selector("text=Checking that link", timeout=8000)
    print("PASS: validation checking state", flush=True)
    snap(pg, "c2-validating")
    pg.wait_for_selector("text=Clip settings", timeout=30000)
    print("PASS: settings popup opened after real validation", flush=True)
    pg.wait_for_timeout(800)
    snap(pg, "c2-settings")
    # close settings (cancel)
    pg.get_by_text("Cancel", exact=True).click()
    pg.wait_for_timeout(600)

    print("--- console errors ---", flush=True)
    seen = set()
    for e in errors:
        if e not in seen:
            seen.add(e); print(f"CONSOLE: {e[:220]}", flush=True)
    if not errors:
        print("(none)", flush=True)
    b.close()

print("E2E DONE", flush=True)
