# UI-01 offline fixture (review-01 evidence, updated for review-02)

Renders the **real** `AppShell` (real React root, real hooks, real
`useWorkspaceController` → real `createProjectController` + real serialized
edit queue) in headless Chromium with **zero network**: the API/chat/session
modules are aliased to offline stubs by the throwaway
`vite.ui01fixture.config.js` (never used for production builds).

## Layout

- `main.jsx` — fixture entry; query params below. Imports the real
  global CSS/theme (`../index.css` — the same import production
  `src/main.jsx` makes, review-02), so screenshots from this shell can
  speak for production theme/dimensions/contrast.
- `dev-main.jsx` (+ `dashboard-chat/ui01-fixture-dev.html`) —
  development-React entry for StrictMode evidence; `?ui01dev=1` runs
  its checks. See "Test kinds" below.
- `driver.js` — scripted browser tests (`?ui01test=1` desktop lifecycle
  incl. draft-source selection and rendered/historical playback on
  real playable media, `?ui01test=mobile` narrow-viewport tab switching)
- `verify.js` — screenshot-mode control verification (bounding boxes,
  keyboard focus, non-overlapping identity, production theme, editor
  content-area width, transcript contrast) rendered into the badge
- `stubs/` — offline `useChat`, `realApi` (fake versioned backend with a
  gateable save), `chatApi`, `sessionReset`, `projectControllerShim`
  (counts controller creations/disposals)

## Query params

- `sel=0` — no clip selected
- `actions=closeTranscript,hideChat,mobileEditor,mobileChat` — post-mount UI actions
- `ui01test=1|mobile` — run the scripted test instead of screenshot verify
- `label=...` — human label for the screenshot badge

## Build (from `dashboard-chat/`)

```
node node_modules/vite/bin/vite.js build --config vite.ui01fixture.config.js
node node_modules/vite/bin/vite.js build --config vite.ui01fixture.config.js --mode development --outDir C:/tmp/ui01-fixture-dist-dev
```

Output goes to `/tmp/ui01-fixture-dist` (production-mode bundle) and
`/tmp/ui01-fixture-dist-dev` (development-React bundle) — never
`dashboard-chat/dist`.

## Synthetic media (review-02)

The stub backend references small playable mp4s by relative name
(`fixture-source.mp4`, `fixture-source-2.mp4`,
`fixture-composed-v3.mp4`, `fixture-render-v3.mp4`). They are generated
artifacts — never committed — produced with the installed ffmpeg into
BOTH dist folders after building (6s, 360x640, test pattern + tone;
composed/render use a distinct pattern):

```
foreach ($d in @('C:\tmp\ui01-fixture-dist','C:\tmp\ui01-fixture-dist-dev')) {
  ffmpeg -y -v error -f lavfi -i testsrc2=size=360x640:rate=30:duration=6 -f lavfi -i sine=frequency=440:duration=6 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest -movflags +faststart "$d\fixture-source.mp4"
  ffmpeg -y -v error -f lavfi -i testsrc2=size=360x640:rate=30:duration=6 -f lavfi -i sine=frequency=550:duration=6 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest -movflags +faststart "$d\fixture-source-2.mp4"
  ffmpeg -y -v error -f lavfi -i smptebars=size=360x640:rate=30:duration=6 -f lavfi -i sine=frequency=660:duration=6 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest -movflags +faststart "$d\fixture-composed-v3.mp4"
  ffmpeg -y -v error -f lavfi -i smptebars=size=360x640:rate=30:duration=6 -f lavfi -i sine=frequency=770:duration=6 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest -movflags +faststart "$d\fixture-render-v3.mp4"
}
```

## Test kinds (what each one actually proves)

1. **Real-root browser tests on the production-mode bundle**
   (`main.jsx` + `driver.js`). The real `AppShell` React root is
   mounted in a real browser via `createRoot` — no hook mocking — and
   the driver asserts controller creation/disposal counts around
   mount, unmount and remount. NOTE: this bundle is built in
   production mode, and in a production React bundle `<StrictMode>`
   does NOT double-invoke effects (the driver asserts `disposed=0` on
   initial mount). These are real-root browser tests; they are NOT
   StrictMode evidence.
2. **Development-React StrictMode evidence** (`dev-main.jsx`, built
   `--mode development`, `?ui01dev=1` on `ui01-fixture-dev.html`). The
   development React build makes `<StrictMode>` mount effects
   setup -> cleanup -> setup; the dev driver asserts those counts
   (a production bundle fails them), a deferred save held across a
   Hide chat -> Show chat layout switch, and a remount cycle.
3. **Framework-independent controller lifecycle tests (Vitest).**
   `src/chat/ui01ControllerLifecycle.test.js` drives
   `createProjectController` directly for the queue/ack/dirty/restore
   contracts; it does not mount React and does not claim to. The
   review-02 identity/timebase/source contracts are pinned in
   `src/chat/ui01Review02.test.js` and the coordinator probe
   `src/chat/coordinatorUi01Review02.test.js`.

## Run transport (review-02)

All browser runs serve the dist folders from a **throwaway localhost
server** (`python -m http.server 8931 --bind 127.0.0.1 --directory
C:/tmp/ui01-fixture-dist`, started for the run and stopped after it).
Reason: under `--virtual-time-budget`, media data over `file://`
stalls at HAVE_METADATA (the file loader is not virtualized), so the
playable-media timebase checks can never observe playback; over
127.0.0.1 HTTP the same budget fast-forwards the loads and the media
reaches HAVE_FUTURE_DATA. Nothing else about the fixture changes —
all references are relative.

## Screenshot one variant (headless Chrome, throwaway profile)

```
& 'C:\Program Files\Google\Chrome\Application\chrome.exe' --headless=new `
  --disable-gpu `
  --user-data-dir=$env:TEMP\ui01-chrome --hide-scrollbars `
  --window-size=1440,900 --virtual-time-budget=12000 `
  --screenshot=C:\shots\selected-1440.png `
  'http://127.0.0.1:8931/ui01-fixture.html?label=selected'
```

## Run the scripted browser tests

Same Chrome flags with `--dump-dom` and `?ui01test=1` (1440px) /
`?ui01test=mobile` (390px); the DOM contains `#ui01-test-results` JSON and
the title becomes `UI01-TEST-DONE p/t`. Add
`--autoplay-policy=no-user-gesture-required` — the desktop driver now
plays real media for the timebase checks. The dev bundle's checks run
at `ui01-fixture-dev.html?ui01dev=1`; the title becomes
`UI01DEV p/t PASS|FAIL` and the results render into `#ui01-dev-results`.
