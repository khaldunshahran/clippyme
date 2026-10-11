# UI-01 offline fixture (review-01 evidence)

Renders the **real** `AppShell` (real React root, real hooks, real
`useWorkspaceController` → real `createProjectController` + real serialized
edit queue) in headless Chromium with **zero network**: the API/chat/session
modules are aliased to offline stubs by the throwaway
`vite.ui01fixture.config.js` (never used for production builds).

## Layout

- `main.jsx` — fixture entry; query params below
- `driver.js` — scripted browser tests (`?ui01test=1` desktop lifecycle,
  `?ui01test=mobile` narrow-viewport tab switching)
- `verify.js` — screenshot-mode control verification (bounding boxes,
  keyboard focus, non-overlapping identity) rendered into the badge
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
```

Output goes to `/tmp/ui01-fixture-dist` — never `dashboard-chat/dist`.

## Screenshot one variant (headless Chrome, throwaway profile)

```
& 'C:\Program Files\Google\Chrome\Application\chrome.exe' --headless=new `
  --disable-gpu --allow-file-access-from-files `
  --user-data-dir=$env:TEMP\ui01-chrome --hide-scrollbars `
  --window-size=1440,900 --virtual-time-budget=8000 `
  --screenshot=C:\shots\selected-1440.png `
  'file:///C:/Users/<you>/AppData/Local/Temp/ui01-fixture-dist/ui01-fixture.html?label=selected'
```

## Run the scripted browser tests

Same Chrome flags with `--dump-dom` and `?ui01test=1` (1440px) /
`?ui01test=mobile` (390px); the DOM contains `#ui01-test-results` JSON and
the title becomes `UI01-TEST-DONE p/t`.
