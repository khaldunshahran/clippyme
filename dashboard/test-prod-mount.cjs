// Production-bundle mount regression test.
// Guards against the 2026-09-30 incident: a TDZ ReferenceError
// ("Cannot access 'pushToast' before initialization") crashed the entire app
// on load because the auth effect's dependency array read pushToast before
// its useCallback declaration executed. Unit tests run against source with
// ESM semantics and never caught it — only the minified production bundle
// exhibited the crash. This test mounts the real built bundle in jsdom and
// fails if the app throws during initial render or shows the error boundary.
//
// Usage: npm run build && npm run test:prod-mount
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const distDir = path.join(__dirname, 'dist');

function fail(msg) {
  console.error('PROD-MOUNT FAIL:', msg);
  process.exit(1);
}

if (!fs.existsSync(path.join(distDir, 'index.html'))) {
  fail('dist/index.html not found — run `npm run build` first.');
}

const html = fs.readFileSync(path.join(distDir, 'index.html'), 'utf8');
const assets = fs.readdirSync(path.join(distDir, 'assets'));
const jsFile = assets.find((f) => f.endsWith('.js') && !f.endsWith('.map'));
if (!jsFile) fail('no JS bundle found in dist/assets');
const js = fs.readFileSync(path.join(distDir, 'assets', jsFile), 'utf8');

const errors = [];
const dom = new JSDOM(html, {
  url: 'https://tellagbe.com/',
  runScripts: 'outside-only',
  beforeParse(window) {
    window.console = console;
    window.addEventListener('error', (e) => {
      errors.push('window.onerror: ' + e.message + '\n' + (e.error && e.error.stack));
    });
  },
});

try {
  dom.window.eval(js);
} catch (e) {
  fail('bundle threw during evaluation:\n' + (e && e.stack));
}

setTimeout(() => {
  const root = dom.window.document.getElementById('root');
  if (!root) fail('#root element missing after mount');
  const text = root.textContent || '';
  if (/unexpected UI error/i.test(text)) {
    fail('error boundary rendered:\n' + text.slice(0, 500));
  }
  if (errors.length > 0) {
    fail('window errors captured:\n' + errors.join('\n---\n'));
  }
  if (root.children.length === 0 || text.trim().length < 20) {
    fail('app mounted but rendered no meaningful content');
  }
  console.log('PROD-MOUNT OK: bundle', jsFile, 'mounted without errors, root text starts with:');
  console.log('  ' + text.slice(0, 120).replace(/\s+/g, ' '));
  process.exit(0);
}, 2000);
