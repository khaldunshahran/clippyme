
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
  await page.setViewport({ width: 608, height: 1080 });
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
