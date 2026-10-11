// serve-dist.mjs — range-capable static server for the UI-01 fixture
// evidence runs (review-04). Node builtins only.
//
// Why not `python -m http.server` (the review-02 recipe)? That server
// speaks HTTP/1.0 and sends no Accept-Ranges; current Chrome then
// reports the 30s source media as seekable [0,0] and silently drops
// every currentTime write, so the driver's real-media seek checks
// (deferred source seek, artifact seeks) can never land — the app
// code is not involved. Any static server with Range support works;
// this one keeps the evidence recipe dependency-free.
//
// Usage: node serve-dist.mjs <port> <directory>
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { join, normalize, extname } from 'node:path';

const port = Number(process.argv[2] || 8931);
const root = process.argv[3];
if (!root) { console.error('usage: node serve-dist.mjs <port> <directory>'); process.exit(1); }

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.mp4': 'video/mp4',
  '.woff2': 'font/woff2', '.ico': 'image/x-icon',
};

createServer((req, res) => {
  try {
    const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
    const rel = normalize(urlPath).replace(/^(\.\.[/\\])+/, '');
    const file = join(root, rel);
    const st = statSync(file);
    if (!st.isFile()) throw new Error('not a file');
    const type = MIME[extname(file).toLowerCase()] || 'application/octet-stream';
    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      const start = m && m[1] ? Number(m[1]) : 0;
      const end = m && m[2] ? Number(m[2]) : st.size - 1;
      res.writeHead(206, {
        'Content-Type': type, 'Accept-Ranges': 'bytes',
        'Content-Range': `bytes ${start}-${end}/${st.size}`,
        'Content-Length': end - start + 1,
      });
      createReadStream(file, { start, end }).pipe(res);
    } else {
      res.writeHead(200, {
        'Content-Type': type, 'Accept-Ranges': 'bytes',
        'Content-Length': st.size,
      });
      createReadStream(file).pipe(res);
    }
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`serve-dist: http://127.0.0.1:${port} -> ${root}`);
});
