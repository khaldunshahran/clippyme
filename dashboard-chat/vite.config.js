import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Nugget chat UI — dev server on :5176 (the classic dashboard keeps :5175).
// /api, /videos, /thumbnails, /fonts are proxied to the laptop backend (:8000).
// NOTE: the backend's trusted-Origin gate only allows :5175 (plus prod origins)
// and it reads ALLOWED_ORIGINS at import time, so we present the already-trusted
// :5175 origin on proxied API calls. This is dev-only; Phase D adds the real
// production origin to ALLOWED_ORIGINS with a proper backend restart.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5176,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('origin', 'http://localhost:5175');
          });
        },
      },
      '/videos': { target: 'http://localhost:8000', changeOrigin: true },
      '/thumbnails': { target: 'http://localhost:8000', changeOrigin: true },
      '/fonts': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
});
