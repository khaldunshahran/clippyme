/**
 * UI-01 offline fixture — THROWAWAY build config (never used for
 * production). Extends the app's real plugins and aliases the
 * network/session/chat modules to offline stubs so the REAL AppShell,
 * the REAL useWorkspaceController and the REAL createProjectController +
 * queue can run in a browser with zero network.
 *
 * Aliases are exact import specifiers (as written by the importers), so
 * the shim's own import of the real projectController with a ?ui01real
 * query never matches './projectController.js'.
 *
 * Build:  node node_modules/vite/bin/vite.js build --config vite.ui01fixture.config.js
 * Output: /tmp/ui01-fixture-dist (NOT dashboard-chat/dist).
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const stub = (p) => path.join(here, 'src', 'ui01-fixture', 'stubs', p);

// Resolves the virtual 'ui01-real-project-controller' specifier used by
// the observation shim to the real module, before the alias plugin runs.
function realProjectControllerPlugin() {
  return {
    name: 'ui01-real-project-controller',
    enforce: 'pre',
    resolveId(source) {
      if (source === 'ui01-real-project-controller') {
        return path.join(here, 'src', 'chat', 'projectController.js');
      }
      return null;
    },
  };
}

export default defineConfig({
  plugins: [realProjectControllerPlugin(), react(), tailwindcss()],
  base: './',
  resolve: {
    alias: {
      './chat/useChat.js': stub('useChat.js'),
      './api/chatApi.js': stub('chatApi.js'),
      '../api/chatApi.js': stub('chatApi.js'),
      '../api/realApi.js': stub('realApi.js'),
      '../api/realApi': stub('realApi.js'), // clipEditor.jsx omits the extension
      '../api/sessionReset.js': stub('sessionReset.js'),
      '../api/config.js': stub('config.js'),
      './projectController.js': stub('projectControllerShim.js'),
    },
  },
  build: {
    outDir: '/tmp/ui01-fixture-dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        fixture: path.join(here, 'ui01-fixture.html'),
        frame390: path.join(here, 'frame-390.html'),
      },
    },
  },
});
