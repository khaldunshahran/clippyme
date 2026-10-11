/**
 * UI-01 offline fixture — observation shim around the REAL
 * createProjectController.
 *
 * The real module is imported via the virtual specifier
 * 'ui01-real-project-controller', resolved by a dedicated plugin in
 * vite.ui01fixture.config.js (it never matches the './projectController.js'
 * alias). Everything is re-exported; createProjectController is wrapped to
 * count creations and disposals. Every controller instance is also
 * published on window.__ui01Controllers for the scripted browser test.
 */
import { createProjectController as realCreate } from 'ui01-real-project-controller';

export * from 'ui01-real-project-controller';

const stats = { created: 0, disposed: 0 };

export function createProjectController(opts) {
  const c = realCreate(opts);
  stats.created += 1;
  const origDispose = c.dispose.bind(c);
  c.dispose = (...args) => {
    stats.disposed += 1;
    return origDispose(...args);
  };
  if (typeof window !== 'undefined') {
    window.__ui01Controllers = window.__ui01Controllers || [];
    window.__ui01Controllers.push(c);
  }
  return c;
}

export function __ui01ControllerStats() {
  return { ...stats };
}
