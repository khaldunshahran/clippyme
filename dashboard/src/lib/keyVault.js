// Nugget — per-account API key vault.
//
// Problem: third-party keys (Gemini, …) used to live only in browser
// localStorage, which is scoped to the exact origin — every new deployment URL
// started with empty storage, so keys looked "removed" after each update.
//
// How it works: when signed in (Supabase Auth), keys are stored in the user's
// own auth metadata under `nugget_api_keys`. Only that user (via their access
// token) can read/write their own metadata, so keys follow the account across
// deploys, devices, and browsers. localStorage stays as an instant read cache
// and as the full store when signed out (today's behavior, unchanged).
//
// Sync rules (last-writer-wins, server is the shared truth):
// - set: localStorage immediately; server push debounced (typing shouldn't
//   fire a request per keystroke); skips the push when the value is unchanged.
// - sign-in / boot with session: pull server keys into localStorage; adopt a
//   local-only key up to the server once.
// - clear: removes locally and pushes the removal.
//
// Security posture: keys are stored as given — end-to-end encryption isn't
// possible from a static frontend (there is nowhere to hide an encryption
// key). Protection is the user's login + Supabase's per-user access control on
// auth metadata + HTTPS in transit. Same tradeoff as localStorage, minus the
// disappearing act.

import { fetchAuthUser, getSession, isAuthEnabled, updateAuthUserData } from './supabaseClient';

export const VAULT_META_KEY = 'nugget_api_keys';

// Backwards-compatible localStorage names for existing keys.
const LS_NAMES = { gemini: 'gemini_key' };
function lsKey(name) {
  return LS_NAMES[name] || `nugget_vault_${name}`;
}

function lsGet(name) {
  try {
    return localStorage.getItem(lsKey(name)) || '';
  } catch {
    return '';
  }
}

function lsSet(name, value) {
  try {
    if (value) localStorage.setItem(lsKey(name), value);
    else localStorage.removeItem(lsKey(name));
  } catch {
    /* quota / privacy mode — the in-memory app state still works */
  }
}

/** Single read path: the local cache (kept in sync with the server). */
export function getVaultKey(name) {
  return lsGet(name);
}

// --- server push (debounced) -------------------------------------------------

const PUSH_DEBOUNCE_MS = 800;
const pushTimers = {};
const lastPushed = {}; // name -> value last confirmed on the server

function serverKeysFromSession() {
  const user = getSession()?.user;
  const meta = user?.user_metadata?.[VAULT_META_KEY];
  return meta && typeof meta === 'object' ? meta : {};
}

async function pushToServer(name, value) {
  if (!isAuthEnabled() || !getSession()?.access_token) return;
  if (lastPushed[name] === value) return; // already there — skip
  try {
    const current = { ...serverKeysFromSession() };
    if (value) current[name] = value;
    else delete current[name];
    await updateAuthUserData({ [VAULT_META_KEY]: current });
    lastPushed[name] = value;
  } catch {
    // Offline / expired token: the local cache still holds the value and the
    // next successful sync adopts it. Never throw from a keystroke handler.
  }
}

function schedulePush(name, value) {
  clearTimeout(pushTimers[name]);
  pushTimers[name] = setTimeout(() => pushToServer(name, value), PUSH_DEBOUNCE_MS);
}

/** For tests: run pending debounced pushes immediately. */
export function __flushVaultPushes() {
  const pending = Object.entries(pushTimers).map(([name, t]) => {
    clearTimeout(t);
    return pushToServer(name, lsGet(name));
  });
  for (const k of Object.keys(pushTimers)) delete pushTimers[k];
  return Promise.all(pending);
}

/** Write a key: local cache now, account vault (debounced) when signed in. */
export function setVaultKey(name, value) {
  const v = (value || '').trim();
  lsSet(name, v);
  if (isAuthEnabled() && getSession()?.access_token) schedulePush(name, v);
}

/** Remove a key locally and from the account vault. */
export function clearVaultKey(name) {
  setVaultKey(name, '');
}

// --- sign-in sync ------------------------------------------------------------

/**
 * Pull the account vault into the local cache. One-time adoption: a key that
 * exists only locally (e.g. typed while signed out) is pushed up once.
 * Server wins on conflict. Never throws.
 */
export async function syncVaultOnSignIn() {
  if (!isAuthEnabled() || !getSession()?.access_token) return;
  let serverKeys = {};
  try {
    const user = await fetchAuthUser();
    const meta = user?.user_metadata?.[VAULT_META_KEY];
    if (meta && typeof meta === 'object') serverKeys = meta;
  } catch {
    return; // offline — stay on the local cache
  }
  const names = new Set([...Object.keys(serverKeys), ...Object.keys(LS_NAMES)]);
  for (const name of names) {
    const serverVal = typeof serverKeys[name] === 'string' ? serverKeys[name] : '';
    const localVal = lsGet(name);
    if (serverVal) {
      if (localVal !== serverVal) lsSet(name, serverVal);
      lastPushed[name] = serverVal;
    } else if (localVal) {
      // Adopted once: push the local-only key up to the account vault.
      try {
        await updateAuthUserData({ [VAULT_META_KEY]: { ...serverKeys, [name]: localVal } });
        lastPushed[name] = localVal;
      } catch {
        /* retry on next sign-in */
      }
    } else {
      lastPushed[name] = '';
    }
  }
}
