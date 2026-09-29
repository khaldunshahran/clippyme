// Nugget — Collections store (favorites + custom clip collections).
//
// Pure localStorage persistence, keyed per signed-in user so two people on
// one machine don't share shelves. Available to every user — no paywall.
//
// Store shape:
//   {
//     favorites: { [clipKey]: clipRef },
//     collections: [ { id, name, createdAt, clips: { [clipKey]: clipRef } } ],
//   }
// A `clipRef` is a lightweight snapshot so the Favorites/Collections views
// can render cards even when the source job isn't loaded:
//   { key, jobId, index, title, score, duration, thumb }

const PREFIX = 'nugget_collections_';

export function collectionsKey(userId) {
  return PREFIX + (userId || 'anon');
}

/** Read storage into the canonical shape; corrupt/foreign data → empty store. */
export function loadCollections(userId) {
  try {
    const raw = localStorage.getItem(collectionsKey(userId));
    if (!raw) return emptyStore();
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return emptyStore();
    const favorites = parsed.favorites && typeof parsed.favorites === 'object' ? parsed.favorites : {};
    const collections = Array.isArray(parsed.collections)
      ? parsed.collections
          .filter((c) => c && typeof c.id === 'string' && typeof c.name === 'string')
          .map((c) => ({
            id: c.id,
            name: c.name,
            createdAt: Number(c.createdAt) || Date.now(),
            clips: c.clips && typeof c.clips === 'object' ? c.clips : {},
          }))
      : [];
    return { favorites, collections };
  } catch {
    return emptyStore();
  }
}

export function saveCollections(userId, store) {
  try {
    localStorage.setItem(collectionsKey(userId), JSON.stringify(store || emptyStore()));
  } catch { /* ignore quota */ }
}

export function emptyStore() {
  return { favorites: {}, collections: [] };
}

/** Stable clip identity across sessions: job + clip index. */
export function clipKeyFor(jobId, index) {
  return `${jobId || 'local'}:${index ?? 0}`;
}

/** Build a renderable snapshot from a live clip object. */
export function clipRefFromClip(jobId, clip, index) {
  return {
    key: clipKeyFor(jobId, index),
    jobId: jobId || null,
    index: index ?? 0,
    title: clip?.video_title_for_youtube_short || clip?.title || `Clip ${(index ?? 0) + 1}`,
    score: Number(clip?.viral_score) || 0,
    duration: Math.max(0, Math.round((Number(clip?.end) || 0) - (Number(clip?.start) || 0))) || null,
    thumb: clip?.thumbnail_url || clip?.poster_url || null,
  };
}

// ---------- favorites ----------

export function isFavorite(store, key) {
  return !!(store?.favorites && store.favorites[key]);
}

export function toggleFavorite(store, clipRef) {
  const next = { ...store, favorites: { ...(store?.favorites || {}) } };
  if (next.favorites[clipRef.key]) delete next.favorites[clipRef.key];
  else next.favorites[clipRef.key] = clipRef;
  return next;
}

export function favoriteList(store) {
  return Object.values(store?.favorites || {});
}

// ---------- custom collections ----------

export function newCollectionId() {
  return `col_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function createCollection(store, name) {
  const clean = String(name || '').trim().slice(0, 60);
  if (!clean) return { store, id: null };
  const col = { id: newCollectionId(), name: clean, createdAt: Date.now(), clips: {} };
  return { store: { ...store, collections: [...(store?.collections || []), col] }, id: col.id };
}

export function renameCollection(store, id, name) {
  const clean = String(name || '').trim().slice(0, 60);
  if (!clean) return store;
  return {
    ...store,
    collections: (store?.collections || []).map((c) => (c.id === id ? { ...c, name: clean } : c)),
  };
}

export function deleteCollection(store, id) {
  return { ...store, collections: (store?.collections || []).filter((c) => c.id !== id) };
}

export function addClipToCollection(store, id, clipRef) {
  return {
    ...store,
    collections: (store?.collections || []).map((c) =>
      c.id === id ? { ...c, clips: { ...(c.clips || {}), [clipRef.key]: clipRef } } : c
    ),
  };
}

export function removeClipFromCollection(store, id, key) {
  return {
    ...store,
    collections: (store?.collections || []).map((c) => {
      if (c.id !== id) return c;
      const clips = { ...(c.clips || {}) };
      delete clips[key];
      return { ...c, clips };
    }),
  };
}

/** Ids of collections that already contain this clip key. */
export function collectionsContaining(store, key) {
  return (store?.collections || []).filter((c) => c.clips && c.clips[key]).map((c) => c.id);
}
