// Shared option lists for the chat UI. Caption style ids MUST match the
// backend/compose-known subtitle presets (see editor/data.js SUBTITLE_PRESETS)
// — unknown styles are rejected with 400 by /api/process.

export const CAPTION_OPTIONS = [
  { id: '', label: 'Auto (default)' },
  { id: 'hormozi_bold', label: 'Hormozi' },
  { id: 'mrbeast_box', label: 'MrBeast' },
  { id: 'neon_glow', label: 'Neon' },
  { id: 'minimal_clean', label: 'Minimal' },
  { id: 'classic_white', label: 'Classic' },
  { id: 'fire_impact', label: 'Fire' },
];

export const captionLabel = (id) =>
  (CAPTION_OPTIONS.find((o) => o.id === id) || {}).label || id || 'Auto';

export const MODEL_OPTIONS = [
  { id: '', label: 'Auto (recommended)' },
  { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
];

export const GENRE_OPTIONS = [
  { id: '', label: 'Auto-detect' },
  { id: 'podcast', label: 'Podcast' },
  { id: 'interview', label: 'Interview' },
  { id: 'vlog', label: 'Vlog' },
  { id: 'gaming', label: 'Gaming' },
  { id: 'sports', label: 'Sports' },
  { id: 'comedy', label: 'Comedy' },
  { id: 'education', label: 'Education' },
  { id: 'business', label: 'Business' },
  { id: 'stream', label: 'Stream' },
];

export const STYLE_OPTIONS = [
  { id: '', label: 'Auto' },
  { id: 'viral', label: 'Viral punchy' },
  { id: 'story', label: 'Story-driven' },
  { id: 'highlights', label: 'Best moments' },
  { id: 'educational', label: 'Key insights' },
];

export const ASPECT_OPTIONS = ['9:16', '1:1', '16:9'];

// Clip-length tiers mirror the system's duration modes (old dashboard
// create.jsx + backend min/max_duration): mix, <60s, 1-3m, 3-10m, custom.
export const LENGTH_OPTIONS = [
  { id: 'all', label: 'Mix / All' },
  { id: 'shorts', label: '<60s' },
  { id: 'mid', label: '1–3m' },
  { id: 'long', label: '3–10m' },
  { id: 'custom', label: 'Custom' },
];

export const CAPTION_DEFAULT_LS_KEY = 'nugget-caption-default-v1';

export function parseTime(str) {
  // "m:ss" or "h:mm:ss" -> seconds
  if (str == null || str === '') return null;
  const parts = String(str).trim().split(':').map(Number);
  if (parts.some((n) => !isFinite(n) || n < 0)) return null;
  let s = 0;
  for (const p of parts) s = s * 60 + p;
  return s;
}

export function formatTime(sec) {
  if (sec == null || !isFinite(sec)) return '';
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
    : `${m}:${String(r).padStart(2, '0')}`;
}
