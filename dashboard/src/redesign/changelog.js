// Nugget — What's New changelog.
//
// ONLY real, shipped features go here. Every entry must describe something
// the user can actually do in the product right now. Never add aspirational
// or planned items.

export const CHANGELOG = [
  {
    id: 'collections-2026-09-29',
    date: '2026-09-29',
    title: 'Collections: favorites + your own shelves',
    tag: 'New',
    body: 'Save the clips you love and organise them however you like.',
    bullets: [
      'Tap the heart on any clip card to favorite it — the Favorites view keeps them all in one place.',
      'Create your own collections, rename or delete them, and add or remove clips from any card menu.',
      'Collections are saved on this device for your account — no paywall, no limits.',
    ],
  },
  {
    id: 'projects-2026-09-29',
    date: '2026-09-29',
    title: 'Projects: your source videos, grouped',
    tag: 'New',
    body: 'Every video you clip from now has a home.',
    bullets: [
      'The new Projects tab groups your clips by the source video they came from.',
      'Each project card shows the title, clip count, date and job status.',
      'Open a project to jump straight back into its clips.',
    ],
  },
  {
    id: 'sidebar-progress-2026-09-29',
    date: '2026-09-29',
    title: 'Live generation progress in the sidebar',
    tag: 'New',
    body: 'Watch your render without babysitting the Create tab.',
    bullets: [
      'While a job is running, the sidebar shows its name, an animated progress bar and the live percent.',
      'Click it to jump back to the job. It hides itself the moment nothing is rendering.',
    ],
  },
  {
    id: 'lower-thirds-2026-09-29',
    date: '2026-09-29',
    title: 'Lower thirds: broadcast-style name tags',
    tag: 'New',
    body: 'Design on-screen name tags and location titles before you export.',
    bullets: [
      'Six broadcast presets — name-tag slide-in, location fade, minimal bar, bold color block and more — each with color variants.',
      'Edit the name and title lines, pick a font, and set start/end timing with a live preview.',
      'Burn-in rendering is still being added to the backend — the control says so plainly until it lands.',
    ],
  },
  {
    id: 'deploy-badge-2026-09-29',
    date: '2026-09-29',
    title: 'Deploy badge in the sidebar',
    tag: 'New',
    body: 'Always know exactly which build you are looking at.',
    bullets: [
      'The sidebar footer shows the deployed git short SHA (or "dev" locally).',
      'Hover or tap for the full SHA, commit time and message.',
    ],
  },
  {
    id: 'google-signin-2026-09-29',
    date: '2026-09-29',
    title: 'Sign in with Google',
    tag: 'Auth',
    body: 'One-click sign-in is here.',
    bullets: [
      'Use “Continue with Google” instead of typing an email and password.',
      'Your clips and settings still belong to your account either way.',
    ],
  },
  {
    id: 'supabase-project-2026-09-29',
    date: '2026-09-29',
    title: 'Nugget gets its own backend project',
    tag: 'Infra',
    body: 'Your data now lives in Nugget\u2019s own dedicated Supabase project.',
    bullets: [
      'Authentication, accounts and settings run on infrastructure provisioned just for Nugget.',
      'Faster sign-in, and room to grow as the product does.',
    ],
  },
];

export const WHATSNEW_DISMISS_KEY = 'nugget_whatsnew_dismissed';

/** The newest entry id — the banner shows it until dismissed. */
export function latestEntryId() {
  return CHANGELOG.length ? CHANGELOG[0].id : null;
}

export function isWhatsNewDismissed(entryId) {
  try {
    return localStorage.getItem(WHATSNEW_DISMISS_KEY) === entryId;
  } catch {
    return false;
  }
}

export function dismissWhatsNew(entryId) {
  try {
    localStorage.setItem(WHATSNEW_DISMISS_KEY, entryId);
  } catch { /* ignore */ }
}
