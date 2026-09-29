import { beforeEach, describe, expect, test } from 'vitest';
import { CHANGELOG, WHATSNEW_DISMISS_KEY, latestEntryId, isWhatsNewDismissed, dismissWhatsNew } from './changelog';

beforeEach(() => { window.localStorage.clear(); });

describe('changelog', () => {
  test('entries are newest-first with unique ids and real shipped features only', () => {
    expect(CHANGELOG.length).toBeGreaterThan(0);
    const ids = new Set(CHANGELOG.map((e) => e.id));
    expect(ids.size).toBe(CHANGELOG.length);
    for (const e of CHANGELOG) {
      expect(e.title).toBeTruthy();
      expect(e.body).toBeTruthy();
      expect(Array.isArray(e.bullets)).toBe(true);
      expect(e.bullets.length).toBeGreaterThan(0);
      expect(/^\d{4}-\d{2}-\d{2}$/.test(e.date)).toBe(true);
    }
    // newest first
    const dates = CHANGELOG.map((e) => e.date);
    expect([...dates].sort().reverse()).toEqual(dates);
  });

  test('covers the parity-build features plus the real auth/infra ships', () => {
    const titles = CHANGELOG.map((e) => e.title).join(' | ');
    for (const needle of ['Collections', 'Projects', 'sidebar', 'Lower thirds', 'Deploy badge', 'Google']) {
      expect(titles).toContain(needle);
    }
  });

  test('dismissal persists per entry id in localStorage', () => {
    const id = latestEntryId();
    expect(isWhatsNewDismissed(id)).toBe(false);
    dismissWhatsNew(id);
    expect(window.localStorage.getItem(WHATSNEW_DISMISS_KEY)).toBe(id);
    expect(isWhatsNewDismissed(id)).toBe(true);
    expect(isWhatsNewDismissed('other-id')).toBe(false);
  });
});
