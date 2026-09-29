import { beforeEach, describe, expect, test, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { WhatsNew, WhatsNewBanner, WhatsNewModal } from './WhatsNew';
import { CHANGELOG, WHATSNEW_DISMISS_KEY } from './changelog';

beforeEach(() => { window.localStorage.clear(); });

describe('WhatsNewBanner', () => {
  test('shows the newest entry with actions', () => {
    const entry = CHANGELOG[0];
    render(<WhatsNewBanner entry={entry} onDismiss={() => {}} onOpenLog={() => {}} />);
    expect(screen.getByText(entry.title)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'See all updates' })).toBeTruthy();
    expect(screen.getByRole('button', { name: "Dismiss what's new" })).toBeTruthy();
  });

  test('renders nothing without an entry', () => {
    const { container } = render(<WhatsNewBanner entry={null} onDismiss={() => {}} onOpenLog={() => {}} />);
    expect(container.innerHTML).toBe('');
  });
});

describe('WhatsNew', () => {
  test('banner shows until dismissed, then stays dismissed', () => {
    const { unmount } = render(<WhatsNew />);
    expect(screen.getByText(CHANGELOG[0].title)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: "Dismiss what's new" }));
    expect(screen.queryByText(CHANGELOG[0].title)).toBeNull();
    unmount();
    render(<WhatsNew />);
    expect(screen.queryByText(CHANGELOG[0].title)).toBeNull();
    expect(window.localStorage.getItem(WHATSNEW_DISMISS_KEY)).toBe(CHANGELOG[0].id);
  });

  test('opening the log shows every changelog entry', () => {
    render(<WhatsNew />);
    fireEvent.click(screen.getByRole('button', { name: 'See all updates' }));
    for (const e of CHANGELOG) {
      // the banner (newest entry) stays mounted behind the modal
      expect(screen.getAllByText(e.title).length).toBeGreaterThanOrEqual(1);
    }
  });
});

describe('WhatsNewModal', () => {
  test('closes via the Done button', () => {
    const onClose = vi.fn();
    render(<WhatsNewModal onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
