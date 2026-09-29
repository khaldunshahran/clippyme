import { describe, expect, test, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SidebarProgress } from './SidebarProgress';

const activeJob = {
  status: 'processing',
  step: 'transcribing',
  logs: [],
  clipsCount: 2,
  media: { type: 'url', payload: 'https://www.youtube.com/watch?v=abc123' },
  paused: false,
};

describe('SidebarProgress', () => {
  test('hidden completely when no job is active', () => {
    const { container } = render(<SidebarProgress status="idle" />);
    expect(container.innerHTML).toBe('');
  });

  test('hidden when status is complete (no fake 100% chip)', () => {
    const { container } = render(<SidebarProgress status="complete" />);
    expect(container.innerHTML).toBe('');
  });

  test('shows job name, animated bar and percent while processing', () => {
    render(<SidebarProgress {...activeJob} onOpen={() => {}} />);
    expect(screen.getByText('youtube.com')).toBeTruthy();
    const pct = screen.getByText(/^\d+%$/);
    expect(pct).toBeTruthy();
    const bar = screen.getByRole('progressbar');
    expect(bar.getAttribute('aria-valuenow')).toBe(pct.textContent.replace('%', ''));
    expect(screen.getByText('transcribing')).toBeTruthy();
  });

  test('click opens the job', () => {
    const onOpen = vi.fn();
    render(<SidebarProgress {...activeJob} onOpen={onOpen} />);
    fireEvent.click(screen.getByRole('button'));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  test('paused phase is surfaced honestly', () => {
    render(<SidebarProgress {...activeJob} paused onOpen={() => {}} />);
    expect(screen.getByText('paused')).toBeTruthy();
  });
});
