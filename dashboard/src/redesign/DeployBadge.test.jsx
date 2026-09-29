import { afterEach, describe, expect, test, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DeployBadge } from './DeployBadge';

describe('DeployBadge', () => {
  afterEach(() => { vi.unstubAllEnvs(); });

  test('shows subtle "dev" when no deploy SHA is set', () => {
    vi.stubEnv('VITE_DEPLOY_SHA', '');
    render(<DeployBadge />);
    expect(screen.getByText('dev')).toBeTruthy();
    expect(screen.getByRole('button', { name: /Development build/ })).toBeTruthy();
  });

  test('shows the short SHA with full details on hover/tap', () => {
    vi.stubEnv('VITE_DEPLOY_SHA', 'abc1234');
    vi.stubEnv('VITE_DEPLOY_SHA_FULL', 'abc1234def5678');
    vi.stubEnv('VITE_DEPLOY_TIME', '2026-09-29 17:00 EEST');
    vi.stubEnv('VITE_DEPLOY_SUBJECT', 'Add deploy badge');
    render(<DeployBadge />);
    const btn = screen.getByRole('button', { name: /abc1234def5678/ });
    expect(screen.getByText('abc1234')).toBeTruthy();
    const title = btn.getAttribute('title');
    expect(title).toContain('abc1234def5678');
    expect(title).toContain('2026-09-29 17:00 EEST');
    expect(title).toContain('Add deploy badge');
  });

  test('tap reveals details on touch devices', () => {
    vi.stubEnv('VITE_DEPLOY_SHA', 'abc1234');
    const alert = vi.spyOn(window, 'alert').mockImplementation(() => {});
    render(<DeployBadge />);
    fireEvent.click(screen.getByRole('button'));
    expect(alert).toHaveBeenCalledTimes(1);
    alert.mockRestore();
  });
});
