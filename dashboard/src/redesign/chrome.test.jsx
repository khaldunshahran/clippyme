import { cleanup, render, screen, fireEvent } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import { TopNav, Hero } from './chrome';

afterEach(() => cleanup());

test('TopNav renders both desktop and mobile navigation with all 5 tabs', () => {
  const setTab = vi.fn();
  render(<TopNav tab="create" setTab={setTab} busy={false} />);

  // Desktop navigation exists with aria-label
  const primaryNav = screen.getByRole('navigation', { name: /primary navigation/i });
  expect(primaryNav).toBeInTheDocument();

  // Mobile navigation exists with aria-label
  const mobileNav = screen.getByRole('navigation', { name: /mobile navigation/i });
  expect(mobileNav).toBeInTheDocument();

  // Both contain all 8 tabs
  const desktopButtons = primaryNav.querySelectorAll('button');
  expect(desktopButtons).toHaveLength(8);
  const activeDesktopTab = primaryNav.querySelector('button[aria-current="page"]');
  expect(activeDesktopTab).toBeInTheDocument();

  // Check mobile tabs
  const mobileButtons = mobileNav.querySelectorAll('button');
  expect(mobileButtons).toHaveLength(8);

  // Active tab in mobile nav has aria-current="page" and active class
  const activeMobileTab = mobileNav.querySelector('button[aria-current="page"]');
  expect(activeMobileTab).toHaveTextContent(/create/i);
  expect(activeMobileTab.className).toContain('active');

  // Clicking a mobile tab calls setTab
  const settingsBtn = Array.from(mobileButtons).find((b) => /settings/i.test(b.textContent));
  expect(settingsBtn).toBeDefined();
  fireEvent.click(settingsBtn);
  expect(setTab).toHaveBeenCalledWith('settings');
});

test('Hero renders title, gradient, and eyebrow', () => {
  render(<Hero eyebrow="Welcome" line1="Create viral" grad="Shorts" sub="Subtitle text" />);
  expect(screen.getByText('Welcome')).toBeInTheDocument();
  expect(screen.getByText(/Create viral/)).toBeInTheDocument();
  expect(screen.getByText('Shorts')).toBeInTheDocument();
  expect(screen.getByText('Subtitle text')).toBeInTheDocument();
});

test('TopNav renders Sign in button when logged out and triggers onSignIn', () => {
  const onSignIn = vi.fn();
  render(<TopNav tab="create" setTab={vi.fn()} busy={false} user={null} onSignIn={onSignIn} />);

  const signInBtn = screen.getByRole('button', { name: /sign in/i });
  expect(signInBtn).toBeInTheDocument();
  fireEvent.click(signInBtn);
  expect(onSignIn).toHaveBeenCalledTimes(1);
});

test('TopNav renders user avatar and sign out button when logged in', () => {
  const onSignOut = vi.fn();
  const user = { email: 'alex@example.com' };
  render(<TopNav tab="create" setTab={vi.fn()} busy={false} user={user} onSignOut={onSignOut} />);

  expect(screen.getByText('AL')).toBeInTheDocument();
  const userBtn = screen.getByRole('button', { name: /user account/i });
  expect(userBtn).toBeInTheDocument();
  fireEvent.click(userBtn);
  expect(onSignOut).toHaveBeenCalledTimes(1);
});

