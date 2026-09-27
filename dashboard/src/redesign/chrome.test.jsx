import { render, screen, fireEvent } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { SidebarNav, MobileNav, MoreSheet, NotFoundPage, NAV_GROUPS, GROW_TABS } from './chrome';

test('sidebar renders the grouped SaaS navigation', () => {
  render(<SidebarNav tab="home" goTab={() => {}} />);
  // Ungrouped primaries
  for (const label of ['Home', 'Create', 'Clips']) {
    expect(screen.getByRole('button', { name: label })).toBeTruthy();
  }
  // Grow group
  for (const label of ['Trend Radar', 'Channels', 'Analytics', 'Live Monitor']) {
    expect(screen.getByRole('button', { name: label })).toBeTruthy();
  }
  // Library group + settings
  for (const label of ['History', 'Highlights', 'Settings']) {
    expect(screen.getByRole('button', { name: label })).toBeTruthy();
  }
});

test('sidebar marks the active tab with aria-current', () => {
  render(<SidebarNav tab="analytics" goTab={() => {}} />);
  expect(screen.getByRole('button', { name: 'Analytics' }).getAttribute('aria-current')).toBe('page');
  expect(screen.getByRole('button', { name: 'Home' }).getAttribute('aria-current')).toBeNull();
});

test('sidebar navigates and offers sign-in / view-website', () => {
  const goTab = vi.fn();
  const onOpenSite = vi.fn();
  const onSignIn = vi.fn();
  render(<SidebarNav tab="home" goTab={goTab} onOpenSite={onOpenSite} onSignIn={onSignIn} />);
  fireEvent.click(screen.getByRole('button', { name: 'Clips' }));
  expect(goTab).toHaveBeenCalledWith('clips');
  fireEvent.click(screen.getByRole('button', { name: 'View website' }));
  expect(onOpenSite).toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
  expect(onSignIn).toHaveBeenCalled();
});

test('mobile nav has five primaries; Grow routes to trends, More opens the sheet', () => {
  const goTab = vi.fn();
  const onMore = vi.fn();
  render(<MobileNav tab="home" goTab={goTab} onMore={onMore} />);
  for (const label of ['Home', 'Create', 'Clips', 'Grow']) {
    expect(screen.getByRole('button', { name: label })).toBeTruthy();
  }
  const moreBtn = screen.getByRole('button', { name: 'More destinations' });
  expect(moreBtn).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Grow' }));
  expect(goTab).toHaveBeenCalledWith('trends');
  fireEvent.click(moreBtn);
  expect(onMore).toHaveBeenCalled();
});

test('mobile nav marks a grow tab active on the Grow item', () => {
  render(<MobileNav tab="channels" goTab={() => {}} onMore={() => {}} />);
  expect(screen.getByRole('button', { name: 'Grow' }).getAttribute('aria-current')).toBe('page');
});

test('more sheet exposes every destination and closes', () => {
  const goTab = vi.fn();
  const onClose = vi.fn();
  render(<MoreSheet tab="home" goTab={goTab} onClose={onClose} />);
  const allIds = NAV_GROUPS.flatMap((g) => g.items.map((i) => i.label));
  for (const label of [...allIds, 'Settings']) {
    expect(screen.getByRole('button', { name: label })).toBeTruthy();
  }
  fireEvent.click(screen.getByRole('button', { name: 'Highlights' }));
  expect(goTab).toHaveBeenCalledWith('highlights');
  expect(onClose).toHaveBeenCalled();
  fireEvent.keyDown(window, { key: 'Escape' });
  expect(onClose).toHaveBeenCalledTimes(2);
});

test('404 page offers working ways back', () => {
  const onHome = vi.fn();
  const onCreate = vi.fn();
  render(<NotFoundPage onHome={onHome} onCreate={onCreate} />);
  expect(screen.getByText('This page got clipped.')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Back to Home' }));
  expect(onHome).toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Create a clip' }));
  expect(onCreate).toHaveBeenCalled();
});

test('grow tab set covers the sidebar grow group', () => {
  const growIds = NAV_GROUPS.find((g) => g.label === 'Grow').items.map((i) => i.id);
  expect([...GROW_TABS].sort()).toEqual([...growIds].sort());
});
