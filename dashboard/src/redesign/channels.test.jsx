import { test, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChannelsView } from './channels.jsx';
import * as realApi from './realApi.js';

const MOCK_CHANNELS = [
  {
    id: 'ch_us_politics',
    name: 'US Politics Daily',
    description: 'High-stakes congressional debates, election updates, and political commentary for US audiences.',
    niches: ['politics', 'nation'],
    default_preset: 'viral',
    banner_platform: 'youtube',
    banner_handle: '@USPoliticsDaily',
    banner_y_pct: 0.85,
    reframe_mode: 'auto',
    sub_preset: 'hormozi_bold',
    publishing_targets: { youtube: true, tiktok: true, instagram: false },
  },
  {
    id: 'ch_breaking_world',
    name: 'Global Breaking & Crisis',
    description: 'Rapid response coverage on breaking world events and disasters.',
    niches: ['breaking_world', 'world'],
    default_preset: 'viral',
    banner_platform: 'tiktok',
    banner_handle: '@GlobalBreakingNow',
    banner_y_pct: 0.85,
    reframe_mode: 'auto',
    sub_preset: 'classic',
    publishing_targets: { youtube: true, tiktok: true, instagram: true },
  },
];

beforeEach(() => {
  vi.restoreAllMocks();
});

test('renders channel cards and active count', async () => {
  vi.spyOn(realApi, 'getChannels').mockResolvedValue({
    channels: MOCK_CHANNELS,
    total: 2,
  });

  render(<ChannelsView pushToast={vi.fn()} />);

  expect(screen.getByText('Loading channel profiles...')).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByText('US Politics Daily')).toBeInTheDocument();
  });

  expect(screen.getByText('Global Breaking & Crisis')).toBeInTheDocument();
  expect(screen.getByText('@USPoliticsDaily')).toBeInTheDocument();
  expect(screen.getByText('@GlobalBreakingNow')).toBeInTheDocument();
  expect(screen.getByText('#politics')).toBeInTheDocument();
  expect(screen.getByText('#breaking_world')).toBeInTheDocument();
  expect(screen.getByText('2 Active Channels')).toBeInTheDocument();
});

test('opens create channel modal and submits new channel', async () => {
  vi.spyOn(realApi, 'getChannels').mockResolvedValue({
    channels: MOCK_CHANNELS,
    total: 2,
  });
  const createSpy = vi.spyOn(realApi, 'createChannel').mockResolvedValue({
    channel: { id: 'ch_crypto', name: 'Crypto Pulse' },
  });
  const toastMock = vi.fn();

  render(<ChannelsView pushToast={toastMock} />);

  await waitFor(() => {
    expect(screen.getByText('US Politics Daily')).toBeInTheDocument();
  });

  // Click "+ New Channel"
  fireEvent.click(screen.getByRole('button', { name: /\+ New Channel/i }));

  expect(screen.getByText('Add New Channel Profile')).toBeInTheDocument();

  // Fill in form
  fireEvent.change(screen.getByLabelText(/Channel Name \*/i), {
    target: { value: 'Crypto Pulse' },
  });
  fireEvent.change(screen.getByLabelText(/Topic Niches & Matching Keywords/i), {
    target: { value: 'crypto, bitcoin, web3' },
  });
  fireEvent.change(screen.getByLabelText(/Channel Handle/i), {
    target: { value: '@CryptoPulse' },
  });

  // Submit
  fireEvent.click(screen.getByRole('button', { name: 'Create Channel' }));

  await waitFor(() => {
    expect(createSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'Crypto Pulse',
        niches: ['crypto', 'bitcoin', 'web3'],
        banner_handle: '@CryptoPulse',
      })
    );
  });

  expect(toastMock).toHaveBeenCalledWith('success', expect.stringContaining('Crypto Pulse'));
});

test('opens edit modal and updates an existing channel', async () => {
  vi.spyOn(realApi, 'getChannels').mockResolvedValue({
    channels: MOCK_CHANNELS,
    total: 2,
  });
  const updateSpy = vi.spyOn(realApi, 'updateChannel').mockResolvedValue({
    channel: { ...MOCK_CHANNELS[0], name: 'US Politics Live' },
  });
  const toastMock = vi.fn();

  render(<ChannelsView pushToast={toastMock} />);

  await waitFor(() => {
    expect(screen.getByText('US Politics Daily')).toBeInTheDocument();
  });

  const editButtons = screen.getAllByRole('button', { name: /Edit Channel/i });
  fireEvent.click(editButtons[0]);

  expect(screen.getByText('Edit Channel: US Politics Daily')).toBeInTheDocument();

  // Change name
  fireEvent.change(screen.getByLabelText(/Channel Name \*/i), {
    target: { value: 'US Politics Live' },
  });

  fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

  await waitFor(() => {
    expect(updateSpy).toHaveBeenCalledWith(
      'ch_us_politics',
      expect.objectContaining({
        name: 'US Politics Live',
      })
    );
  });

  expect(toastMock).toHaveBeenCalledWith('success', expect.stringContaining('US Politics Live'));
});

test('handles channel deletion with confirmation', async () => {
  vi.spyOn(realApi, 'getChannels').mockResolvedValue({
    channels: MOCK_CHANNELS,
    total: 2,
  });
  const deleteSpy = vi.spyOn(realApi, 'deleteChannel').mockResolvedValue({
    success: true,
  });
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  const toastMock = vi.fn();

  render(<ChannelsView pushToast={toastMock} />);

  await waitFor(() => {
    expect(screen.getByText('US Politics Daily')).toBeInTheDocument();
  });

  const deleteButtons = screen.getAllByTitle('Delete channel');
  fireEvent.click(deleteButtons[0]);

  await waitFor(() => {
    expect(deleteSpy).toHaveBeenCalledWith('ch_us_politics');
  });

  expect(toastMock).toHaveBeenCalledWith('info', expect.stringContaining('deleted'));
});

test('displays empty state when no channels exist', async () => {
  vi.spyOn(realApi, 'getChannels').mockResolvedValue({
    channels: [],
    total: 0,
  });

  render(<ChannelsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('No Channels Found')).toBeInTheDocument();
  });
});
