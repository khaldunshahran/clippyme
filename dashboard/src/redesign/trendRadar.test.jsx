import { test, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrendRadarView } from './trendRadar.jsx';
import * as realApi from './realApi.js';

const MOCK_TOPICS = [
  {
    id: 'tr_1',
    topic_title: 'Senate Debate on AI Safety Regulation',
    category: 'politics',
    virality_score: 95,
    viral_hook: 'Heated confrontation between senators and tech executives.',
    matched_video: {
      video_id: 'vid1',
      video_url: 'https://www.youtube.com/watch?v=vid1',
      title: 'Full Senate Hearing: AI Regulation 2026',
      channel: 'C-SPAN',
      duration: 3600,
      duration_string: '1:00:00',
      thumbnail_url: 'https://i.ytimg.com/vi/vid1/hqdefault.jpg',
      view_count: 50000,
    },
    clipped: false,
  },
  {
    id: 'tr_2',
    topic_title: 'Massive Earthquake in Nepal',
    category: 'breaking_world',
    virality_score: 91,
    viral_hook: 'Rescue operations captured on live footage.',
    matched_video: {
      video_id: 'vid2',
      video_url: 'https://www.youtube.com/watch?v=vid2',
      title: 'Nepal Disaster Briefing and Live Updates',
      channel: 'World News',
      duration: 1200,
      duration_string: '20:00',
      thumbnail_url: 'https://i.ytimg.com/vi/vid2/hqdefault.jpg',
      view_count: 120000,
    },
    clipped: false,
  },
];

beforeEach(() => {
  vi.restoreAllMocks();
});

test('renders trending topics and handles category filtering', async () => {
  vi.spyOn(realApi, 'getTrends').mockResolvedValue({
    topics: MOCK_TOPICS,
    last_scanned: '2026-09-16T01:00:00Z',
  });

  render(<TrendRadarView pushToast={vi.fn()} />);

  expect(screen.getByText('Scanning US news feeds & search trends...')).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByText('Senate Debate on AI Safety Regulation')).toBeInTheDocument();
  });

  expect(screen.getByText('Massive Earthquake in Nepal')).toBeInTheDocument();
  expect(screen.getByText('95% Viral Potential')).toBeInTheDocument();
  expect(screen.getByText('C-SPAN')).toBeInTheDocument();

  // Filter to politics only
  fireEvent.click(screen.getByRole('button', { name: 'US Politics' }));
  expect(screen.getByText('Senate Debate on AI Safety Regulation')).toBeInTheDocument();
  expect(screen.queryByText('Massive Earthquake in Nepal')).not.toBeInTheDocument();
});

test('1-Click Clip triggers clipping submission and calls onStartClip', async () => {
  vi.spyOn(realApi, 'getTrends').mockResolvedValue({
    topics: MOCK_TOPICS,
    last_scanned: '2026-09-16T01:00:00Z',
  });
  const clipSpy = vi.spyOn(realApi, 'clipTrendVideo').mockResolvedValue({
    status: 'queued',
    job_id: 'job_clip_test_123',
  });

  const onStartClip = vi.fn();
  const pushToast = vi.fn();

  render(
    <TrendRadarView
      apiKey="test-key"
      onStartClip={onStartClip}
      pushToast={pushToast}
    />
  );

  await waitFor(() => {
    expect(screen.getByText('Senate Debate on AI Safety Regulation')).toBeInTheDocument();
  });

  const clipButtons = screen.getAllByRole('button', { name: '⚡ 1-Click Clip' });
  fireEvent.click(clipButtons[0]);

  await waitFor(() => {
    expect(clipSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        topicId: 'tr_1',
        videoUrl: 'https://www.youtube.com/watch?v=vid1',
      }),
      'test-key'
    );
  });

  expect(onStartClip).toHaveBeenCalledWith('job_clip_test_123', 'https://www.youtube.com/watch?v=vid1');
  expect(pushToast).toHaveBeenCalledWith('success', expect.stringContaining('enqueued'));
});

test('customize button calls onCustomizeClip with topic parameters', async () => {
  vi.spyOn(realApi, 'getTrends').mockResolvedValue({
    topics: MOCK_TOPICS,
    last_scanned: '2026-09-16T01:00:00Z',
  });

  const onCustomizeClip = vi.fn();
  render(<TrendRadarView onCustomizeClip={onCustomizeClip} pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Senate Debate on AI Safety Regulation')).toBeInTheDocument();
  });

  const customizeButtons = screen.getAllByRole('button', { name: 'Customize' });
  fireEvent.click(customizeButtons[0]);

  expect(onCustomizeClip).toHaveBeenCalledWith(
    expect.objectContaining({
      url: 'https://www.youtube.com/watch?v=vid1',
      instructions: expect.stringContaining('Senate Debate'),
    })
  );
});

test('scan now button triggers research scan', async () => {
  vi.spyOn(realApi, 'getTrends').mockResolvedValue({ topics: [], last_scanned: null });
  const scanSpy = vi.spyOn(realApi, 'triggerTrendScan').mockResolvedValue({
    topics: MOCK_TOPICS,
    topics_count: 2,
    last_scanned: '2026-09-16T02:00:00Z',
  });

  const pushToast = vi.fn();
  render(<TrendRadarView apiKey="key123" pushToast={pushToast} />);

  await waitFor(() => {
    expect(screen.getByText('No trending topics found')).toBeInTheDocument();
  });

  const scanBtn = screen.getByRole('button', { name: 'Scan US Trends Now' });
  fireEvent.click(scanBtn);

  await waitFor(() => {
    expect(scanSpy).toHaveBeenCalledWith('key123');
  });

  expect(pushToast).toHaveBeenCalledWith('success', expect.stringContaining('Trend scan complete'));
});

test('renders channel selector and routes 1-click clip to chosen channel', async () => {
  vi.spyOn(realApi, 'getTrends').mockResolvedValue({
    topics: MOCK_TOPICS,
    last_scanned: '2026-09-16T01:00:00Z',
  });
  vi.spyOn(realApi, 'getChannels').mockResolvedValue({
    channels: [
      { id: 'ch_us_politics', name: 'US Politics Daily', niches: ['politics'] },
      { id: 'ch_breaking_world', name: 'Global Breaking & Crisis', niches: ['breaking_world'] },
    ],
    total: 2,
  });
  const clipSpy = vi.spyOn(realApi, 'clipTrendVideo').mockResolvedValue({
    status: 'queued',
    job_id: 'job_clip_channel_123',
  });

  render(<TrendRadarView apiKey="key123" pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Senate Debate on AI Safety Regulation')).toBeInTheDocument();
  });

  const channelSelectors = screen.getAllByLabelText('Target Channel');
  expect(channelSelectors.length).toBeGreaterThan(0);

  // Switch the target channel on the first card to 'Global Breaking & Crisis'
  fireEvent.change(channelSelectors[0], { target: { value: 'ch_breaking_world' } });

  const clipButtons = screen.getAllByRole('button', { name: '⚡ 1-Click Clip' });
  fireEvent.click(clipButtons[0]);

  await waitFor(() => {
    expect(clipSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        topicId: 'tr_1',
        channelId: 'ch_breaking_world',
      }),
      'key123'
    );
  });
});
