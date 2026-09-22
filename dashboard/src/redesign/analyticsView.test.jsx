import { test, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AnalyticsView } from './analyticsView.jsx';
import * as realApi from './realApi.js';

const MOCK_SUMMARY = {
  total_published: 2,
  total_views: 230900,
  total_likes: 19000,
  total_shares: 6310,
  total_comments: 2530,
  avg_retention: 0.815,
  platform_breakdown: {
    tiktok: { views: 230900, shares: 6310, likes: 19000, posts: 2 },
    instagram: { views: 142500, shares: 4210, likes: 11800, posts: 1 },
    youtube: { views: 88400, shares: 2100, likes: 7200, posts: 1 },
  },
  clips: [
    {
      clip_id: 'f797246f-c1',
      title: 'Rudy Giuliani asked him for 2 MILLION dollars and he laughed in his face',
      duration: 86.8,
      narrative_tier: 'mid (60s–120s)',
      platforms: ['tiktok', 'instagram', 'youtube'],
      views: 185500,
      likes: 15400,
      shares: 5120,
      comments: 2100,
      retention_rate: 0.84,
    },
    {
      clip_id: 'f797246f-c2',
      title: 'The Hidden Formula Behind AI Video Models',
      duration: 45.2,
      narrative_tier: 'short (30s–60s)',
      platforms: ['tiktok'],
      views: 45400,
      likes: 3600,
      shares: 1190,
      comments: 430,
      retention_rate: 0.79,
    },
  ],
};

const MOCK_INSIGHTS = {
  patterns: {
    has_data: true,
    sample_size: 2,
    story_duration_multiplier: 8.0,
    top_duration_tier: 'mid (60s–120s)',
    avg_completion_rate: 0.84,
    top_hook_patterns: [
      'Stakes & Consequence',
      'Contradiction / Proof',
    ],
  },
  prompt_snippet: '## LEARNED VIRAL PERFORMANCE PATTERNS\n- Favor mid (60s-120s) for complete narrative arcs.',
};

beforeEach(() => {
  vi.restoreAllMocks();
});

test('renders AnalyticsView with KPI cards and leaderboard', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);

  render(<AnalyticsView pushToast={vi.fn()} />);

  expect(screen.getByText('Loading analytics leaderboard…')).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByText('Rudy Giuliani asked him for 2 MILLION dollars and he laughed in his face')).toBeInTheDocument();
  });

  // KPI cards
  expect(screen.getByText('230,900')).toBeInTheDocument();
  expect(screen.getByText('6,310')).toBeInTheDocument();
  expect(screen.getByText('82%')).toBeInTheDocument(); // avg_retention 0.815 rounded to 82%

  // Continuous Learning Engine banner
  expect(screen.getByText(/AI Continuous Learning Engine/i)).toBeInTheDocument();
  expect(screen.getByText(/8x higher retention on complete arcs vs fragments/i)).toBeInTheDocument();
  expect(screen.getAllByText('mid (60s–120s)').length).toBeGreaterThanOrEqual(1);
});

test('opens Update Metrics modal and saves metrics', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);
  const trackSpy = vi.spyOn(realApi, 'trackClipAnalytics').mockResolvedValue({ success: true });
  const pushToast = vi.fn();

  render(<AnalyticsView pushToast={pushToast} />);

  await waitFor(() => {
    expect(screen.getByText('Rudy Giuliani asked him for 2 MILLION dollars and he laughed in his face')).toBeInTheDocument();
  });

  const updateButtons = screen.getAllByRole('button', { name: /update/i });
  expect(updateButtons.length).toBeGreaterThan(0);
  fireEvent.click(updateButtons[0]);

  // Modal appears
  expect(screen.getByRole('heading', { name: /update clip metrics/i })).toBeInTheDocument();
  const saveBtn = screen.getByRole('button', { name: /save metrics/i });
  fireEvent.click(saveBtn);

  await waitFor(() => {
    expect(trackSpy).toHaveBeenCalledWith(
      'f797246f-c1',
      expect.objectContaining({
        views: 185500,
        likes: 15400,
        shares: 5120,
      })
    );
    expect(pushToast).toHaveBeenCalledWith('success', expect.stringContaining('AI continuous learning updated'));
  });
});
