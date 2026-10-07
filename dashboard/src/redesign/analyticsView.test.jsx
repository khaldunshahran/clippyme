import { test, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AnalyticsView } from './analyticsView.jsx';
import * as realApi from './realApi.js';

const MOCK_SUMMARY = {
  total_published: 3,
  total_views: 230900,
  total_likes: 19000,
  total_shares: 6310,
  total_comments: 2530,
  avg_retention: 0.815,
  clips: [
    {
      clip_id: 'clip-recorded',
      title: 'Recorded retention clip',
      duration: 86.8,
      platforms: ['tiktok'],
      metrics: { views: 185500, likes: 15400, shares: 5120, comments: 2100,
                 retention_rate: 0.84, retention_recorded: true },
    },
    {
      clip_id: 'clip-zero',
      title: 'Explicit zero retention clip',
      duration: 45.2,
      platforms: ['tiktok'],
      metrics: { views: 45400, likes: 3600, shares: 1190, comments: 420,
                 retention_rate: 0.0, retention_recorded: true },
    },
    {
      clip_id: 'clip-unknown',
      title: 'Unknown retention clip',
      duration: 30.0,
      platforms: ['tiktok'],
      metrics: { views: 1000, likes: 10, shares: 2, comments: 1,
                 retention_rate: 0.0 },
    },
  ],
};

// Honest insights: null/empty causal fields, explicit observations.
const MOCK_INSIGHTS = {
  patterns: {
    has_data: true,
    sample_size: 3,
    status: 'descriptive_observations',
    story_duration_multiplier: null,
    top_duration_tier: null,
    avg_completion_rate: null,
    top_hook_patterns: [],
    recommendations: [],
    totals: { views: 230900, likes: 19000, shares: 6310, comments: 2530 },
    retention: { average_recorded: 0.42, recorded_count: 2, measured_count: 3 },
  },
  prompt_snippet: '',
};

const EMPTY_INSIGHTS = {
  patterns: {
    has_data: false,
    sample_size: 0,
    status: 'no_measurements',
    reason: 'no valid clip measurements recorded',
    story_duration_multiplier: null,
    top_duration_tier: null,
    avg_completion_rate: null,
    top_hook_patterns: [],
  },
  prompt_snippet: '',
};

beforeEach(() => {
  vi.restoreAllMocks();
});

test('renders honest evidence state, no invented learning claims', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Recorded retention clip')).toBeInTheDocument();
  });

  // Honest banner, not "AI Continuous Learning Engine / Active".
  expect(screen.getByText(/Performance Observations/i)).toBeInTheDocument();
  expect(screen.getByText(/Not Active in Selection/i)).toBeInTheDocument();
  expect(screen.queryByText(/AI Continuous Learning Engine/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/Active in Pipeline/i)).not.toBeInTheDocument();

  // No invented multiplier / hook archetypes / completion claims.
  expect(screen.queryByText(/higher retention on complete arcs/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/Winning Duration Tier/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/84%.*completion/i)).not.toBeInTheDocument();

  // Honest observations with denominators.
  expect(screen.getByText(/3 clips with recorded views/i)).toBeInTheDocument();
  expect(screen.getByText(/Average of 2 explicitly recorded values/i)).toBeInTheDocument();
});

test('distinguishes recorded zero from unknown retention', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Explicit zero retention clip')).toBeInTheDocument();
  });

  // Recorded 0% shows as 0%, unknown shows as Unknown (not a fallback %).
  const zeroRow = screen.getByText('Explicit zero retention clip').closest('div');
  expect(zeroRow.textContent).toMatch(/0%/);
  const unknownRow = screen.getByText('Unknown retention clip').closest('div');
  expect(unknownRow.textContent).toMatch(/Unknown/);
});

test('empty insights show honest inactive state', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue({ ...MOCK_SUMMARY, clips: [] });
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(EMPTY_INSIGHTS);

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText(/audience-based selection rules are not active yet/i)).toBeInTheDocument();
  });
  expect(screen.queryByText(/WHOLE-STORY/i)).not.toBeInTheDocument();
});

test('saves metrics with honest wording, no learning claim', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);
  const trackSpy = vi.spyOn(realApi, 'trackClipAnalytics').mockResolvedValue({ success: true });
  const pushToast = vi.fn();

  render(<AnalyticsView pushToast={pushToast} />);

  await waitFor(() => {
    expect(screen.getByText('Recorded retention clip')).toBeInTheDocument();
  });

  const updateButtons = screen.getAllByRole('button', { name: /update/i });
  expect(updateButtons.length).toBeGreaterThan(0);
  fireEvent.click(updateButtons[0]);

  expect(screen.getByRole('heading', { name: /update clip metrics/i })).toBeInTheDocument();
  const saveBtn = screen.getByRole('button', { name: /save metrics/i });
  fireEvent.click(saveBtn);

  await waitFor(() => {
    expect(trackSpy).toHaveBeenCalledWith(
      'clip-recorded',
      expect.objectContaining({ views: 185500, likes: 15400, shares: 5120 })
    );
    // Honest wording: saved, not "learning updated".
    expect(pushToast).toHaveBeenCalledWith('success', 'Performance metrics saved.');
    expect(pushToast).not.toHaveBeenCalledWith('success', expect.stringContaining('learning'));
  });
});

test('unknown retention not sent when untouched', async () => {
  const summaryUnknown = {
    ...MOCK_SUMMARY,
    clips: [MOCK_SUMMARY.clips[2]], // only the unknown-retention clip
  };
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(summaryUnknown);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);
  const trackSpy = vi.spyOn(realApi, 'trackClipAnalytics').mockResolvedValue({ success: true });

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Unknown retention clip')).toBeInTheDocument();
  });

  fireEvent.click(screen.getAllByRole('button', { name: /update/i })[0]);
  fireEvent.click(screen.getByRole('button', { name: /save metrics/i }));

  await waitFor(() => {
    expect(trackSpy).toHaveBeenCalled();
  });
  const sent = trackSpy.mock.calls[0][1];
  // Unknown retention must not be fabricated as a value.
  expect('retention_rate' in sent).toBe(false);
});
