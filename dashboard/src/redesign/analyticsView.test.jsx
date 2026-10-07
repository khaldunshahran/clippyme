import { test, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { AnalyticsView } from './analyticsView.jsx';
import * as realApi from './realApi.js';

// Real summary shape from get_analytics_summary (backend).
const MOCK_SUMMARY = {
  total_published: 3,
  total_views: 230900,
  total_likes: 19000,
  total_shares: 6310,
  total_comments: 2530,
  avg_retention: 0.42, // (0.84 + 0.0) / 2, unmarked zero excluded
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
      // Legacy unmarked zero: retention_rate 0.0, no marker -> Unknown.
      metrics: { views: 1000, likes: 10, shares: 2, comments: 1,
                 retention_rate: 0.0 },
    },
  ],
};

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

const MALFORMED_INSIGHTS = {
  patterns: null,
  prompt_snippet: null,
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

  expect(screen.getByText(/Performance Observations/i)).toBeInTheDocument();
  expect(screen.getByText(/Not Active in Selection/i)).toBeInTheDocument();
  expect(screen.queryByText(/AI Continuous Learning Engine/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/Active in Pipeline/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/higher retention on complete arcs/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/Winning Duration Tier/i)).not.toBeInTheDocument();
  // No universal benchmark labels.
  expect(screen.queryByText(/Average \(60-70%\)/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/Viral \(80%\+\)/i)).not.toBeInTheDocument();

  expect(screen.getByText(/3 clips with recorded views/i)).toBeInTheDocument();
  expect(screen.getByText(/Average of 2 explicitly recorded values/i)).toBeInTheDocument();
});

test('leaderboard distinguishes recorded zero from unknown', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);

  const { container } = render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Explicit zero retention clip')).toBeInTheDocument();
  });

  // Find each clip row by its title, then check the retention field within.
  const rows = container.querySelectorAll('[data-clip-row]');
  // Fallback: find by title text and traverse to row container.
  const findRow = (title) => {
    const el = screen.getByText(title);
    // Climb to the row container (grid with clip_id key).
    let node = el;
    while (node && node !== container) {
      if (node.getAttribute && node.getAttribute('data-clip-id')) return node;
      node = node.parentElement;
    }
    return el.closest('div[style*="grid"]') || el.parentElement;
  };

  const zeroRow = findRow('Explicit zero retention clip');
  expect(within(zeroRow).getByText('0%')).toBeInTheDocument();

  const unknownRow = findRow('Unknown retention clip');
  expect(within(unknownRow).getByText('Unknown')).toBeInTheDocument();

  const recordedRow = findRow('Recorded retention clip');
  expect(within(recordedRow).getByText('84%')).toBeInTheDocument();
});

test('KPI shows null average as Unknown, not 0%', async () => {
  const summaryNoRetention = { ...MOCK_SUMMARY, avg_retention: null };
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(summaryNoRetention);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Recorded retention clip')).toBeInTheDocument();
  });
  // Avg Retention card shows Unknown, not 0%.
  expect(screen.getByText('Unknown')).toBeInTheDocument();
});

test('empty insights show honest inactive state', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue({ ...MOCK_SUMMARY, clips: [] });
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(EMPTY_INSIGHTS);

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText(/audience-based selection rules are not active yet/i)).toBeInTheDocument();
  });
});

test('malformed insights handled defensively', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MALFORMED_INSIGHTS);

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Recorded retention clip')).toBeInTheDocument();
  });
  // No crash, honest fallback.
  expect(screen.getByText(/audience-based selection rules are not active yet/i)).toBeInTheDocument();
});

test('modal initializes unknown as empty, explicit zero as 0', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);
  vi.spyOn(realApi, 'trackClipAnalytics').mockResolvedValue({ success: true });

  render(<AnalyticsView pushToast={vi.fn()} />);

  await waitFor(() => {
    expect(screen.getByText('Unknown retention clip')).toBeInTheDocument();
  });

  // Open modal for unknown clip.
  const updateButtons = screen.getAllByRole('button', { name: /update/i });
  fireEvent.click(updateButtons[2]); // third clip = unknown

  const retentionInput = screen.getByLabelText(/audience retention rate/i);
  expect(retentionInput.value).toBe(''); // unknown -> empty, not 0
  expect(screen.getByText('Unknown')).toBeInTheDocument();
});

test('saves metrics with honest wording, preserves untouched retention', async () => {
  vi.spyOn(realApi, 'getAnalyticsSummary').mockResolvedValue(MOCK_SUMMARY);
  vi.spyOn(realApi, 'getAnalyticsInsights').mockResolvedValue(MOCK_INSIGHTS);
  const trackSpy = vi.spyOn(realApi, 'trackClipAnalytics').mockResolvedValue({ success: true });
  const pushToast = vi.fn();

  render(<AnalyticsView pushToast={pushToast} />);

  await waitFor(() => {
    expect(screen.getByText('Unknown retention clip')).toBeInTheDocument();
  });

  fireEvent.click(screen.getAllByRole('button', { name: /update/i })[2]);
  fireEvent.click(screen.getByRole('button', { name: /save metrics/i }));

  await waitFor(() => {
    expect(trackSpy).toHaveBeenCalled();
  });
  const sent = trackSpy.mock.calls[0][1];
  // Untouched unknown retention is omitted, not fabricated.
  expect('retention_rate' in sent).toBe(false);
  expect(pushToast).toHaveBeenCalledWith('success', 'Performance metrics saved.');
});
