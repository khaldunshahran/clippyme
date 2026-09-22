import { fireEvent, render, screen } from '@testing-library/react';
import { expect, test, vi, beforeEach } from 'vitest';
import { HighlightsStudioView } from './highlightsStudio';
import * as realApi from './realApi';

beforeEach(() => {
  vi.spyOn(realApi, 'getHighlights').mockResolvedValue({
    highlights: [
      {
        id: 'hl_1',
        title: 'Best Moments Micro',
        tier: 'micro',
        viral_score: 96,
        target_duration: 50,
        actual_duration: 48.2,
        aspect: '9:16',
        video_url: '/videos/job_test_1/hl_1.mp4',
        filename: 'hl_1.mp4',
        cuts: [{ role: 'hook', start: 0, end: 5, summary: 'Intro Hook' }],
      },
    ],
  });
});

test('renders Highlights Studio hero header, preset cards, and source panel', () => {
  render(<HighlightsStudioView onToast={vi.fn()} />);

  expect(screen.getByText(/Drop a link · get viral highlight reels/i)).toBeInTheDocument();
  expect(screen.getByText(/Long videos in\./i)).toBeInTheDocument();
  expect(screen.getByText(/Supercut highlights out\./i)).toBeInTheDocument();
  expect(screen.getByText(/All Tiers \(Auto\)/i)).toBeInTheDocument();
  expect(screen.getByText(/Micro-Shorts \(<60s\)/i)).toBeInTheDocument();
  expect(screen.getByText(/Story Digest \(~2m\)/i)).toBeInTheDocument();
  expect(screen.getByText(/Paste a link or drop a file/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /URL/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Upload/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Generate AI Highlights/i })).toBeInTheDocument();
});

test('switches between URL and Upload source tabs', () => {
  render(<HighlightsStudioView onToast={vi.fn()} />);

  const urlTab = screen.getByRole('button', { name: /URL/i });
  expect(urlTab).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/Paste a video link/i)).toBeInTheDocument();

  const uploadTab = screen.getByRole('button', { name: /Upload/i });
  fireEvent.click(uploadTab);
  expect(screen.getByText(/Drop a video or/i)).toBeInTheDocument();
});

test('renders recipe options panel and config drawers', () => {
  render(<HighlightsStudioView onToast={vi.fn()} />);

  expect(screen.getByRole('heading', { name: /Recipe/i })).toBeInTheDocument();
  expect(screen.getByText(/Content focus/i)).toBeInTheDocument();
  expect(screen.getByText(/Highlights generation tiers/i)).toBeInTheDocument();
  expect(screen.getByText(/Find viral moments/i)).toBeInTheDocument();
  expect(screen.getByText(/Auto face-track · Subject FrameShift crop · Off letterbox bands/i)).toBeInTheDocument();
  expect(screen.getByText(/Subtitles & Karaoke/i)).toBeInTheDocument();
});

test('picks different highlight presets', () => {
  render(<HighlightsStudioView onToast={vi.fn()} />);

  const microPreset = screen.getByText(/Micro-Shorts \(<60s\)/i);
  fireEvent.click(microPreset);
  expect(microPreset).toBeInTheDocument();
});

test('clears preset selection when recipe tier is manually adjusted', () => {
  render(<HighlightsStudioView onToast={vi.fn()} />);

  const allTiersPreset = screen.getByRole('button', { name: /All Tiers \(Auto\)/i });
  expect(allTiersPreset).toHaveAttribute('aria-pressed', 'true');

  // Change tier manually to 3-10m
  const extendedTierBtn = screen.getByRole('button', { name: '3-10m' });
  fireEvent.click(extendedTierBtn);

  // Preset card is no longer active
  expect(allTiersPreset).toHaveAttribute('aria-pressed', 'false');
});

test('restores in-flight processing session with telemetry and terminal logs', () => {
  const sessionData = {
    jobId: 'job_persist_test',
    status: 'processing',
    statusText: 'Transcribing speech with Whisper...',
    logs: [
      '[runtime] stage=transcribing cpu=25 rss_mb=450 disk_free_gb=120',
      '📥 Downloaded video successfully',
      '🎙️ Transcribing speech with Whisper...',
    ],
    timestamp: Date.now(),
  };
  localStorage.setItem('clippyme_highlights_session', JSON.stringify(sessionData));

  render(<HighlightsStudioView onToast={vi.fn()} />);

  expect(screen.getByText(/Creating highlight reels\./i)).toBeInTheDocument();
  expect(screen.getAllByText(/Transcribing speech with Whisper\.\.\./i).length).toBeGreaterThan(0);
  expect(screen.getByText(/Operations/i)).toBeInTheDocument();
  expect(screen.getByText(/25%/i)).toBeInTheDocument();
  expect(screen.getByText(/450 MB/i)).toBeInTheDocument();
  expect(screen.getByText(/highlights · console/i)).toBeInTheDocument();
  expect(screen.getByText(/📥 Downloaded video successfully/i)).toBeInTheDocument();

  // Cancel & Start Over resets session
  const cancelBtn = screen.getByRole('button', { name: /Cancel & Start Over/i });
  fireEvent.click(cancelBtn);

  expect(screen.getByText(/Drop a link · get viral highlight reels/i)).toBeInTheDocument();
  expect(localStorage.getItem('clippyme_highlights_session')).toBeNull();
});

test('shows Retry from Checkpoint button on failed session and triggers retry', () => {
  const sessionData = {
    jobId: 'job_failed_test',
    status: 'error',
    statusText: 'Subtitle burn failed on highlight reel',
    logs: [
      '✨ Synthesizing full-video multi-tier highlight reels...',
      'Subtitle burn failed on highlight reel: burn_subtitles() error',
    ],
    timestamp: Date.now(),
  };
  localStorage.setItem('clippyme_highlights_session', JSON.stringify(sessionData));

  render(<HighlightsStudioView onToast={vi.fn()} />);

  expect(screen.getByText(/Something broke\./i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Retry from Checkpoint/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Cancel & Start Over/i })).toBeInTheDocument();
});

test('renders editorial steering controls: content mode, output style, focus theme, and gap merging', () => {
  render(<HighlightsStudioView onToast={vi.fn()} />);

  expect(screen.getByText(/Content mode/i)).toBeInTheDocument();
  expect(screen.getByText(/Output style/i)).toBeInTheDocument();
  expect(screen.getByText(/Focus theme \/ topic/i)).toBeInTheDocument();
  expect(screen.getByText(/Adjacent gap merging/i)).toBeInTheDocument();

  // Change theme
  const themeInput = screen.getByLabelText(/Focus theme or topic/i);
  fireEvent.change(themeInput, { target: { value: 'Cryptocurrency and Web3 debate' } });
  expect(themeInput.value).toBe('Cryptocurrency and Web3 debate');

  // Verify theme chip is displayed in the SummaryBar
  expect(screen.getByText(/theme: "Cryptocurrency…"/i)).toBeInTheDocument();
  expect(screen.getByText(/gap merge/i)).toBeInTheDocument();
  expect(screen.getByText(/podcast mode/i)).toBeInTheDocument();
  expect(screen.getByText(/recap style/i)).toBeInTheDocument();
});


