/**
 * Batch poller behaviour in useJobSubmission (the resilient rewrite).
 *
 * Pins the three properties the old setInterval implementation lacked:
 * no overlapping rounds, termination (with a visible message) when the
 * backend dies, and onBatchFinished when every job reaches terminal state.
 */
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('../lib/api', () => ({
  submitProcessJob: vi.fn(),
  submitBatchJob: vi.fn(),
}));
vi.mock('../lib/apiToken', () => ({
  apiFetch: vi.fn(),
}));

import { submitBatchJob, submitProcessJob } from '../lib/api';
import { apiFetch } from '../lib/apiToken';
import { useJobSubmission } from './useJobSubmission';

function mountHook(overrides = {}) {
  const props = {
    apiKey: 'k',
    setShowKeyModal: vi.fn(),
    setStatus: vi.fn(),
    setLogs: vi.fn(),
    setResults: vi.fn(),
    setProcessingMedia: vi.fn(),
    setPreselections: vi.fn(),
    setJobId: vi.fn(),
    onBatchFinished: vi.fn(),
    ...overrides,
  };
  const { result, unmount } = renderHook(() => useJobSubmission(props));
  return { handlers: result.current, props, unmount };
}

function okStatus(status) {
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ status, logs: [`${status} log`] }),
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  submitBatchJob.mockReset();
  submitProcessJob.mockReset();
  submitBatchJob.mockResolvedValue({
    jobs: [{ job_id: 'job-a' }, { job_id: 'job-b' }],
    total: 2,
  });
  submitProcessJob.mockResolvedValue({ job_id: 'job-1' });
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

test('all jobs terminal → status complete and onBatchFinished fires once', async () => {
  apiFetch.mockImplementation(() => okStatus('completed'));
  const { handlers, props } = mountHook();

  await act(() => handlers.handleBatchProcess({ urls: ['u1', 'u2'] }));
  await act(() => vi.advanceTimersByTimeAsync(2000)); // first poll round

  expect(props.setStatus).toHaveBeenLastCalledWith('complete');
  expect(props.onBatchFinished).toHaveBeenCalledTimes(1);
  expect(props.onBatchFinished).toHaveBeenCalledWith({
    jobIds: ['job-a', 'job-b'],
    succeeded: 2,
    failed: 0,
    total: 2,
  });
  expect(vi.getTimerCount()).toBe(0); // loop fully stopped
});

test('cancelled/failed jobs count as failures, not hangs', async () => {
  apiFetch
    .mockImplementationOnce(() => okStatus('completed'))
    .mockImplementationOnce(() => okStatus('cancelled'));
  const { handlers, props } = mountHook();

  await act(() => handlers.handleBatchProcess({ urls: ['u1', 'u2'] }));
  await act(() => vi.advanceTimersByTimeAsync(2000));

  expect(props.onBatchFinished).toHaveBeenCalledWith(
    expect.objectContaining({ succeeded: 1, failed: 1 }),
  );
});

test('dead backend terminates polling with a visible error instead of forever', async () => {
  apiFetch.mockRejectedValue(new Error('ECONNREFUSED'));
  const { handlers, props } = mountHook();

  await act(() => handlers.handleBatchProcess({ urls: ['u1'] }));
  // Rounds fire at 2s, then back off 4s/8s/16s/30s before giving up.
  await act(() => vi.advanceTimersByTimeAsync(120_000));

  expect(props.setStatus).toHaveBeenLastCalledWith('error');
  const logUpdater = props.setLogs.mock.calls.at(-1)[0];
  expect(logUpdater(['prev']).join(' ')).toMatch(/Lost contact with the backend/);
  expect(vi.getTimerCount()).toBe(0); // no zombie timer left behind
});

test('rounds never overlap: a slow response holds the next round', async () => {
  let release;
  apiFetch.mockImplementation(
    () => new Promise((resolve) => { release = () => resolve({ ok: false }); }),
  );
  const { handlers } = mountHook();

  await act(() => handlers.handleBatchProcess({ urls: ['u1'] }));
  await act(() => vi.advanceTimersByTimeAsync(2000)); // round 1 starts, hangs
  // The old setInterval would have fired 10 more rounds in these 20s; the
  // recursive setTimeout must not start round 2 while round 1 is in flight.
  await act(() => vi.advanceTimersByTimeAsync(20_000));
  expect(apiFetch).toHaveBeenCalledTimes(1);
  await act(async () => { release(); });
});

test('poll loop is cleared on unmount', async () => {
  apiFetch.mockImplementation(() => okStatus('processing'));
  const { handlers, unmount } = mountHook();

  await act(() => handlers.handleBatchProcess({ urls: ['u1'] }));
  expect(vi.getTimerCount()).toBe(1);
  unmount();
  expect(vi.getTimerCount()).toBe(0);
});

// --- H2/H4: single-submit guard, job_id validation, duplicate handling --------

test('a second submit while one is in flight is ignored (no duplicate job)', async () => {
  let release;
  submitProcessJob.mockImplementation(
    () => new Promise((resolve) => { release = () => resolve({ job_id: 'job-1' }); }),
  );
  const { handlers, props } = mountHook();

  let p1; let p2;
  await act(async () => {
    p1 = handlers.handleProcess({ type: 'url', payload: 'u' });
    p2 = handlers.handleProcess({ type: 'url', payload: 'u' });
  });
  await act(async () => { release(); await p1; await p2; });

  expect(submitProcessJob).toHaveBeenCalledTimes(1);
  expect(props.setJobId).toHaveBeenCalledTimes(1);
  expect(props.setJobId).toHaveBeenCalledWith('job-1');
});

test('submit without a job_id surfaces an error instead of wedging on processing', async () => {
  submitProcessJob.mockResolvedValue({ status: 'queued' }); // no job_id
  const { handlers, props } = mountHook();

  await act(() => handlers.handleProcess({ type: 'url', payload: 'u' }));

  expect(props.setStatus).toHaveBeenLastCalledWith('error');
  const logUpdater = props.setLogs.mock.calls.at(-1)[0];
  expect(logUpdater(['x']).join(' ')).toMatch(/no job ID returned/);
  expect(props.setJobId).not.toHaveBeenCalled();
});

test('duplicate:true picks up the returned job instead of treating it as new', async () => {
  submitProcessJob.mockResolvedValue({ job_id: 'job-dup', status: 'queued', duplicate: true });
  const { handlers, props } = mountHook();

  await act(() => handlers.handleProcess({ type: 'url', payload: 'u' }));

  expect(props.setJobId).toHaveBeenCalledWith('job-dup');
  const logUpdater = props.setLogs.mock.calls.at(-1)[0];
  expect(logUpdater(['x']).join(' ')).toMatch(/already submitted/);
});

test('unreachable backend shows a connection error, not the key modal', async () => {
  const { handlers, props } = mountHook({
    apiKey: '',
    serverHasKey: false,
    checkServerKey: async () => null, // unreachable
  });

  await act(() => handlers.handleProcess({ type: 'url', payload: 'u' }));

  expect(props.setShowKeyModal).not.toHaveBeenCalled();
  expect(props.setStatus).toHaveBeenLastCalledWith('error');
  const logUpdater = props.setLogs.mock.calls.at(-1)[0];
  expect(logUpdater(['x']).join(' ')).toMatch(/Could not reach the backend/);
  expect(submitProcessJob).not.toHaveBeenCalled();
});

test('a positive no-key check opens the key modal', async () => {
  const { handlers, props } = mountHook({
    apiKey: '',
    serverHasKey: false,
    checkServerKey: async () => false, // reachable, no key
  });

  await act(() => handlers.handleProcess({ type: 'url', payload: 'u' }));

  expect(props.setShowKeyModal).toHaveBeenCalledWith(true);
  expect(submitProcessJob).not.toHaveBeenCalled();
});

test('a fresh positive key check submits without the modal', async () => {
  submitProcessJob.mockResolvedValue({ job_id: 'job-9' });
  const { handlers, props } = mountHook({
    apiKey: '',
    serverHasKey: false, // stale mount-time snapshot...
    checkServerKey: async () => true, // ...overridden by the fresh check
  });

  await act(() => handlers.handleProcess({ type: 'url', payload: 'u' }));

  expect(props.setShowKeyModal).not.toHaveBeenCalled();
  expect(props.setJobId).toHaveBeenCalledWith('job-9');
});

test('L2: failed submit clears the pre-set processingMedia (no stale preview)', async () => {
  submitProcessJob.mockRejectedValue(new Error('tunnel down'));
  const { handlers, props } = mountHook();

  await act(() => handlers.handleProcess({ type: 'url', payload: 'u' }));

  expect(props.setStatus).toHaveBeenLastCalledWith('error');
  expect(props.setProcessingMedia).toHaveBeenCalledWith(null);
});

test('L2: submit with no job_id clears processingMedia too', async () => {
  submitProcessJob.mockResolvedValue({ status: 'queued' }); // no job_id
  const { handlers, props } = mountHook();

  await act(() => handlers.handleProcess({ type: 'url', payload: 'u' }));

  expect(props.setStatus).toHaveBeenLastCalledWith('error');
  expect(props.setProcessingMedia).toHaveBeenCalledWith(null);
});
