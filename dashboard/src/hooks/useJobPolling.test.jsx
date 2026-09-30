
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('../lib/api', () => ({ pollJob: vi.fn() }));
import { pollJob } from '../lib/api';
import { useJobPolling } from './useJobPolling';

beforeEach(() => { vi.useFakeTimers(); vi.clearAllMocks(); });
afterEach(() => vi.useRealTimers());

function mount(overrides = {}) {
  const callbacks = {
    onResult: vi.fn(), onCompleted: vi.fn(), onStopped: vi.fn(), onCancelled: vi.fn(),
    onFailed: vi.fn(), onProgress: vi.fn(), onConnectionChange: vi.fn(), ...overrides,
  };
  const hook = renderHook(() => useJobPolling({ jobId: 'j', isActive: true, ...callbacks }));
  return { ...hook, callbacks };
}

test('polls immediately and completes once', async () => {
  pollJob.mockResolvedValue({ status: 'completed', result: { clips: [] } });
  const { callbacks } = mount();
  await act(async () => {});
  expect(pollJob).toHaveBeenCalledTimes(1);
  expect(callbacks.onCompleted).toHaveBeenCalledTimes(1);
  expect(vi.getTimerCount()).toBe(0);
});

test('network errors do not falsely mark a durable job as failed', async () => {
  pollJob.mockRejectedValue(new Error('offline'));
  const { callbacks, unmount } = mount();
  await act(() => vi.advanceTimersByTimeAsync(20_000));
  expect(callbacks.onFailed).not.toHaveBeenCalled();
  expect(callbacks.onConnectionChange).toHaveBeenCalledWith(false, expect.any(Error));
  unmount();
  expect(vi.getTimerCount()).toBe(0);
});

test('aborts an in-flight request on unmount', async () => {
  let signal;
  pollJob.mockImplementation((_id, options) => { signal = options.signal; return new Promise(() => {}); });
  const { unmount } = mount();
  await act(async () => {});
  expect(signal.aborted).toBe(false);
  unmount();
  expect(signal.aborted).toBe(true);
});

test('HTTP 404 is terminal: job not found on server', async () => {
  const err = new Error('Not Found');
  err.status = 404;
  pollJob.mockRejectedValue(err);
  const { callbacks } = mount();
  await act(async () => {});
  expect(callbacks.onFailed).toHaveBeenCalledTimes(1);
  expect(callbacks.onFailed).toHaveBeenCalledWith(expect.stringMatching(/not found on server/i));
  expect(callbacks.onConnectionChange).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(0); // no more polling
});

test('stops after 10 consecutive poll errors with a terminal message', async () => {
  pollJob.mockRejectedValue(new Error('tunnel down'));
  const { callbacks } = mount();
  // 1 immediate poll + 9 backoff ticks = 10 consecutive errors
  await act(async () => {});
  for (let i = 0; i < 9; i++) {
    await act(() => vi.advanceTimersByTimeAsync(60_000));
  }
  expect(pollJob).toHaveBeenCalledTimes(10);
  expect(callbacks.onFailed).toHaveBeenCalledTimes(1);
  expect(callbacks.onFailed).toHaveBeenCalledWith(
    expect.stringMatching(/lost contact with the backend after 10 failed polls/i)
  );
  expect(vi.getTimerCount()).toBe(0); // polling stopped, no more timers
});

test('error counter resets after a successful poll', async () => {
  pollJob.mockRejectedValue(new Error('flaky'));
  const { callbacks } = mount();
  // Drive exactly 5 failed polls: 1 immediate + 4 on the backoff schedule
  // (4s, 8s, 16s, 30s after consecutive errors 1-4).
  await act(async () => {});
  for (const ms of [4_000, 8_000, 16_000, 30_000]) {
    await act(() => vi.advanceTimersByTimeAsync(ms));
  }
  expect(pollJob).toHaveBeenCalledTimes(5);
  expect(callbacks.onFailed).not.toHaveBeenCalled();
  // Backend recovers: the next poll succeeds and the streak resets.
  pollJob.mockResolvedValue({ status: 'processing', progress: 10 });
  await act(() => vi.advanceTimersByTimeAsync(30_000));
  expect(pollJob).toHaveBeenCalledTimes(6);
  // Back to failing — 4 more errors is only a streak of 4, not 10.
  pollJob.mockRejectedValue(new Error('flaky again'));
  for (const ms of [2_000, 4_000, 8_000, 16_000]) {
    await act(() => vi.advanceTimersByTimeAsync(ms));
  }
  expect(pollJob).toHaveBeenCalledTimes(10);
  expect(callbacks.onFailed).not.toHaveBeenCalled();
});
