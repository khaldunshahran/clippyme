// Unit tests for lib/api: error-text sanitization, submit timeouts and the
// idempotency key, and the X-Gemini-Key header contract.
import { test, expect, vi, beforeEach } from 'vitest';

vi.mock('./apiToken', () => ({ apiFetch: vi.fn() }));

import { apiFetch } from './apiToken';
import {
  throwFromResponse,
  submitProcessJob,
  submitBatchJob,
  newIdempotencyKey,
  SUBMIT_TIMEOUT_MS,
} from './api';

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

function failedRes(status, text) {
  return { status, ok: false, text: () => Promise.resolve(text) };
}

test('throwFromResponse strips HTML tags and truncates long bodies', async () => {
  const html = `<html><head><title>502</title></head><body>${'x'.repeat(2000)}</body></html>`;
  const err = await throwFromResponse(failedRes(500, html)).catch((e) => e);
  expect(err).toBeInstanceOf(Error);
  expect(err.message.length).toBeLessThanOrEqual(300);
  expect(err.message).not.toMatch(/<[^>]*>/);
  expect(err.status).toBe(500);
  expect(err.retryable).toBe(true);
});

test('throwFromResponse maps gateway statuses to friendly text', async () => {
  for (const status of [502, 503, 504]) {
    const err = await throwFromResponse(failedRes(status, '<html><body>Bad Gateway</body></html>')).catch((e) => e);
    expect(err.message).toMatch(/backend/i);
    expect(err.message).not.toMatch(/</);
  }
});

test('throwFromResponse keeps a real JSON detail from the backend', async () => {
  const err = await throwFromResponse(
    failedRes(400, JSON.stringify({ detail: 'URL is not reachable' })),
  ).catch((e) => e);
  expect(err.message).toBe('URL is not reachable');
});

test('throwFromResponse falls back to HTTP status when the body is empty', async () => {
  const err = await throwFromResponse(failedRes(500, '')).catch((e) => e);
  expect(err.message).toMatch(/500/);
});

test('newIdempotencyKey returns a fresh unique key per call', () => {
  const a = newIdempotencyKey();
  const b = newIdempotencyKey();
  expect(typeof a).toBe('string');
  expect(a.length).toBeGreaterThan(0);
  expect(a).not.toBe(b);
});

test('submitProcessJob sends an Idempotency-Key header and a long timeout', async () => {
  apiFetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({ job_id: 'j1' }) });
  await submitProcessJob({ type: 'url', payload: 'https://example.com/v' }, '');
  expect(apiFetch).toHaveBeenCalledTimes(1);
  const init = apiFetch.mock.calls[0][1];
  expect(init.method).toBe('POST');
  expect(init.headers['Idempotency-Key']).toBeTruthy();
  expect(init.timeoutMs).toBe(SUBMIT_TIMEOUT_MS);
  expect(SUBMIT_TIMEOUT_MS).toBeGreaterThan(30_000);
  // No Gemini key anywhere → the header must be absent, not empty.
  expect('X-Gemini-Key' in init.headers).toBe(false);
});

test('submitProcessJob honors a caller-supplied idempotency key', async () => {
  apiFetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({ job_id: 'j1' }) });
  await submitProcessJob({ type: 'url', payload: 'https://example.com/v' }, '', { idempotencyKey: 'k-123' });
  expect(apiFetch.mock.calls[0][1].headers['Idempotency-Key']).toBe('k-123');
});

test('submitBatchJob omits X-Gemini-Key when the key is empty', async () => {
  apiFetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({ jobs: [] }) });
  await submitBatchJob({ urls: ['https://example.com/v'] }, '');
  const init = apiFetch.mock.calls[0][1];
  expect('X-Gemini-Key' in init.headers).toBe(false);
  expect(init.headers['Content-Type']).toBe('application/json');
  expect(init.headers['Idempotency-Key']).toBeTruthy();
});

test('submitBatchJob sends X-Gemini-Key when a key is provided', async () => {
  apiFetch.mockResolvedValue({ ok: true, json: () => Promise.resolve({ jobs: [] }) });
  await submitBatchJob({ urls: ['https://example.com/v'] }, '  secret  ');
  expect(apiFetch.mock.calls[0][1].headers['X-Gemini-Key']).toBe('secret');
});
