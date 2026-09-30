
import { useEffect, useRef } from 'react';
import { pollJob } from '../lib/api';
import { detectPipelineStep } from '../lib/pipelineStep';

const BASE_DELAY = 2000;
const MAX_DELAY = 30000;
// M1: give up after this many consecutive failed polls instead of retrying
// forever, mirroring the batch path's MAX_POLL_ERRORS. The backend job may
// still be running — the message directs the user to the History tab.
const MAX_CONSECUTIVE_ERRORS = 10;

export function useJobPolling({
  jobId,
  isActive,
  onResult,
  onCompleted,
  onStopped,
  onCancelled,
  onFailed,
  onProgress,
  onPaused,
  onConnectionChange,
}) {
  const callbacks = useRef({});
  callbacks.current = { onResult, onCompleted, onStopped, onCancelled, onFailed, onProgress, onPaused, onConnectionChange };

  useEffect(() => {
    if (!isActive || !jobId) return undefined;

    let disposed = false;
    let timer = null;
    let controller = null;
    let consecutiveErrors = 0;
    let disconnectedAnnounced = false;

    const schedule = (delay) => {
      if (!disposed) timer = setTimeout(tick, delay);
    };

    const terminal = (callback, data) => {
      disposed = true;
      callback?.(data);
    };

    const tick = async () => {
      controller = new AbortController();
      try {
        const data = await pollJob(jobId, { signal: controller.signal });
        if (disposed) return;
        consecutiveErrors = 0;
        if (disconnectedAnnounced) callbacks.current.onConnectionChange?.(true);
        disconnectedAnnounced = false;

        if (data.result) callbacks.current.onResult?.(data.result);
        if (data.status === 'completed') return terminal(callbacks.current.onCompleted, data);
        if (data.status === 'stopped') return terminal(callbacks.current.onStopped || callbacks.current.onCompleted, data);
        if (data.status === 'cancelled') return terminal(callbacks.current.onCancelled);
        if (data.status === 'failed') {
          const errorMsg = data.error || data.logs?.at?.(-1) || 'Process failed';
          return terminal(callbacks.current.onFailed, errorMsg);
        }
        if (data.status === 'paused') callbacks.current.onPaused?.(data);
        if (Array.isArray(data.logs)) callbacks.current.onProgress?.(data.logs, data.operations?.stage || detectPipelineStep(data.logs));
        schedule(typeof document !== 'undefined' && document.hidden ? 10000 : BASE_DELAY);
      } catch (error) {
        if (disposed || error?.name === 'AbortError') return;
        // The job is gone from the server (e.g. wiped by a backend restart) —
        // polling it forever can never succeed. Terminal, with a message.
        if (error?.status === 404) {
          return terminal(
            callbacks.current.onFailed,
            'Job not found on server — it may have been removed after a backend restart.',
          );
        }
        consecutiveErrors += 1;
        if (consecutiveErrors >= 3 && !disconnectedAnnounced) {
          disconnectedAnnounced = true;
          callbacks.current.onConnectionChange?.(false, error);
        }
        // M1: stop polling after MAX_CONSECUTIVE_ERRORS dead rounds instead
        // of retrying forever. A network outage is not a pipeline failure,
        // so the message makes clear the job may still be running and points
        // at the History tab for re-attaching once the server is reachable.
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          return terminal(
            callbacks.current.onFailed,
            `Lost contact with the backend after ${MAX_CONSECUTIVE_ERRORS} failed polls — polling stopped. The job may still be running; check the History tab once the server is reachable again.`,
          );
        }
        // A network outage is not a pipeline failure. Keep the durable backend
        // job alive and retry with a ceiling instead of changing its status.
        schedule(Math.min(BASE_DELAY * 2 ** consecutiveErrors, MAX_DELAY));
      }
    };

    tick();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      controller?.abort();
    };
  }, [isActive, jobId]);
}
