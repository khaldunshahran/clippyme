
import { useCallback, useEffect, useRef, useState } from 'react';
import { getApiUrl } from '../config';
import { apiFetch } from '../lib/apiToken';

// A status probe must never hang the header pill in "loading" forever —
// bound it well under the 30s apiFetch default so a tunnel stall resolves
// to "unreachable" quickly.
const STATUS_TIMEOUT_MS = 15_000;

export function useBackendStatus() {
  const [hfTokenSet, setHfTokenSet] = useState(true);
  const [cookiesConfigured, setCookiesConfigured] = useState(false);
  const [state, setState] = useState({ loading: true, reachable: true, error: null, updatedAt: null });
  const controllerRef = useRef(null);

  const refresh = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState((current) => ({ ...current, loading: true }));
    try {
      const [configRes, cookiesRes] = await Promise.all([
        apiFetch(getApiUrl('/api/config'), { signal: controller.signal, timeoutMs: STATUS_TIMEOUT_MS }),
        apiFetch(getApiUrl('/api/config/cookies/status'), { signal: controller.signal, timeoutMs: STATUS_TIMEOUT_MS }),
      ]);
      if (!configRes.ok || !cookiesRes.ok) throw new Error('Backend status request failed');
      const [config, cookies] = await Promise.all([configRes.json(), cookiesRes.json()]);
      // New contract: secret-presence booleans (legacy UPPER_CASE aliases
      // accepted as a fallback).
      setHfTokenSet(!!(config.has_hf_token ?? config.HF_TOKEN));
      setCookiesConfigured(!!cookies.configured);
      setState({ loading: false, reachable: true, error: null, updatedAt: Date.now() });
    } catch (error) {
      if (error?.name === 'AbortError') return;
      setState({ loading: false, reachable: false, error, updatedAt: Date.now() });
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => controllerRef.current?.abort();
  }, [refresh]);

  return { hfTokenSet, setHfTokenSet, cookiesConfigured, setCookiesConfigured, ...state, refresh };
}
