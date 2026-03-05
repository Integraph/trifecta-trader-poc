import { useCallback, useEffect, useRef, useState } from 'react';
import { apiGet } from './client';
import type {
  AccuracySummary,
  AnalysesStats,
  AnalysisRow,
  HealthResponse,
  QueueStatus,
  SchedulerStatus,
} from './types';

// ── Generic polling hook ───────────────────────────────────────────────────────

export function usePolling<T>(
  fetcher: () => Promise<T>,
  interval: number,
): { data: T | null; loading: boolean; error: string | null; refresh: () => void } {
  const [data, setData]       = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const timerRef              = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetch_ = useCallback(async () => {
    if (document.hidden) return;
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    fetch_();
    timerRef.current = setInterval(fetch_, interval);
    const onVisibility = () => { if (!document.hidden) fetch_(); };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [fetch_, interval]);

  return { data, loading, error, refresh: fetch_ };
}

// ── Specific hooks ─────────────────────────────────────────────────────────────

export function useHealth(interval = 10_000) {
  const fetcher = useCallback(() => apiGet<HealthResponse>('/health'), []);
  return usePolling(fetcher, interval);
}

export function useSchedulerStatus(interval = 15_000) {
  const fetcher = useCallback(() => apiGet<SchedulerStatus>('/scheduler/status'), []);
  return usePolling(fetcher, interval);
}

export function useQueueStatus(interval = 10_000) {
  const fetcher = useCallback(() => apiGet<QueueStatus>('/queue/status'), []);
  return usePolling(fetcher, interval);
}

export function useAccuracySummary(days = 30, interval = 60_000) {
  const fetcher = useCallback(
    () => apiGet<AccuracySummary>('/accuracy/summary', { days }),
    [days],
  );
  return usePolling(fetcher, interval);
}

export function useAnalysesStats(interval = 30_000) {
  const fetcher = useCallback(() => apiGet<AnalysesStats>('/analyses/stats'), []);
  return usePolling(fetcher, interval);
}

export function useRecentAnalyses(
  params: { days?: number; ticker?: string; limit?: number } = {},
  interval = 30_000,
) {
  const { days = 7, ticker, limit = 50 } = params;
  const fetcher = useCallback(() => {
    const p: Record<string, string | number> = { days, limit };
    if (ticker) p.ticker = ticker;
    return apiGet<{ analyses: AnalysisRow[]; total: number }>('/analyses/recent', p);
  }, [days, ticker, limit]);
  return usePolling(fetcher, interval);
}

// ── One-shot task poller ───────────────────────────────────────────────────────

export function useTaskPoller(taskId: string | null, pollInterval = 2000) {
  const [status, setStatus]   = useState<'running' | 'complete' | 'error' | null>(null);
  const [result, setResult]   = useState<unknown>(null);
  const [error, setError]     = useState<string | null>(null);
  const timerRef              = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!taskId) return;
    setStatus('running');
    setResult(null);
    setError(null);

    const poll = async () => {
      try {
        const res = await apiGet<{
          status: 'running' | 'complete' | 'error';
          result: unknown;
          error: string | null;
        }>(`/tasks/${taskId}`);
        setStatus(res.status);
        if (res.status !== 'running') {
          if (timerRef.current) clearInterval(timerRef.current);
          setResult(res.result);
          setError(res.error);
        }
      } catch (e) {
        if (timerRef.current) clearInterval(timerRef.current);
        setStatus('error');
        setError(e instanceof Error ? e.message : String(e));
      }
    };

    poll();
    timerRef.current = setInterval(poll, pollInterval);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [taskId, pollInterval]);

  return { status, result, error };
}
