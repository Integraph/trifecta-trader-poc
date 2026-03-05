import { useEffect, useState } from 'react';
import { Plus, RotateCcw, Trash2, RefreshCw } from 'lucide-react';
import { useQueueStatus } from '../../api/hooks';
import { apiGet, apiPost, apiDelete } from '../../api/client';
import { DataTable, type Column } from '../shared/DataTable';
import { StatusDot } from '../shared/StatusDot';
import { JsonViewer } from '../shared/JsonViewer';
import { priorityColor, formatDateTime, formatElapsed, formatScore, clsx } from '../../lib/utils';
import type { QueueCandidate, QueueCompleted, QueueStatus } from '../../api/types';

// ── Status panel ──────────────────────────────────────────────────────────────

function StatusPanel({ status }: { status: QueueStatus }) {
  const color = status.is_running ? 'green' : status.enabled ? 'yellow' : 'red';
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <div className="flex items-center gap-2 mb-4">
        <StatusDot color={color as 'green' | 'yellow' | 'red' | 'gray'} pulse={status.is_running} />
        <span className="text-sm font-semibold text-slate-200">
          Queue Reader — {status.is_running ? 'Running' : 'Stopped'}
        </span>
      </div>
      <div className="grid grid-cols-4 gap-4">
        {([['Pending', status.counts.pending, 'text-yellow-400'],
           ['Processing', status.counts.processing, 'text-blue-400'],
           ['Completed', status.counts.completed, 'text-green-400'],
           ['Failed', status.counts.failed, 'text-red-400']] as const).map(([l, v, cls]) => (
          <div key={l} className="text-center">
            <p className={`text-3xl font-bold ${cls}`}>{v}</p>
            <p className="text-xs text-slate-500 mt-1">{l}</p>
          </div>
        ))}
      </div>
      <div className="mt-4 pt-4 border-t border-slate-700 flex gap-6 text-xs text-slate-400">
        <span>Poll every {status.config.poll_interval_seconds}s</span>
        <span>Max retries: {status.config.max_retries}</span>
        <span>Cooldown: {status.config.cooldown_seconds}s</span>
        {status.last_poll && <span>Last poll: {formatDateTime(status.last_poll)}</span>}
      </div>
    </div>
  );
}

// ── Enqueue form ──────────────────────────────────────────────────────────────

function EnqueueForm({ onSuccess }: { onSuccess: () => void }) {
  const [ticker,   setTicker]   = useState('');
  const [priority, setPriority] = useState('high');
  const [reason,   setReason]   = useState('');
  const [toast,    setToast]    = useState<string | null>(null);
  const [loading,  setLoading]  = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticker.trim()) return;
    setLoading(true);
    try {
      const res = await apiPost<{ filename: string }>('/queue/enqueue', {
        ticker: ticker.toUpperCase(),
        priority,
        reason: reason || 'Manual admin request',
      });
      setToast(`Queued: ${res.filename}`);
      setTicker(''); setReason('');
      setTimeout(() => setToast(null), 3000);
      onSuccess();
    } catch (e) {
      setToast(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">Enqueue Ticker</h3>
      <form onSubmit={submit} className="flex gap-3 flex-wrap">
        <input
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="TICKER"
          className="w-28 px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 mono"
          required
        />
        <select
          value={priority}
          onChange={e => setPriority(e.target.value)}
          className="px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded text-slate-200"
        >
          {['high', 'medium', 'low'].map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <input
          value={reason}
          onChange={e => setReason(e.target.value)}
          placeholder="Reason (optional)"
          className="flex-1 min-w-48 px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="flex items-center gap-1.5 px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded font-medium"
        >
          <Plus className="w-4 h-4" /> Enqueue
        </button>
      </form>
      {toast && (
        <p className={clsx('mt-2 text-xs', toast.startsWith('Error') ? 'text-red-400' : 'text-green-400')}>
          {toast}
        </p>
      )}
    </div>
  );
}

// ── Pending table ─────────────────────────────────────────────────────────────

const PENDING_COLS: Column<QueueCandidate & Record<string, unknown>>[] = [
  { key: 'ticker',   label: 'Ticker', sortable: true, render: r => <span className="font-mono font-medium text-white">{r.ticker as string}</span> },
  { key: 'priority', label: 'Priority', render: r => <span className={priorityColor(r.priority as string)}>{r.priority as string}</span> },
  { key: 'score',    label: 'Score', sortable: true, render: r => <>{r.score != null ? (r.score as number).toFixed(2) : '—'}</> },
  { key: 'catalysts', label: 'Catalysts', render: r => <span className="text-xs">{(r.catalysts as string[]).join(', ') || '—'}</span> },
  { key: 'retry_count', label: 'Retries' },
  { key: 'queued_at', label: 'Queued', render: r => <>{formatDateTime(r.queued_at as string)}</> },
];

// ── Completed table ───────────────────────────────────────────────────────────

function CompletedTable({
  rows, onRetry,
}: {
  rows: QueueCompleted[];
  onRetry: (filename: string) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!rows.length) return <p className="text-xs text-slate-500 px-4 py-8 text-center">No completed items.</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-left text-xs text-slate-400 uppercase tracking-wider">
            {['Ticker', 'Decision', 'Quality', 'Elapsed', 'Completed', 'Actions'].map(h => (
              <th key={h} className="px-4 py-2">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <>
              <tr
                key={r.filename}
                className="border-b border-slate-800 hover:bg-slate-800/50 cursor-pointer"
                onClick={() => setExpanded(e => e === r.filename ? null : r.filename)}
              >
                <td className="px-4 py-2 font-mono font-medium text-white">{r.ticker}</td>
                <td className="px-4 py-2">
                  <span className={clsx('px-2 py-0.5 rounded text-xs font-medium',
                    r.decision === 'BUY' ? 'bg-green-700 text-green-100' :
                    r.decision === 'SELL' ? 'bg-red-700 text-red-100' : 'bg-slate-600 text-slate-200')}>
                    {r.decision ?? '—'}
                  </span>
                </td>
                <td className="px-4 py-2">{formatScore(r.quality_score)}</td>
                <td className="px-4 py-2">{formatElapsed(r.elapsed_seconds)}</td>
                <td className="px-4 py-2">{formatDateTime(r.completed_at)}</td>
                <td className="px-4 py-2">
                  <button
                    onClick={e => { e.stopPropagation(); onRetry(r.filename); }}
                    className="p-1 rounded hover:bg-slate-600 text-slate-400 hover:text-slate-200"
                    title="Retry"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </td>
              </tr>
              {expanded === r.filename && (
                <tr key={`${r.filename}-detail`} className="bg-slate-900">
                  <td colSpan={6} className="px-4 py-3">
                    <JsonViewer data={r.analysis_result} defaultOpen label="Analysis result" />
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function QueuePage() {
  const { data: status, refresh: refreshStatus } = useQueueStatus(10_000);
  const [pending,   setPending]   = useState<QueueCandidate[]>([]);
  const [completed, setCompleted] = useState<QueueCompleted[]>([]);
  const [clearing,  setClearing]  = useState(false);

  const loadPending = () =>
    apiGet<{ candidates: QueueCandidate[] }>('/queue/pending')
      .then(r => setPending(r.candidates)).catch(() => {});

  const loadCompleted = () =>
    apiGet<{ completed: QueueCompleted[] }>('/queue/completed', { days: 1, limit: 20 })
      .then(r => setCompleted(r.completed)).catch(() => {});

  useEffect(() => {
    loadPending();
    loadCompleted();
    const t = setInterval(() => { loadPending(); loadCompleted(); refreshStatus(); }, 10_000);
    return () => clearInterval(t);
  }, []);

  const retry = (filename: string) =>
    apiPost(`/queue/retry/${filename}`).then(() => { loadPending(); loadCompleted(); }).catch(() => {});

  const clearCompleted = async () => {
    if (!confirm('Clear all completed queue files?')) return;
    setClearing(true);
    await apiDelete('/queue/clear', { target: 'completed' }).catch(() => {});
    setClearing(false);
    loadCompleted();
    refreshStatus();
  };

  return (
    <div className="space-y-6">
      {status && <StatusPanel status={status} />}

      <EnqueueForm onSuccess={loadPending} />

      <div className="bg-slate-800 rounded-lg border border-slate-700">
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-slate-300">Pending Candidates</h3>
          <button onClick={loadPending} className="text-xs text-slate-400 hover:text-slate-200">
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
        <DataTable
          columns={PENDING_COLS}
          rows={pending as unknown as (QueueCandidate & Record<string, unknown>)[]}
          keyFn={r => r.filename}
          emptyMsg="No pending candidates"
        />
      </div>

      <div className="bg-slate-800 rounded-lg border border-slate-700">
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-slate-300">Completed (last 24h)</h3>
          <button
            onClick={clearCompleted}
            disabled={clearing}
            className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 disabled:opacity-50"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear Completed
          </button>
        </div>
        <CompletedTable rows={completed} onRetry={retry} />
      </div>
    </div>
  );
}
