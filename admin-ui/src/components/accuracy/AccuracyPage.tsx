import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { useAccuracySummary } from '../../api/hooks';
import { apiGet, apiPost } from '../../api/client';
import { TaskPoller } from '../shared/TaskPoller';
import { DataTable, type Column } from '../shared/DataTable';
import { EmptyState } from '../shared/EmptyState';
import { formatPercent, formatScore, formatDate, clsx } from '../../lib/utils';
import type { AccuracySummary, SignalRow, TickerReport } from '../../api/types';

// ── Summary cards ─────────────────────────────────────────────────────────────

function SummaryCards({ summary }: { summary: AccuracySummary }) {
  const allD   = Object.values(summary.by_decision);
  const overallT5 = allD.length
    ? allD.reduce((a, d) => a + (d.direction_accuracy_t5 ?? 0) * d.count, 0) /
      allD.reduce((a, d) => a + d.count, 0)
    : null;
  const overallRet = allD.length
    ? allD.reduce((a, d) => a + (d.avg_return_t5 ?? 0) * d.count, 0) /
      allD.reduce((a, d) => a + d.count, 0)
    : null;

  const cards = [
    { label: 'Total Signals (complete)', value: summary.complete_outcomes },
    { label: 'Pending (awaiting T+10)',  value: summary.pending_outcomes  },
    { label: 'Direction Accuracy T+5',  value: formatPercent(overallT5)  },
    { label: 'Avg Return T+5',          value: formatPercent(overallRet)  },
  ];

  return (
    <div className="grid grid-cols-4 gap-4">
      {cards.map(c => (
        <div key={c.label} className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <p className="text-2xl font-bold text-white">{c.value}</p>
          <p className="text-xs text-slate-400 mt-1">{c.label}</p>
        </div>
      ))}
    </div>
  );
}

// ── Direction accuracy chart ──────────────────────────────────────────────────

function DirectionChart({ summary }: { summary: AccuracySummary }) {
  const data = ['BUY', 'SELL'].map(d => {
    const s = summary.by_decision[d];
    if (!s) return null;
    return {
      name: d,
      'T+1':  s.direction_accuracy_t1  != null ? +(s.direction_accuracy_t1  * 100).toFixed(1) : 0,
      'T+5':  s.direction_accuracy_t5  != null ? +(s.direction_accuracy_t5  * 100).toFixed(1) : 0,
      'T+10': s.direction_accuracy_t10 != null ? +(s.direction_accuracy_t10 * 100).toFixed(1) : 0,
    };
  }).filter(Boolean);

  if (!data.length) return <EmptyState message="No directional accuracy data yet" />;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">Direction Accuracy by Decision</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data as Record<string, unknown>[]}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="T+1"  fill="#3b82f6" radius={[2, 2, 0, 0]} />
          <Bar dataKey="T+5"  fill="#22c55e" radius={[2, 2, 0, 0]} />
          <Bar dataKey="T+10" fill="#a855f7" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Quality tier chart ────────────────────────────────────────────────────────

const TIER_COLORS: Record<string, string> = {
  'High (8-10)':   '#22c55e',
  'Medium (6-8)':  '#eab308',
  'Low (0-6)':     '#ef4444',
};

function QualityChart({ summary }: { summary: AccuracySummary }) {
  const data = Object.entries(summary.by_quality_tier).map(([tier, s]) => ({
    name: tier,
    'Avg Return T+5': s.avg_return_t5 != null ? +(s.avg_return_t5 * 100).toFixed(2) : 0,
    fill: TIER_COLORS[tier] ?? '#64748b',
  }));

  if (!data.length) return <EmptyState message="No quality tier data yet" />;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-300 mb-1">Quality → Return Correlation (T+5)</h3>
      <p className="text-xs text-slate-500 mb-4">Does quality scoring predict accuracy?</p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }} />
          <Bar dataKey="Avg Return T+5" radius={[2, 2, 0, 0]}>
            {data.map((entry, idx) => (
              <rect key={idx} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Signal tables ─────────────────────────────────────────────────────────────

const SIG_COLS: Column<SignalRow & Record<string, unknown>>[] = [
  { key: 'ticker',    label: 'Ticker',   render: r => <span className="mono font-medium text-white">{r.ticker as string}</span> },
  { key: 'trade_date', label: 'Date',    render: r => <>{formatDate(r.trade_date as string)}</> },
  { key: 'decision',  label: 'Decision', render: r => (
    <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium',
      r.decision === 'BUY' ? 'bg-green-700 text-green-100' :
      r.decision === 'SELL' ? 'bg-red-700 text-red-100' : 'bg-slate-600 text-slate-200')}>
      {r.decision as string}
    </span>
  )},
  { key: 'quality_score', label: 'Quality', render: r => <>{formatScore(r.quality_score as number)}</> },
  { key: 'return_t5',  label: 'Return T+5', render: r => (
    <span className={(r.return_t5 as number) > 0 ? 'text-green-400' : 'text-red-400'}>
      {formatPercent(r.return_t5 as number)}
    </span>
  )},
  { key: 'return_t10', label: 'Return T+10', render: r => <>{formatPercent(r.return_t10 as number)}</> },
];

// ── Ticker drill-down ─────────────────────────────────────────────────────────

function TickerDrilldown() {
  const [input,   setInput]   = useState('');
  const [report,  setReport]  = useState<TickerReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const search = async () => {
    if (!input.trim()) return;
    setLoading(true); setError(null);
    try {
      const r = await apiGet<TickerReport>(`/accuracy/ticker/${input.toUpperCase()}`);
      setReport(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">Ticker Drill-Down</h3>
      <div className="flex gap-2 mb-4">
        <input
          value={input}
          onChange={e => setInput(e.target.value.toUpperCase())}
          onKeyDown={e => e.key === 'Enter' && search()}
          placeholder="AAPL"
          className="w-28 px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 mono focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={search}
          disabled={loading}
          className="px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded"
        >
          {loading ? 'Loading…' : 'Search'}
        </button>
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {report && (
        <div className="space-y-3">
          <div className="flex gap-6 text-sm">
            <span className="text-slate-400">Total: <strong className="text-white">{report.total}</strong></span>
            <span className="text-slate-400">Complete: <strong className="text-green-400">{report.complete}</strong></span>
            <span className="text-slate-400">Pending: <strong className="text-yellow-400">{report.pending}</strong></span>
          </div>
          <DataTable
            columns={SIG_COLS}
            rows={report.signals as unknown as (SignalRow & Record<string, unknown>)[]}
            keyFn={r => r.id}
            emptyMsg="No signals"
          />
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function AccuracyPage() {
  const [days,    setDays]    = useState(30);
  const { data: summary, refresh } = useAccuracySummary(days);
  const [taskId,  setTaskId]  = useState<string | null>(null);
  const [backfillDays, setBackfillDays] = useState(30);

  const updateNow = async () => {
    await apiPost('/accuracy/update').catch(() => {});
    refresh();
  };

  const backfill = async () => {
    const res = await apiPost<{ task_id: string }>('/accuracy/backfill', { days_back: backfillDays });
    setTaskId(res.task_id);
  };

  const bestSigs  = summary?.best_signals  ?? [];
  const worstSigs = summary?.worst_signals ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-slate-200">Accuracy Tracker</h2>
        <div className="flex gap-2 items-center">
          <select
            value={days}
            onChange={e => setDays(Number(e.target.value))}
            className="text-xs bg-slate-700 text-slate-200 border border-slate-600 rounded px-2 py-1"
          >
            {[7, 14, 30, 90].map(d => <option key={d} value={d}>{d} days</option>)}
          </select>
          <button
            onClick={updateNow}
            className="px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 rounded"
          >
            Update Now
          </button>
          <input
            type="number"
            value={backfillDays}
            onChange={e => setBackfillDays(Number(e.target.value))}
            className="w-16 px-2 py-1.5 text-xs bg-slate-700 border border-slate-600 rounded text-white"
          />
          <button
            onClick={backfill}
            className="px-3 py-1.5 text-xs bg-purple-700 hover:bg-purple-600 text-white rounded"
          >
            Backfill
          </button>
        </div>
      </div>

      {taskId && (
        <TaskPoller taskId={taskId} label="Backfill" onComplete={() => { setTaskId(null); refresh(); }} />
      )}

      {summary ? (
        <>
          <SummaryCards summary={summary} />
          <div className="grid grid-cols-2 gap-4">
            <DirectionChart summary={summary} />
            <QualityChart   summary={summary} />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-800 rounded-lg border border-slate-700">
              <div className="px-5 py-3 border-b border-slate-700">
                <h3 className="text-sm font-semibold text-green-400">Best Signals</h3>
              </div>
              <DataTable
                columns={SIG_COLS}
                rows={bestSigs as unknown as (SignalRow & Record<string, unknown>)[]}
                keyFn={r => r.id}
                emptyMsg="No data"
              />
            </div>
            <div className="bg-slate-800 rounded-lg border border-slate-700">
              <div className="px-5 py-3 border-b border-slate-700">
                <h3 className="text-sm font-semibold text-red-400">Worst Signals</h3>
              </div>
              <DataTable
                columns={SIG_COLS}
                rows={worstSigs as unknown as (SignalRow & Record<string, unknown>)[]}
                keyFn={r => r.id}
                emptyMsg="No data"
              />
            </div>
          </div>
        </>
      ) : (
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-20 bg-slate-800 rounded-lg border border-slate-700 animate-pulse" />
          ))}
        </div>
      )}

      <TickerDrilldown />
    </div>
  );
}
