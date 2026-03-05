import { useState } from 'react';
import { useAnalysesStats, useRecentAnalyses } from '../../api/hooks';
import { apiGet } from '../../api/client';
import { JsonViewer } from '../shared/JsonViewer';
import {
  formatScore, formatCurrency, formatElapsed, formatPrice, formatDate, decisionBg, clsx,
} from '../../lib/utils';
import type { AnalysisDetail, AnalysesStats } from '../../api/types';

// ── Stats header ──────────────────────────────────────────────────────────────

function StatsHeader({ stats }: { stats: AnalysesStats }) {
  return (
    <div className="grid grid-cols-5 gap-4">
      {([
        ['Total Analyses',  stats.total_analyses],
        ['Today',           stats.analyses_today],
        ['Unique Tickers',  stats.unique_tickers],
        ['Avg Quality',     formatScore(stats.avg_quality_score)],
        ['Total Cost',      formatCurrency(stats.total_cost_usd, 2)],
      ] as const).map(([l, v]) => (
        <div key={l} className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <p className="text-2xl font-bold text-white">{v}</p>
          <p className="text-xs text-slate-400 mt-1">{l}</p>
        </div>
      ))}
    </div>
  );
}

// ── Outcome badge ─────────────────────────────────────────────────────────────

function OutcomeBadge({ status }: { status: string | null }) {
  if (!status) return <span className="text-slate-600 text-xs">—</span>;
  const map: Record<string, string> = {
    pending:  'text-yellow-400',
    partial:  'text-blue-400',
    complete: 'text-green-400',
    error:    'text-red-400',
  };
  return <span className={clsx('text-xs', map[status] ?? 'text-slate-400')}>{status}</span>;
}

// ── Detail panel ──────────────────────────────────────────────────────────────

function DetailPanel({ id }: { id: number }) {
  const [detail,  setDetail]  = useState<AnalysisDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useState(() => {
    setLoading(true);
    apiGet<AnalysisDetail>(`/analyses/${id}`)
      .then(r => setDetail(r))
      .catch(() => {})
      .finally(() => setLoading(false));
  });

  if (loading) return <div className="p-4 text-xs text-slate-500">Loading…</div>;
  if (!detail)  return <div className="p-4 text-xs text-red-400">Failed to load</div>;

  return (
    <div className="px-4 py-3 bg-slate-900 space-y-3">
      <div className="grid grid-cols-3 gap-4 text-xs">
        <div className="space-y-1">
          <p className="text-slate-500 font-medium">Trade Parameters</p>
          {([
            ['Entry Price',  formatPrice(detail.entry_price)],
            ['Stop Loss',    formatPrice(detail.stop_loss)],
            ['Price Target', formatPrice(detail.price_target)],
          ] as const).map(([l, v]) => (
            <div key={l} className="flex justify-between">
              <span className="text-slate-500">{l}</span>
              <span className="text-slate-200">{v}</span>
            </div>
          ))}
        </div>
        <div className="space-y-1">
          <p className="text-slate-500 font-medium">Metadata</p>
          {([
            ['Hybrid Config', detail.hybrid_config ?? '—'],
            ['Cost',          formatCurrency(detail.total_cost_usd)],
            ['Elapsed',       formatElapsed(detail.elapsed_seconds)],
          ] as const).map(([l, v]) => (
            <div key={l} className="flex justify-between">
              <span className="text-slate-500">{l}</span>
              <span className="text-slate-200">{v}</span>
            </div>
          ))}
        </div>
        {detail.outcome && (
          <div className="space-y-1">
            <p className="text-slate-500 font-medium">Outcome</p>
            {Object.entries(detail.outcome).slice(0, 5).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-slate-500 capitalize">{k.replace(/_/g, ' ')}</span>
                <span className="text-slate-200">{String(v ?? '—')}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      {detail.raw_result && <JsonViewer data={detail.raw_result} label="Raw Result" />}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export function AnalysesPage() {
  const [days,    setDays]    = useState(7);
  const [ticker,  setTicker]  = useState('');
  const [limit]               = useState(50);
  const [expanded, setExpanded] = useState<number | null>(null);

  const { data: stats }   = useAnalysesStats(30_000);
  const { data: result }  = useRecentAnalyses({ days, ticker: ticker || undefined, limit }, 30_000);
  const analyses = result?.analyses ?? [];

  return (
    <div className="space-y-6">
      {stats && <StatsHeader stats={stats} />}

      <div className="flex gap-3 items-center">
        <select
          value={days}
          onChange={e => setDays(Number(e.target.value))}
          className="text-xs bg-slate-700 text-slate-200 border border-slate-600 rounded px-2 py-1.5"
        >
          {[7, 14, 30, 90].map(d => <option key={d} value={d}>{d} days</option>)}
        </select>
        <input
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="Filter by ticker…"
          className="px-3 py-1.5 text-xs bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 mono w-36 focus:outline-none focus:border-blue-500"
        />
        <span className="text-xs text-slate-500">{result?.total ?? 0} results</span>
      </div>

      <div className="bg-slate-800 rounded-lg border border-slate-700">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 text-left text-xs text-slate-400 uppercase tracking-wider">
                {['ID', 'Ticker', 'Date', 'Decision', 'Quality', 'Entry', 'Stop', 'Target', 'Cost', 'Elapsed', 'Outcome'].map(h => (
                  <th key={h} className="px-3 py-2 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {analyses.map(row => (
                <>
                  <tr
                    key={row.id}
                    className="border-b border-slate-800 hover:bg-slate-700/50 cursor-pointer"
                    onClick={() => setExpanded(e => e === row.id ? null : row.id)}
                  >
                    <td className="px-3 py-2 text-slate-500 text-xs">{row.id}</td>
                    <td className="px-3 py-2 font-mono font-medium text-white">{row.ticker}</td>
                    <td className="px-3 py-2 text-slate-300">{formatDate(row.trade_date)}</td>
                    <td className="px-3 py-2">
                      <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium', decisionBg(row.decision))}>
                        {row.decision}
                      </span>
                    </td>
                    <td className="px-3 py-2">{formatScore(row.quality_score)}</td>
                    <td className="px-3 py-2">{formatPrice(row.entry_price)}</td>
                    <td className="px-3 py-2">{formatPrice(row.stop_loss)}</td>
                    <td className="px-3 py-2">{formatPrice(row.price_target)}</td>
                    <td className="px-3 py-2">{formatCurrency(row.total_cost_usd)}</td>
                    <td className="px-3 py-2">{formatElapsed(row.elapsed_seconds)}</td>
                    <td className="px-3 py-2"><OutcomeBadge status={row.outcome_status} /></td>
                  </tr>
                  {expanded === row.id && (
                    <tr key={`${row.id}-detail`}>
                      <td colSpan={11}><DetailPanel id={row.id} /></td>
                    </tr>
                  )}
                </>
              ))}
              {!analyses.length && (
                <tr>
                  <td colSpan={11} className="text-center py-12 text-xs text-slate-500">
                    No analyses found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
