import { useEffect, useState } from 'react';
import { FlaskConical, AlertTriangle } from 'lucide-react';
import { apiGet, apiPost } from '../../api/client';
import { TaskPoller } from '../shared/TaskPoller';
import { JsonViewer } from '../shared/JsonViewer';
import { today, formatScore, formatCurrency, formatElapsed, formatDateTime, decisionBg, clsx } from '../../lib/utils';
import type { HybridConfig, TaskStatus } from '../../api/types';

// ── Result card ────────────────────────────────────────────────────────────────

function ResultCard({ result }: { result: unknown }) {
  const r = result as Record<string, unknown>;
  if (!r) return null;

  const decision     = r.decision as string | undefined;
  const qScore       = r.quality_score as Record<string, number> | number | undefined;
  const composite    = typeof qScore === 'object' && qScore ? qScore.composite : (qScore as number);
  const tradeParams  = r.trade_params as Record<string, unknown> | undefined;
  const cost         = r.total_cost_usd as number | undefined;
  const elapsed      = r.elapsed_seconds as number | undefined;
  const published    = r.published as boolean | undefined;

  const qFields: [string, unknown][] = typeof qScore === 'object' && qScore
    ? Object.entries(qScore)
    : [];

  return (
    <div className="space-y-4">
      {/* Decision header */}
      <div className="flex items-center gap-4 p-5 bg-slate-800 rounded-lg border border-slate-700">
        <span className={clsx('px-6 py-2 rounded-lg text-lg font-bold', decisionBg(decision))}>
          {decision ?? 'N/A'}
        </span>
        <div>
          <p className="text-sm text-slate-400">Quality Score</p>
          <p className="text-3xl font-bold text-white">{formatScore(composite)}<span className="text-slate-400 text-base">/10</span></p>
        </div>
        <div className="ml-auto text-right text-xs text-slate-400 space-y-1">
          <p>Cost: <span className="text-slate-200">{formatCurrency(cost)}</span></p>
          <p>Elapsed: <span className="text-slate-200">{formatElapsed(elapsed)}</span></p>
          <p>Published: <span className={published ? 'text-green-400' : 'text-slate-400'}>{published ? 'Yes' : 'No'}</span></p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Trade parameters */}
        {tradeParams && (
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Trade Parameters</h4>
            <div className="space-y-2 text-sm">
              {Object.entries(tradeParams).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-slate-400 capitalize">{k.replace(/_/g, ' ')}</span>
                  <span className="text-slate-200 font-medium">{String(v ?? '—')}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quality breakdown */}
        {qFields.length > 0 && (
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Quality Breakdown</h4>
            <div className="space-y-2">
              {qFields.map(([k, v]) => (
                <div key={k} className="space-y-0.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400 capitalize">{k.replace(/_/g, ' ')}</span>
                    <span className="text-slate-200">{typeof v === 'number' ? v.toFixed(1) : String(v)}</span>
                  </div>
                  {typeof v === 'number' && (
                    <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={clsx('h-full rounded-full', v >= 7 ? 'bg-green-500' : v >= 5 ? 'bg-yellow-500' : 'bg-red-500')}
                        style={{ width: `${(v / 10) * 100}%` }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <JsonViewer data={result} label="Full Result" />
    </div>
  );
}

// ── Recent test runs ───────────────────────────────────────────────────────────

function RecentRuns() {
  const [tasks, setTasks] = useState<TaskStatus[]>([]);

  useEffect(() => {
    const load = () =>
      apiGet<{ tasks: TaskStatus[] }>('/tasks', { limit: 20 })
        .then(r => setTasks(r.tasks.filter(t => t.task_id.startsWith('test_'))))
        .catch(() => {});
    load();
    const t = setInterval(load, 10_000);
    return () => clearInterval(t);
  }, []);

  if (!tasks.length) return null;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700">
      <div className="px-5 py-3 border-b border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300">Recent Test Runs</h3>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-left text-xs text-slate-400 uppercase tracking-wider">
            {['Ticker', 'Decision', 'Quality', 'Elapsed', 'Status', 'Time'].map(h => (
              <th key={h} className="px-4 py-2">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tasks.map(t => {
            const r = t.result as Record<string, unknown> | null;
            const q = r?.quality_score;
            const composite = typeof q === 'object' && q
              ? (q as Record<string, number>).composite
              : (q as number | undefined);
            return (
              <tr key={t.task_id} className="border-b border-slate-800">
                <td className="px-4 py-2 mono font-medium text-white">
                  {(r?.ticker as string) ?? t.task_id.split('_')[1] ?? '—'}
                </td>
                <td className="px-4 py-2">
                  {r?.decision ? (
                    <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium', decisionBg(r.decision as string))}>
                      {r.decision as string}
                    </span>
                  ) : '—'}
                </td>
                <td className="px-4 py-2">{formatScore(composite)}</td>
                <td className="px-4 py-2">{formatElapsed(r?.elapsed_seconds as number)}</td>
                <td className="px-4 py-2">
                  <span className={clsx('text-xs', t.status === 'complete' ? 'text-green-400' : t.status === 'error' ? 'text-red-400' : 'text-blue-400')}>
                    {t.status === 'complete' ? '✅' : t.status === 'error' ? '❌' : '⏳'}  {t.status}
                  </span>
                </td>
                <td className="px-4 py-2 text-slate-400">{formatDateTime(t.started_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export function TestRunPage() {
  const [ticker,       setTicker]       = useState('');
  const [hybridConfig, setHybridConfig] = useState('');
  const [tradeDate,    setTradeDate]    = useState(today());
  const [publish,      setPublish]      = useState(false);
  const [configs,      setConfigs]      = useState<HybridConfig[]>([]);
  const [taskId,       setTaskId]       = useState<string | null>(null);
  const [submitting,   setSubmitting]   = useState(false);

  useEffect(() => {
    apiGet<{ configs: HybridConfig[]; active: string | null }>('/config/hybrid-configs')
      .then(r => {
        setConfigs(r.configs);
        if (r.active) setHybridConfig(r.active);
        else if (r.configs.length) setHybridConfig(r.configs[0].name);
      })
      .catch(() => {});
  }, []);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticker.trim()) return;
    setSubmitting(true);
    setTaskId(null);
    try {
      const res = await apiPost<{ task_id: string }>('/test-run', {
        ticker: ticker.toUpperCase(),
        hybrid_config: hybridConfig || undefined,
        trade_date: tradeDate,
        publish,
      });
      setTaskId(res.task_id);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <div className="flex items-center gap-2 mb-5">
          <FlaskConical className="w-5 h-5 text-orange-400" />
          <h2 className="text-sm font-semibold text-slate-200">Run Single-Ticker Analysis</h2>
        </div>

        <form onSubmit={run} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Ticker *</label>
              <input
                value={ticker}
                onChange={e => setTicker(e.target.value.toUpperCase())}
                placeholder="AAPL"
                required
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 mono focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Hybrid Config</label>
              <select
                value={hybridConfig}
                onChange={e => setHybridConfig(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-slate-200"
              >
                {configs.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Trade Date</label>
              <input
                type="date"
                value={tradeDate}
                onChange={e => setTradeDate(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={publish}
                  onChange={e => setPublish(e.target.checked)}
                  className="w-4 h-4 rounded"
                />
                <span className="text-sm text-slate-300">Publish to Supabase</span>
                {publish && (
                  <span className="flex items-center gap-1 text-xs text-yellow-400">
                    <AlertTriangle className="w-3 h-3" /> Writes to production
                  </span>
                )}
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={submitting || !ticker.trim()}
            className="w-full py-2.5 text-sm font-semibold bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white rounded-lg transition-colors"
          >
            {submitting ? `Analyzing ${ticker}…` : 'Run Analysis'}
          </button>
        </form>
      </div>

      {taskId && (
        <TaskPoller
          taskId={taskId}
          pollInterval={3000}
          label={`Analysis: ${ticker}`}
          renderResult={(result) => <ResultCard result={result} />}
        />
      )}

      <RecentRuns />
    </div>
  );
}
