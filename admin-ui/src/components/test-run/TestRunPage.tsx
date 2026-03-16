import { useEffect, useRef, useState } from 'react';
import { FlaskConical, AlertTriangle, Beaker, Loader2 } from 'lucide-react';
import { apiGet, apiPost } from '../../api/client';
import { TaskPoller } from '../shared/TaskPoller';
import { JsonViewer } from '../shared/JsonViewer';
import { today, formatScore, formatCurrency, formatElapsed, formatDateTime, decisionBg, clsx } from '../../lib/utils';
import type { HybridConfigFull, HybridConfigsResponse, TaskStatus, ABCompareResponse } from '../../api/types';

// ── Shared result card ─────────────────────────────────────────────────────────

function ResultCard({ result }: { result: unknown }) {
  const r = result as Record<string, unknown>;
  if (!r) return null;

  const decision     = r.decision as string | undefined;
  const qScore       = r.quality_score as Record<string, number> | number | undefined;
  const composite    = typeof qScore === 'object' && qScore ? qScore.composite : (qScore as number);
  const tradeParams  = r.trade_params as Record<string, unknown> | undefined;
  const cost         = (r.cost_breakdown as Record<string, unknown> | undefined)?.total_usd as number | undefined;
  const elapsed      = r.elapsed_seconds as number | undefined;
  const published    = r.published as boolean | undefined;

  const qFields: [string, unknown][] = typeof qScore === 'object' && qScore
    ? Object.entries(qScore)
    : [];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <span className={clsx('px-5 py-1.5 rounded-lg text-lg font-bold', decisionBg(decision))}>
          {decision ?? 'N/A'}
        </span>
        <div>
          <p className="text-xs text-slate-400">Quality</p>
          <p className="text-2xl font-bold text-white">{formatScore(composite)}<span className="text-slate-400 text-sm">/10</span></p>
        </div>
        <div className="ml-auto text-right text-xs text-slate-400 space-y-0.5">
          <p>Cost: <span className="text-slate-200">{formatCurrency(cost)}</span></p>
          <p>Elapsed: <span className="text-slate-200">{formatElapsed(elapsed)}</span></p>
          <p>Published: <span className={published ? 'text-green-400' : 'text-slate-400'}>{published ? 'Yes' : 'No'}</span></p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {tradeParams && (
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
            <p className="text-xs font-semibold text-slate-400 mb-2">Trade Parameters</p>
            <div className="space-y-1 text-xs">
              {Object.entries(tradeParams).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-slate-500 capitalize">{k.replace(/_/g, ' ')}</span>
                  <span className="text-slate-200 font-medium">{String(v ?? '—')}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {qFields.length > 0 && (
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
            <p className="text-xs font-semibold text-slate-400 mb-2">Quality Breakdown</p>
            <div className="space-y-1.5">
              {qFields.map(([k, v]) => (
                <div key={k} className="space-y-0.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400 capitalize">{k.replace(/_/g, ' ')}</span>
                    <span className="text-slate-200">{typeof v === 'number' ? v.toFixed(1) : String(v)}</span>
                  </div>
                  {typeof v === 'number' && (
                    <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
                      <div className={clsx('h-full rounded-full', v >= 7 ? 'bg-green-500' : v >= 5 ? 'bg-yellow-500' : 'bg-red-500')}
                        style={{ width: `${(v / 10) * 100}%` }} />
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
            const composite = typeof q === 'object' && q ? (q as Record<string, number>).composite : (q as number | undefined);
            return (
              <tr key={t.task_id} className="border-b border-slate-800">
                <td className="px-4 py-2 mono font-medium text-white">{(r?.ticker as string) ?? t.task_id.split('_')[1] ?? '—'}</td>
                <td className="px-4 py-2">
                  {r?.decision ? <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium', decisionBg(r.decision as string))}>{r.decision as string}</span> : '—'}
                </td>
                <td className="px-4 py-2">{formatScore(composite)}</td>
                <td className="px-4 py-2">{formatElapsed(r?.elapsed_seconds as number)}</td>
                <td className="px-4 py-2">
                  <span className={clsx('text-xs', t.status === 'complete' ? 'text-green-400' : t.status === 'error' ? 'text-red-400' : 'text-blue-400')}>
                    {t.status}
                  </span>
                </td>
                <td className="px-4 py-2 text-slate-400 text-xs">{formatDateTime(t.started_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Single Run ─────────────────────────────────────────────────────────────────

function SingleRunMode({ configs, active }: { configs: HybridConfigFull[]; active: string | null }) {
  const [ticker,       setTicker]       = useState('');
  const [hybridConfig, setHybridConfig] = useState(active ?? '');
  const [tradeDate,    setTradeDate]    = useState(today());
  const [publish,      setPublish]      = useState(false);
  const [taskId,       setTaskId]       = useState<string | null>(null);
  const [submitting,   setSubmitting]   = useState(false);

  useEffect(() => {
    if (active && !hybridConfig) setHybridConfig(active);
    else if (configs.length && !hybridConfig) setHybridConfig(configs[0].name);
  }, [active, configs]);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticker.trim()) return;
    setSubmitting(true);
    setTaskId(null);
    try {
      const res = await apiPost<{ task_id: string }>('/test-run', {
        ticker: ticker.toUpperCase(), hybrid_config: hybridConfig || undefined, trade_date: tradeDate, publish,
      });
      setTaskId(res.task_id);
    } finally { setSubmitting(false); }
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <form onSubmit={run} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Ticker *</label>
              <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="AAPL" required
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 mono focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Hybrid Config</label>
              <select value={hybridConfig} onChange={e => setHybridConfig(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-slate-200">
                {configs.map(c => <option key={c.name} value={c.name}>{c.name}{c.name === active ? ' (active)' : ''}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Trade Date</label>
              <input type="date" value={tradeDate} onChange={e => setTradeDate(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-white focus:outline-none focus:border-blue-500" />
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={publish} onChange={e => setPublish(e.target.checked)} className="w-4 h-4 rounded" />
                <span className="text-sm text-slate-300">Publish to Supabase</span>
                {publish && <span className="flex items-center gap-1 text-xs text-yellow-400"><AlertTriangle className="w-3 h-3" /> Production</span>}
              </label>
            </div>
          </div>
          <button type="submit" disabled={submitting || !ticker.trim()}
            className="w-full py-2.5 text-sm font-semibold bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white rounded-lg">
            {submitting ? `Analyzing ${ticker}…` : 'Run Analysis'}
          </button>
        </form>
      </div>

      {taskId && (
        <TaskPoller taskId={taskId} pollInterval={3000} label={`Analysis: ${ticker}`}
          renderResult={result => <ResultCard result={result} />} />
      )}
      <RecentRuns />
    </div>
  );
}

// ── A/B Compare side panel ─────────────────────────────────────────────────────

function ABSidePanel({
  label, cfgName, side,
}: {
  label:   string;
  cfgName: string;
  side:    ABCompareResponse['config_a'] | ABCompareResponse['config_b'];
}) {
  return (
    <div className="flex-1 min-w-0 bg-slate-800 rounded-lg border border-slate-700 p-4 space-y-3">
      <div className="border-b border-slate-700 pb-2">
        <p className="text-xs font-semibold text-slate-400">{label}</p>
        <p className="text-sm font-medium text-white mono">{cfgName}</p>
      </div>
      {side.status === 'running' ? (
        <div className="flex items-center gap-2 py-8 justify-center">
          <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
          <span className="text-sm text-slate-400">Analyzing…</span>
        </div>
      ) : side.status === 'error' ? (
        <div className="py-4 text-xs text-red-400">
          <p className="font-medium mb-1">Analysis failed</p>
          <p className="mono">{side.error}</p>
        </div>
      ) : side.result ? (
        <ResultCard result={side.result} />
      ) : null}
    </div>
  );
}

function ComparisonSummary({ ab }: { ab: ABCompareResponse }) {
  const ra = ab.config_a.result as Record<string, unknown> | null;
  const rb = ab.config_b.result as Record<string, unknown> | null;
  if (!ra || !rb) return null;

  const decA = ra.decision as string | undefined;
  const decB = rb.decision as string | undefined;
  const qA = ra.quality_score as Record<string, number> | number | undefined;
  const qB = rb.quality_score as Record<string, number> | number | undefined;
  const compA = typeof qA === 'object' && qA ? qA.composite : (qA as number | undefined);
  const compB = typeof qB === 'object' && qB ? qB.composite : (qB as number | undefined);
  const costA = (ra.cost_breakdown as Record<string, unknown> | undefined)?.total_usd as number | undefined;
  const costB = (rb.cost_breakdown as Record<string, unknown> | undefined)?.total_usd as number | undefined;
  const elA  = ra.elapsed_seconds as number | undefined;
  const elB  = rb.elapsed_seconds as number | undefined;

  const sameDec = decA === decB;
  const qDelta  = compA != null && compB != null ? compB - compA : null;
  const costMult = costA && costB ? Math.max(costA, costB) / Math.min(costA, costB) : null;
  const fasterSide = elA && elB ? (elA < elB ? 'A' : 'B') : null;
  const speedPct  = elA && elB ? Math.round(Math.abs(elA - elB) / Math.max(elA, elB) * 100) : null;

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 text-sm">
      <p className="text-xs font-semibold text-slate-400 mb-3">Comparison Summary</p>
      <div className="flex gap-6 flex-wrap text-xs">
        <div>
          <span className="text-slate-400">Decision: </span>
          {sameDec
            ? <span className="text-slate-200">Both {decA} ✓</span>
            : <span className="text-yellow-400">A={decA} vs B={decB} ⚠</span>}
        </div>
        {qDelta != null && (
          <div>
            <span className="text-slate-400">Quality delta: </span>
            <span className={qDelta > 0 ? 'text-green-400' : qDelta < 0 ? 'text-red-400' : 'text-slate-200'}>
              {qDelta > 0 ? '+' : ''}{qDelta.toFixed(1)} (B vs A)
            </span>
          </div>
        )}
        {costMult != null && (
          <div>
            <span className="text-slate-400">Cost: </span>
            <span className="text-slate-200">
              {(costA ?? 0) > (costB ?? 0) ? 'A' : 'B'} is {costMult.toFixed(1)}x more expensive
            </span>
          </div>
        )}
        {fasterSide && speedPct != null && (
          <div>
            <span className="text-slate-400">Speed: </span>
            <span className="text-slate-200">Config {fasterSide} is {speedPct}% faster</span>
          </div>
        )}
      </div>
    </div>
  );
}

function ABCompareMode({ configs, active }: { configs: HybridConfigFull[]; active: string | null }) {
  const [ticker,    setTicker]    = useState('');
  const [tradeDate, setTradeDate] = useState(today());
  const [configA,   setConfigA]   = useState(active ?? (configs[0]?.name ?? ''));
  const [configB,   setConfigB]   = useState(configs[1]?.name ?? configs[0]?.name ?? '');
  const [publish,   setPublish]   = useState(false);
  const [abId,      setAbId]      = useState<string | null>(null);
  const [ab,        setAb]        = useState<ABCompareResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (active) setConfigA(active);
    if (configs.length >= 2) setConfigB(configs.find(c => c.name !== active)?.name ?? configs[0].name);
  }, [active, configs]);

  const pollAb = async (id: string) => {
    try {
      const res = await apiGet<ABCompareResponse>(`/test-run/ab/${id}`);
      setAb(res);
      if (res.status === 'complete' && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch { /* ignore */ }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ticker.trim()) return;
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    setSubmitting(true);
    setAb(null);
    setAbId(null);
    try {
      const res = await apiPost<{ ab_id: string }>('/test-run/ab', {
        ticker: ticker.toUpperCase(), trade_date: tradeDate, config_a: configA, config_b: configB, publish,
      });
      setAbId(res.ab_id);
      pollRef.current = setInterval(() => pollAb(res.ab_id), 3000);
    } finally { setSubmitting(false); }
  };

  const isRunning = ab?.status === 'running' || (abId && !ab);
  const isDone    = ab?.status === 'complete';

  return (
    <div className="space-y-6">
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Beaker className="w-4 h-4 text-purple-400" />
          <h3 className="text-sm font-semibold text-slate-200">A/B LLM Comparison</h3>
        </div>
        <form onSubmit={submit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Ticker *</label>
              <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="AAPL" required
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-white mono focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Trade Date</label>
              <input type="date" value={tradeDate} onChange={e => setTradeDate(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-white focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Config A</label>
              <select value={configA} onChange={e => setConfigA(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-slate-200">
                {configs.map(c => <option key={c.name} value={c.name}>{c.name}{c.name === active ? ' (current)' : ''}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Config B</label>
              <select value={configB} onChange={e => setConfigB(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-slate-700 border border-slate-600 rounded text-slate-200">
                {configs.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300">
              <input type="checkbox" checked={publish} onChange={e => setPublish(e.target.checked)} className="w-4 h-4" />
              Publish to Supabase
              {publish && <span className="text-xs text-yellow-400 flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> Writes to production</span>}
            </label>
          </div>
          <button type="submit" disabled={submitting || !!isRunning || !ticker.trim()}
            className="w-full py-2.5 text-sm font-semibold bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg">
            {submitting ? 'Submitting…' : isRunning ? 'Running A/B…' : 'Run A/B Comparison'}
          </button>
        </form>
      </div>

      {ab && isDone && <ComparisonSummary ab={ab} />}

      {ab && (
        <div className="flex gap-4">
          <ABSidePanel label="Config A" cfgName={ab.config_a.name} side={ab.config_a} />
          <ABSidePanel label="Config B" cfgName={ab.config_b.name} side={ab.config_b} />
        </div>
      )}

      {isRunning && !ab && (
        <div className="flex items-center gap-2 text-sm text-slate-400 p-4">
          <Loader2 className="w-4 h-4 animate-spin" /> Starting A/B comparison…
        </div>
      )}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

type Mode = 'single' | 'ab';

export function TestRunPage() {
  const [mode,    setMode]    = useState<Mode>('single');
  const [configs, setConfigs] = useState<HybridConfigFull[]>([]);
  const [active,  setActive]  = useState<string | null>(null);

  useEffect(() => {
    apiGet<HybridConfigsResponse>('/config/hybrid-configs')
      .then(r => { setConfigs(r.configs); setActive(r.active); })
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      {/* Mode toggle */}
      <div className="flex bg-slate-800 rounded-lg border border-slate-700 p-1 w-fit gap-1">
        {([['single', 'Single Run', FlaskConical], ['ab', 'A/B Compare', Beaker]] as const).map(([m, label, Icon]) => (
          <button key={m} onClick={() => setMode(m)}
            className={clsx('flex items-center gap-1.5 px-4 py-1.5 rounded text-sm transition-colors',
              mode === m ? 'bg-slate-700 text-white font-medium' : 'text-slate-400 hover:text-slate-200')}>
            <Icon className="w-4 h-4" /> {label}
          </button>
        ))}
      </div>

      {mode === 'single'
        ? <SingleRunMode configs={configs} active={active} />
        : <ABCompareMode configs={configs} active={active} />}
    </div>
  );
}
