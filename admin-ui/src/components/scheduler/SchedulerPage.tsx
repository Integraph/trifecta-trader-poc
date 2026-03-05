import { useEffect, useRef, useState } from 'react';
import { Play, RefreshCw } from 'lucide-react';
import { useSchedulerStatus } from '../../api/hooks';
import { apiGet, apiPost } from '../../api/client';
import { TaskPoller } from '../shared/TaskPoller';
import { DataTable, type Column } from '../shared/DataTable';
import { StatusDot } from '../shared/StatusDot';
import {
  formatDateTime, formatElapsed, formatScore, formatCurrency,
  formatCountdown,
} from '../../lib/utils';
import type { SchedulerHistoryItem, SchedulerStatus } from '../../api/types';

// ── Countdown that re-renders every second ────────────────────────────────────

function Countdown({ iso }: { iso: string | null | undefined }) {
  const [label, setLabel] = useState(() => formatCountdown(iso));
  useEffect(() => {
    const t = setInterval(() => setLabel(formatCountdown(iso)), 1000);
    return () => clearInterval(t);
  }, [iso]);
  return <span>{label}</span>;
}

// ── Status panel ──────────────────────────────────────────────────────────────

function StatusPanel({ status }: { status: SchedulerStatus }) {
  const color = status.is_running ? 'green' : status.enabled ? 'yellow' : 'gray';
  const lr    = status.last_run;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <div className="flex items-center gap-2 mb-4">
        <StatusDot color={color as 'green' | 'yellow' | 'red' | 'gray'} pulse={status.is_running} />
        <span className="text-sm font-semibold text-slate-200">
          {status.is_running ? 'Running' : status.enabled ? 'Waiting' : 'Disabled'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
        <div className="text-slate-500 text-xs">Schedule</div>
        <div className="text-slate-200 text-xs">{status.schedule}</div>

        <div className="text-slate-500 text-xs">Next run</div>
        <div className="text-slate-200 text-xs font-medium">
          <Countdown iso={status.next_run} />
          {status.next_run && (
            <span className="text-slate-500 ml-2">({formatDateTime(status.next_run)})</span>
          )}
        </div>

        <div className="text-slate-500 text-xs">Watchlist</div>
        <div className="text-slate-200 text-xs">{status.config.watchlist}</div>

        <div className="text-slate-500 text-xs">Hybrid config</div>
        <div className="text-slate-200 text-xs">{status.config.hybrid_config}</div>

        <div className="text-slate-500 text-xs">Publish</div>
        <div className="text-xs">
          <span className={status.config.publish ? 'text-green-400' : 'text-slate-400'}>
            {status.config.publish ? 'Yes' : 'No'}
          </span>
        </div>
      </div>

      {lr && (
        <div className="mt-4 pt-4 border-t border-slate-700 space-y-1">
          <p className="text-xs font-medium text-slate-400 mb-2">Last Run</p>
          <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-xs">
            <span className="text-slate-500">Time</span>
            <span className="text-slate-200">{formatDateTime(lr.timestamp)}</span>
            <span className="text-slate-500">Result</span>
            <span>
              <span className={lr.result === 'success' ? 'text-green-400' : 'text-red-400'}>
                {lr.result}
              </span>
            </span>
            <span className="text-slate-500">Tickers</span>
            <span className="text-slate-200">{lr.tickers_processed}</span>
            <span className="text-slate-500">Elapsed</span>
            <span className="text-slate-200">{formatElapsed(lr.elapsed_seconds)}</span>
            {lr.decisions && Object.entries(lr.decisions).map(([d, n]) => (
              <>
                <span key={`${d}-l`} className="text-slate-500">{d}</span>
                <span key={`${d}-v`} className="text-slate-200">{n}</span>
              </>
            ))}
            {lr.error && (
              <>
                <span className="text-slate-500">Error</span>
                <span className="text-red-400 break-all">{lr.error}</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── History table ─────────────────────────────────────────────────────────────

const HIST_COLS: Column<SchedulerHistoryItem & Record<string, unknown>>[] = [
  { key: 'trade_date', label: 'Date', sortable: true },
  { key: 'total_analyses', label: 'Tickers', sortable: true },
  { key: 'buy',  label: 'BUY',  render: (r) => <span className="text-green-400">{(r.decisions as Record<string, number>)?.BUY ?? 0}</span> },
  { key: 'sell', label: 'SELL', render: (r) => <span className="text-red-400">{(r.decisions as Record<string, number>)?.SELL ?? 0}</span> },
  { key: 'hold', label: 'HOLD', render: (r) => <span className="text-slate-400">{(r.decisions as Record<string, number>)?.HOLD ?? 0}</span> },
  { key: 'avg_quality', label: 'Avg Quality', sortable: true, render: (r) => <>{formatScore(r.avg_quality as number)}</> },
  { key: 'elapsed_seconds', label: 'Elapsed', sortable: true, render: (r) => <>{formatElapsed(r.elapsed_seconds as number)}</> },
  { key: 'total_cost_usd', label: 'Cost', sortable: true, render: (r) => <>{formatCurrency(r.total_cost_usd as number)}</> },
];

// ── Main page ─────────────────────────────────────────────────────────────────

export function SchedulerPage() {
  const { data: status, refresh } = useSchedulerStatus(15_000);
  const [taskId, setTaskId]       = useState<string | null>(null);
  const [triggering, setTriggering] = useState(false);
  const [days, setDays]           = useState(7);
  const [history, setHistory]     = useState<SchedulerHistoryItem[]>([]);
  const prevDays = useRef(days);

  useEffect(() => {
    prevDays.current = days;
    apiGet<{ history: SchedulerHistoryItem[] }>('/scheduler/history', { days })
      .then(r => setHistory(r.history))
      .catch(() => {});
  }, [days]);

  const trigger = async () => {
    setTriggering(true);
    try {
      const res = await apiPost<{ task_id: string }>('/scheduler/trigger');
      setTaskId(res.task_id);
    } finally {
      setTriggering(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-slate-200">Scheduler</h2>
        <div className="flex gap-2">
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
          <button
            onClick={trigger}
            disabled={triggering || !!taskId}
            className="flex items-center gap-1.5 px-4 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg font-medium"
          >
            <Play className="w-3.5 h-3.5" />
            Run Watchlist Now
          </button>
        </div>
      </div>

      {status && <StatusPanel status={status} />}

      {taskId && (
        <TaskPoller
          taskId={taskId}
          pollInterval={2000}
          label="Watchlist scan"
          onComplete={() => { setTaskId(null); refresh(); }}
        />
      )}

      <div className="bg-slate-800 rounded-lg border border-slate-700">
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-slate-300">Run History</h3>
          <select
            value={days}
            onChange={e => setDays(Number(e.target.value))}
            className="text-xs bg-slate-700 text-slate-200 border border-slate-600 rounded px-2 py-1"
          >
            {[7, 14, 30].map(d => <option key={d} value={d}>{d} days</option>)}
          </select>
        </div>
        <DataTable
          columns={HIST_COLS}
          rows={history as unknown as (SchedulerHistoryItem & Record<string, unknown>)[]}
          keyFn={r => r.trade_date}
          emptyMsg="No history yet"
        />
      </div>
    </div>
  );
}
