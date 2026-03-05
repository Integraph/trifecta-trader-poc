import { useEffect, useRef, useState } from 'react';
import { useHealth, useAnalysesStats } from '../../api/hooks';
import { StatusDot } from '../shared/StatusDot';
import { WebSocketManager } from '../../lib/websocket';
import {
  formatUptime, formatDateTime, formatCurrency, formatScore,
} from '../../lib/utils';
import type { HealthResponse, AnalysesStats, LiveEvent } from '../../api/types';

// ── Subsystem card ─────────────────────────────────────────────────────────────

function SubCard({
  title, color, children,
}: { title: string; color: 'green' | 'yellow' | 'red' | 'gray'; children: React.ReactNode }) {
  const border = {
    green:  'border-green-600/40',
    yellow: 'border-yellow-500/40',
    red:    'border-red-600/40',
    gray:   'border-slate-600/40',
  }[color];

  return (
    <div className={`bg-slate-800 rounded-lg border ${border} p-4 space-y-2`}>
      <div className="flex items-center gap-2">
        <StatusDot color={color} pulse={color === 'green'} size="sm" />
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</span>
      </div>
      <div className="text-sm text-slate-200 space-y-1">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-slate-500 text-xs">{label}</span>
      <span className="text-xs font-medium text-slate-200">{String(value ?? '—')}</span>
    </div>
  );
}

// ── Subsystem cards ────────────────────────────────────────────────────────────

function HealthCards({ health }: { health: HealthResponse }) {
  const s = health.subsystems;

  const daemonColor  = s.daemon.status === 'running'  ? 'green' : 'red';
  const schedColor   = s.scheduler.status === 'running' ? 'green'
                      : s.scheduler.status === 'disabled' ? 'gray' : 'red';
  const queueColor   = s.queue_reader.status === 'running' ? 'green'
                      : s.queue_reader.status === 'stopped' ? 'red' : 'gray';
  const accColor     = s.accuracy_updater.status === 'ok' ? 'green'
                      : s.accuracy_updater.status === 'error' ? 'red' : 'gray';
  const supColor     = s.supabase.configured ? (s.supabase.write_enabled ? 'green' : 'yellow') : 'gray';
  const ollamaColor  = s.ollama.reachable ? 'green' : 'red';

  return (
    <div className="grid grid-cols-3 gap-4">
      <SubCard title="Daemon" color={daemonColor as 'green' | 'yellow' | 'red' | 'gray'}>
        <Row label="PID"    value={s.daemon.pid ?? 'N/A'} />
        <Row label="Uptime" value={formatUptime(s.daemon.uptime_seconds)} />
        <Row label="Status" value={s.daemon.status} />
      </SubCard>

      <SubCard title="Scheduler" color={schedColor as 'green' | 'yellow' | 'red' | 'gray'}>
        <Row label="Status"    value={s.scheduler.status} />
        <Row label="Last run"  value={formatDateTime(s.scheduler.last_run)} />
        <Row label="Next run"  value={formatDateTime(s.scheduler.next_run)} />
        <Row label="Last result" value={s.scheduler.last_run_result ?? '—'} />
      </SubCard>

      <SubCard title="Queue Reader" color={queueColor as 'green' | 'yellow' | 'red' | 'gray'}>
        <Row label="Pending"   value={s.queue_reader.pending_count} />
        <Row label="Completed today" value={s.queue_reader.completed_today} />
        <Row label="Last poll" value={formatDateTime(s.queue_reader.last_poll)} />
      </SubCard>

      <SubCard title="Accuracy" color={accColor as 'green' | 'yellow' | 'red' | 'gray'}>
        <Row label="Pending outcomes"  value={s.accuracy_updater.pending_outcomes} />
        <Row label="Complete outcomes" value={s.accuracy_updater.complete_outcomes} />
        <Row label="Last run"          value={formatDateTime(s.accuracy_updater.last_run)} />
      </SubCard>

      <SubCard title="Supabase" color={supColor as 'green' | 'yellow' | 'red' | 'gray'}>
        <Row label="Configured"    value={s.supabase.configured ? 'Yes' : 'No'} />
        <Row label="Write enabled" value={s.supabase.write_enabled ? 'Yes' : 'No'} />
      </SubCard>

      <SubCard title="Ollama" color={ollamaColor as 'green' | 'yellow' | 'red' | 'gray'}>
        <Row label="Reachable" value={s.ollama.reachable ? 'Yes' : 'No'} />
        <Row label="Model"     value={s.ollama.model ?? '—'} />
        {s.ollama.latency_ms != null && (
          <Row label="Latency" value={`${s.ollama.latency_ms}ms`} />
        )}
      </SubCard>
    </div>
  );
}

// ── Quick stats ────────────────────────────────────────────────────────────────

const DECISION_COLORS: Record<string, string> = {
  BUY:  'bg-green-600',
  SELL: 'bg-red-600',
  HOLD: 'bg-slate-500',
};

function QuickStats({ stats }: { stats: AnalysesStats }) {
  const decisions = stats.by_decision ?? {};
  const total_d   = Object.values(decisions).reduce((a, b) => a + b, 0) || 1;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">Quick Stats</h3>
      <div className="grid grid-cols-5 gap-4">
        <div>
          <p className="text-2xl font-bold text-white">{stats.total_analyses}</p>
          <p className="text-xs text-slate-400 mt-1">Total Analyses</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-white">{stats.analyses_today}</p>
          <p className="text-xs text-slate-400 mt-1">Today</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-white">{stats.unique_tickers}</p>
          <p className="text-xs text-slate-400 mt-1">Unique Tickers</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-white">{formatScore(stats.avg_quality_score)}</p>
          <p className="text-xs text-slate-400 mt-1">Avg Quality</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-white">{formatCurrency(stats.total_cost_usd, 2)}</p>
          <p className="text-xs text-slate-400 mt-1">Total Cost</p>
        </div>
      </div>
      {total_d > 0 && (
        <div className="mt-4">
          <p className="text-xs text-slate-400 mb-2">Decision Breakdown</p>
          <div className="flex h-2 rounded-full overflow-hidden gap-0.5">
            {Object.entries(decisions).map(([d, n]) => (
              <div
                key={d}
                className={DECISION_COLORS[d] ?? 'bg-slate-600'}
                style={{ width: `${(n / total_d) * 100}%` }}
                title={`${d}: ${n}`}
              />
            ))}
          </div>
          <div className="flex gap-4 mt-2">
            {Object.entries(decisions).map(([d, n]) => (
              <span key={d} className="text-xs text-slate-400">
                <span className="text-slate-200">{d}</span> {n}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Event feed ─────────────────────────────────────────────────────────────────

const EVENT_COLORS: Record<string, string> = {
  'scheduler.': 'text-blue-400',
  'queue.':     'text-purple-400',
  'accuracy.':  'text-green-400',
  'test_run.':  'text-orange-400',
};

function eventColor(type: string) {
  for (const [prefix, cls] of Object.entries(EVENT_COLORS)) {
    if (type.startsWith(prefix)) return cls;
  }
  return 'text-slate-400';
}

function eventSummary(ev: LiveEvent): string {
  const d = ev.data;
  switch (ev.event) {
    case 'scheduler.run_started':   return 'Scheduler scan started';
    case 'scheduler.run_completed':
      if (d.error) return `Scheduler failed: ${d.error}`;
      return `Scheduler completed: ${d.tickers_processed ?? 0} tickers, ${Number(d.elapsed_seconds ?? 0).toFixed(0)}s`;
    case 'queue.candidate_picked':  return `Queue picked: ${d.ticker} (${d.priority})`;
    case 'queue.analysis_completed':return `Queue completed: ${d.ticker} → ${d.decision}`;
    case 'queue.analysis_failed':   return `Queue failed: ${d.ticker} — ${d.error}`;
    case 'accuracy.update_started': return `Accuracy update started (${d.pending} pending)`;
    case 'accuracy.update_completed': return 'Accuracy update complete';
    default: return ev.event;
  }
}

function EventFeed({ events }: { events: LiveEvent[] }) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Live Event Feed</h3>
      {events.length === 0 ? (
        <p className="text-xs text-slate-500 py-4 text-center">Waiting for events…</p>
      ) : (
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {[...events].reverse().map((ev, i) => (
            <div key={i} className="flex items-start gap-3 text-xs py-1 border-b border-slate-700/50 last:border-0">
              <span className="text-slate-500 mono shrink-0">
                {new Date(ev.timestamp).toLocaleTimeString('en-US', { hour12: false })}
              </span>
              <span className={`shrink-0 ${eventColor(ev.event)}`}>{ev.event}</span>
              <span className="text-slate-300">{eventSummary(ev)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export function DashboardPage() {
  const { data: health }  = useHealth(10_000);
  const { data: stats }   = useAnalysesStats(30_000);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const wsRef = useRef<WebSocketManager | null>(null);

  useEffect(() => {
    const ws = new WebSocketManager('ws://localhost:8420/ws/events', (msg) => {
      setEvents(prev => [...prev.slice(-19), msg as LiveEvent]);
    });
    ws.connect();
    wsRef.current = ws;
    return () => ws.disconnect();
  }, []);

  return (
    <div className="space-y-6">
      {health ? (
        <HealthCards health={health} />
      ) : (
        <div className="grid grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-28 bg-slate-800 rounded-lg border border-slate-700 animate-pulse" />
          ))}
        </div>
      )}
      {stats && <QuickStats stats={stats} />}
      <EventFeed events={events} />
    </div>
  );
}
