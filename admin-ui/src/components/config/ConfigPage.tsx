import { useEffect, useState } from 'react';
import { Save, Plus, X } from 'lucide-react';
import { apiGet, apiPut } from '../../api/client';
import { clsx } from '../../lib/utils';
import type { AutomationConfig, SupabaseConfig, WatchlistItem, HybridConfig } from '../../api/types';

// ── Save result badge ─────────────────────────────────────────────────────────

function SaveResult({ applied, restart }: { applied: string[]; restart: string[] }) {
  if (!applied.length && !restart.length) return null;
  return (
    <div className="text-xs space-y-1 mt-2">
      {applied.map(k => (
        <span key={k} className="inline-block mr-2 px-2 py-0.5 rounded bg-green-700 text-green-100">
          ✓ Applied: {k}
        </span>
      ))}
      {restart.map(k => (
        <span key={k} className="inline-block mr-2 px-2 py-0.5 rounded bg-yellow-700 text-yellow-100">
          ↻ Restart required: {k}
        </span>
      ))}
    </div>
  );
}

// ── Field row ─────────────────────────────────────────────────────────────────

function FieldRow({
  label, value, onChange, type = 'text',
}: {
  label: string;
  value: string | number | boolean;
  onChange: (v: string | number | boolean) => void;
  type?: 'text' | 'number' | 'boolean';
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2 border-b border-slate-700/50 last:border-0">
      <span className="text-xs text-slate-400 w-48 shrink-0">{label}</span>
      {type === 'boolean' ? (
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={e => onChange(e.target.checked)}
            className="w-4 h-4 rounded"
          />
          <span className="text-xs text-slate-300">{Boolean(value) ? 'Enabled' : 'Disabled'}</span>
        </label>
      ) : (
        <input
          type={type}
          value={String(value ?? '')}
          onChange={e => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
          className="flex-1 max-w-xs px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-white focus:outline-none focus:border-blue-500"
        />
      )}
    </div>
  );
}

// ── Automation config panel ───────────────────────────────────────────────────

function AutomationPanel() {
  const [draft,  setDraft]  = useState<AutomationConfig>({});
  const [result, setResult] = useState<{ applied: string[]; restart: string[] } | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    apiGet<AutomationConfig>('/config/automation')
      .then(r => setDraft(r))
      .catch(() => {});
  }, []);

  const set = (path: string[], value: unknown) => {
    setDraft(prev => {
      const next = JSON.parse(JSON.stringify(prev));
      let cur: Record<string, unknown> = next;
      for (let i = 0; i < path.length - 1; i++) {
        if (!cur[path[i]]) cur[path[i]] = {};
        cur = cur[path[i]] as Record<string, unknown>;
      }
      cur[path[path.length - 1]] = value;
      return next;
    });
  };

  const save = async () => {
    setSaving(true);
    try {
      const res = await apiPut<{ applied: Record<string, boolean>; requires_restart: string[] }>('/config/automation', draft);
      setResult({ applied: Object.keys(res.applied), restart: res.requires_restart });
    } finally {
      setSaving(false);
    }
  };

  const s = draft.scheduler     ?? {};
  const q = draft.queue_reader  ?? {};
  const a = draft.accuracy      ?? {};
  const api = draft.admin_api   ?? {};

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-200">Automation Config</h3>
        <button
          onClick={save}
          disabled={saving}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded"
        >
          <Save className="w-3 h-3" /> Save
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Scheduler</p>
          <FieldRow label="Enabled"      value={Boolean(s.enabled)}   onChange={v => set(['scheduler', 'enabled'], v)} type="boolean" />
          <FieldRow label="Hour"         value={s.hour ?? 8}          onChange={v => set(['scheduler', 'hour'], v)} type="number" />
          <FieldRow label="Minute"       value={s.minute ?? 30}       onChange={v => set(['scheduler', 'minute'], v)} type="number" />
          <FieldRow label="Hybrid config" value={s.hybrid_config ?? ''} onChange={v => set(['scheduler', 'hybrid_config'], v)} />
          <FieldRow label="Watchlist"    value={s.watchlist ?? ''}    onChange={v => set(['scheduler', 'watchlist'], v)} />
          <FieldRow label="Publish"      value={Boolean(s.publish)}   onChange={v => set(['scheduler', 'publish'], v)} type="boolean" />
        </div>
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Queue Reader</p>
          <FieldRow label="Enabled"            value={Boolean(q.enabled)}                onChange={v => set(['queue_reader', 'enabled'], v)} type="boolean" />
          <FieldRow label="Poll interval (s)"  value={q.poll_interval_seconds ?? 30}     onChange={v => set(['queue_reader', 'poll_interval_seconds'], v)} type="number" />
          <FieldRow label="Max retries"        value={q.max_retries ?? 2}                onChange={v => set(['queue_reader', 'max_retries'], v)} type="number" />
          <FieldRow label="Cooldown (s)"       value={q.cooldown_seconds ?? 60}          onChange={v => set(['queue_reader', 'cooldown_seconds'], v)} type="number" />
        </div>
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Accuracy</p>
          <FieldRow label="Enabled"             value={Boolean(a.enabled)}          onChange={v => set(['accuracy', 'enabled'], v)} type="boolean" />
          <FieldRow label="Backfill on start"   value={Boolean(a.backfill_on_first_run)} onChange={v => set(['accuracy', 'backfill_on_first_run'], v)} type="boolean" />
        </div>
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Admin API</p>
          <FieldRow label="Port" value={api.port ?? 8420} onChange={v => set(['admin_api', 'port'], v)} type="number" />
        </div>
      </div>
      {result && <SaveResult applied={result.applied} restart={result.restart} />}
    </div>
  );
}

// ── Supabase config panel ─────────────────────────────────────────────────────

function SupabasePanel() {
  const [draft,  setDraft]  = useState<SupabaseConfig>({});
  const [result, setResult] = useState<{ applied: string[]; restart: string[] } | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    apiGet<SupabaseConfig>('/config/supabase')
      .then(r => setDraft(r))
      .catch(() => {});
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const res = await apiPut<{ applied: Record<string, boolean>; requires_restart: string[] }>('/config/supabase', draft);
      setResult({ applied: Object.keys(res.applied), restart: res.requires_restart });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-200">Supabase Config</h3>
        <button
          onClick={save}
          disabled={saving}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded"
        >
          <Save className="w-3 h-3" /> Save
        </button>
      </div>
      <FieldRow label="Write enabled"    value={Boolean(draft.write_enabled)} onChange={v => setDraft(p => ({ ...p, write_enabled: v as boolean }))} type="boolean" />
      <FieldRow label="Signal TTL (hrs)" value={draft.signal_ttl_hours ?? 48} onChange={v => setDraft(p => ({ ...p, signal_ttl_hours: v as number }))} type="number" />
      {draft.table_name && (
        <div className="flex justify-between py-2 text-xs">
          <span className="text-slate-400">Table name</span>
          <span className="text-slate-200 mono">{draft.table_name}</span>
        </div>
      )}
      {result && <SaveResult applied={result.applied} restart={result.restart} />}
    </div>
  );
}

// ── Watchlist manager ─────────────────────────────────────────────────────────

function WatchlistManager() {
  const [items,    setItems]    = useState<WatchlistItem[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [tickers,  setTickers]  = useState<string[]>([]);
  const [newTicker, setNewTicker] = useState('');
  const [saving,   setSaving]   = useState(false);
  const [toast,    setToast]    = useState<string | null>(null);

  const load = () =>
    apiGet<{ watchlists: WatchlistItem[] }>('/config/watchlists')
      .then(r => setItems(r.watchlists))
      .catch(() => {});

  useEffect(() => { load(); }, []);

  const select = (name: string) => {
    const item = items.find(i => i.name === name);
    setSelected(name);
    setTickers(item?.tickers ?? []);
  };

  const addTicker = () => {
    const t = newTicker.trim().toUpperCase();
    if (t && !tickers.includes(t)) { setTickers(p => [...p, t]); setNewTicker(''); }
  };

  const save = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await apiPut(`/config/watchlists/${selected}`, { tickers });
      setToast('Saved');
      setTimeout(() => setToast(null), 2000);
      load();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <h3 className="text-sm font-semibold text-slate-200 mb-4">Watchlist Manager</h3>
      <div className="flex gap-2 flex-wrap mb-4">
        {items.map(i => (
          <button
            key={i.name}
            onClick={() => select(i.name)}
            className={clsx(
              'px-3 py-1 text-xs rounded border transition-colors',
              selected === i.name
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600',
            )}
          >
            {i.name} ({i.tickers.length})
          </button>
        ))}
      </div>
      {selected && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {tickers.map(t => (
              <span key={t} className="flex items-center gap-1 px-2 py-0.5 bg-slate-700 rounded text-xs mono text-white">
                {t}
                <button onClick={() => setTickers(p => p.filter(x => x !== t))} className="text-slate-400 hover:text-red-400">
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              value={newTicker}
              onChange={e => setNewTicker(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === 'Enter' && addTicker()}
              placeholder="Add ticker…"
              className="px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-white mono w-28 focus:outline-none"
            />
            <button onClick={addTicker} className="p-1 bg-slate-600 hover:bg-slate-500 rounded text-slate-200">
              <Plus className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={save}
              disabled={saving}
              className="flex items-center gap-1 px-3 py-1 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded"
            >
              <Save className="w-3 h-3" /> Save
            </button>
            {toast && <span className="text-xs text-green-400 self-center">{toast}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Hybrid configs table ──────────────────────────────────────────────────────

function HybridConfigsPanel() {
  const [configs, setConfigs] = useState<HybridConfig[]>([]);
  const [active,  setActive]  = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ configs: HybridConfig[]; active: string | null }>('/config/hybrid-configs')
      .then(r => { setConfigs(r.configs); setActive(r.active); })
      .catch(() => {});
  }, []);

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700">
      <div className="px-5 py-3 border-b border-slate-700">
        <h3 className="text-sm font-semibold text-slate-200">Hybrid LLM Configurations</h3>
        <p className="text-xs text-slate-500 mt-0.5">Read-only — edit in code</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-700 text-left text-slate-400 uppercase tracking-wider">
              {['Name', 'Tool Model', 'Quick Model', 'Deep Model', 'Active'].map(h => (
                <th key={h} className="px-4 py-2">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {configs.map(c => (
              <tr key={c.name} className={clsx('border-b border-slate-800', c.name === active && 'bg-blue-900/20')}>
                <td className="px-4 py-2 mono font-medium text-white">{c.name}</td>
                <td className="px-4 py-2 text-slate-300">{c.tool_model ?? '—'}</td>
                <td className="px-4 py-2 text-slate-300">{c.reasoning_quick_model ?? '—'}</td>
                <td className="px-4 py-2 text-slate-300">{c.reasoning_deep_model ?? '—'}</td>
                <td className="px-4 py-2">
                  {c.name === active && <span className="px-1.5 py-0.5 rounded bg-blue-700 text-blue-100 text-xs">active</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export function ConfigPage() {
  return (
    <div className="space-y-6">
      <AutomationPanel />
      <SupabasePanel />
      <WatchlistManager />
      <HybridConfigsPanel />
    </div>
  );
}
