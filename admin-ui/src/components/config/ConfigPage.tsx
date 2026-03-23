import { useEffect, useState } from 'react';
import { Save, Plus, X, Copy, Trash2, CheckCircle2, XCircle, Loader2, FlaskConical } from 'lucide-react';
import { apiGet, apiPost, apiPut, apiDelete } from '../../api/client';
import { InfoTooltip } from '../shared/InfoTooltip';
import { SETTINGS_HELP } from './config-help';
import { clsx } from '../../lib/utils';
import type {
  AutomationConfig, SupabaseConfig, WatchlistItem,
  HybridConfigFull, HybridConfigsResponse, SanityCheckResult,
} from '../../api/types';

// ── Shared FieldRow with optional tooltip ─────────────────────────────────────

function FieldRow({
  label, value, onChange, type = 'text', info,
}: {
  label:     string;
  value:     string | number | boolean;
  onChange:  (v: string | number | boolean) => void;
  type?:     'text' | 'number' | 'boolean';
  info?:     string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2 border-b border-slate-700/50 last:border-0">
      <span className="text-xs text-slate-400 w-52 shrink-0 flex items-center gap-1.5">
        {label}
        {info && <InfoTooltip text={info} />}
      </span>
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

// ── Save result badge ─────────────────────────────────────────────────────────

function SaveResult({ applied, restart }: { applied: string[]; restart: string[] }) {
  if (!applied.length && !restart.length) return null;
  return (
    <div className="text-xs space-y-1 mt-2 flex flex-wrap gap-1">
      {applied.map(k => (
        <span key={k} className="px-2 py-0.5 rounded bg-green-700 text-green-100">✓ Applied: {k}</span>
      ))}
      {restart.map(k => (
        <span key={k} className="px-2 py-0.5 rounded bg-yellow-700 text-yellow-100">↻ Restart: {k}</span>
      ))}
    </div>
  );
}

// ── Automation config panel ───────────────────────────────────────────────────

function AutomationPanel() {
  const [draft,  setDraft]  = useState<AutomationConfig>({});
  const [result, setResult] = useState<{ applied: string[]; restart: string[] } | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    apiGet<AutomationConfig>('/config/automation').then(r => setDraft(r)).catch(() => {});
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

  const s = draft.scheduler    ?? {};
  const q = draft.queue_reader ?? {};
  const a = draft.accuracy     ?? {};
  const api = draft.admin_api  ?? {};

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-200">Automation Config</h3>
        <button onClick={save} disabled={saving}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded">
          <Save className="w-3 h-3" /> Save
        </button>
      </div>
      <div className="space-y-4">
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Scheduler</p>
          <FieldRow label="Enabled"       value={Boolean(s.enabled)}   onChange={v => set(['scheduler','enabled'], v)} type="boolean" info={SETTINGS_HELP['scheduler.enabled']} />
          <FieldRow label="Hour (24h)"    value={s.watchlist_hour ?? 8}   onChange={v => set(['scheduler','watchlist_hour'], v)} type="number"  info={SETTINGS_HELP['scheduler.watchlist_hour']} />
          <FieldRow label="Minute"        value={s.watchlist_minute ?? 30} onChange={v => set(['scheduler','watchlist_minute'], v)} type="number"  info={SETTINGS_HELP['scheduler.watchlist_minute']} />
          <FieldRow label="Hybrid config" value={s.hybrid_config ?? ''} onChange={v => set(['scheduler','hybrid_config'], v)} info={SETTINGS_HELP['scheduler.hybrid_config']} />
          <FieldRow label="Watchlist"     value={s.watchlist ?? ''}    onChange={v => set(['scheduler','watchlist'], v)} info={SETTINGS_HELP['scheduler.watchlist']} />
          <FieldRow label="Publish"       value={Boolean(s.publish)}   onChange={v => set(['scheduler','publish'], v)} type="boolean" info={SETTINGS_HELP['scheduler.publish']} />
        </div>
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Queue Reader</p>
          <FieldRow label="Enabled"           value={Boolean(q.enabled)}             onChange={v => set(['queue_reader','enabled'], v)} type="boolean" info={SETTINGS_HELP['queue_reader.enabled']} />
          <FieldRow label="Poll interval (s)" value={q.poll_interval_seconds ?? 30}  onChange={v => set(['queue_reader','poll_interval_seconds'], v)} type="number" info={SETTINGS_HELP['queue_reader.poll_interval']} />
          <FieldRow label="Max retries"       value={q.max_retries ?? 2}             onChange={v => set(['queue_reader','max_retries'], v)} type="number" info={SETTINGS_HELP['queue_reader.max_retries']} />
          <FieldRow label="Cooldown (s)"      value={q.cooldown_seconds ?? 60}       onChange={v => set(['queue_reader','cooldown_seconds'], v)} type="number" info={SETTINGS_HELP['queue_reader.cooldown']} />
        </div>
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Accuracy</p>
          <FieldRow label="Enabled"          value={Boolean(a.enabled)}               onChange={v => set(['accuracy','enabled'], v)} type="boolean" info={SETTINGS_HELP['accuracy.enabled']} />
          <FieldRow label="Backfill on start" value={Boolean(a.backfill_on_first_run)} onChange={v => set(['accuracy','backfill_on_first_run'], v)} type="boolean" info={SETTINGS_HELP['accuracy.backfill']} />
        </div>
        <div>
          <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Admin API</p>
          <FieldRow label="Port" value={api.port ?? 8420} onChange={v => set(['admin_api','port'], v)} type="number" info={SETTINGS_HELP['admin_api.port']} />
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
    apiGet<SupabaseConfig>('/config/supabase').then(r => setDraft(r)).catch(() => {});
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
        <button onClick={save} disabled={saving}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded">
          <Save className="w-3 h-3" /> Save
        </button>
      </div>
      <FieldRow label="Write enabled"    value={Boolean(draft.write_enabled)} onChange={v => setDraft(p => ({ ...p, write_enabled: v as boolean }))} type="boolean" info={SETTINGS_HELP['supabase.write_enabled']} />
      <FieldRow label="Signal TTL (hrs)" value={draft.signal_ttl_hours ?? 48} onChange={v => setDraft(p => ({ ...p, signal_ttl_hours: v as number }))} type="number" info={SETTINGS_HELP['supabase.signal_ttl']} />
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
    apiGet<{ watchlists: WatchlistItem[] }>('/config/watchlists').then(r => setItems(r.watchlists)).catch(() => {});

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
      setToast('Saved'); setTimeout(() => setToast(null), 2000);
      load();
    } finally { setSaving(false); }
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-5">
      <div className="flex items-center gap-2 mb-4">
        <h3 className="text-sm font-semibold text-slate-200">Watchlist Manager</h3>
        <InfoTooltip text={SETTINGS_HELP['watchlists.section']} />
      </div>
      <div className="flex gap-2 flex-wrap mb-4">
        {items.map(i => (
          <button key={i.name} onClick={() => select(i.name)}
            className={clsx('px-3 py-1 text-xs rounded border transition-colors',
              selected === i.name ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600')}>
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
            <input value={newTicker} onChange={e => setNewTicker(e.target.value.toUpperCase())} onKeyDown={e => e.key === 'Enter' && addTicker()}
              placeholder="Add ticker…" className="px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-white mono w-28 focus:outline-none" />
            <button onClick={addTicker} className="p-1 bg-slate-600 hover:bg-slate-500 rounded text-slate-200"><Plus className="w-3.5 h-3.5" /></button>
            <button onClick={save} disabled={saving}
              className="flex items-center gap-1 px-3 py-1 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded">
              <Save className="w-3 h-3" /> Save
            </button>
            {toast && <span className="text-xs text-green-400 self-center">{toast}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ── LLM config editor ─────────────────────────────────────────────────────────

const EMPTY_CFG: Omit<HybridConfigFull, 'name'> = {
  tool_provider: 'anthropic', tool_model: 'claude-haiku-4-5-20251001',
  reasoning_quick_provider: 'ollama', reasoning_quick_model: 'qwen2.5:14b',
  reasoning_deep_provider: 'anthropic', reasoning_deep_model: 'claude-sonnet-4-5-20250929',
  enhance_local: false, enhance_style: 'financial_analysis',
  enhance_deep: false, enhance_deep_style: 'execution_params_only',
};

interface SanityCheckState {
  loading: boolean;
  result: SanityCheckResult | null;
  error: string | null;
}

function SanityCheckCard({ result }: { result: SanityCheckResult }) {
  const slots = [
    { key: 'tool_calling',    label: 'Tool Calling' },
    { key: 'reasoning_quick', label: 'Reasoning Quick' },
    { key: 'reasoning_deep',  label: 'Reasoning Deep' },
  ] as const;

  const passed = Object.values(result.checks).filter(c => c.status === 'pass').length;

  return (
    <div className="border border-slate-600 rounded-lg p-4 space-y-2 bg-slate-900">
      <p className="text-xs font-semibold text-slate-300 mb-2">Sanity Check Results</p>
      {slots.map(({ key, label }) => {
        const c = result.checks[key];
        return (
          <div key={key} className="flex items-start gap-2 text-xs">
            {c.status === 'pass'
              ? <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5 shrink-0" />
              : c.status === 'skip'
              ? <span className="w-4 h-4 mt-0.5 shrink-0 text-slate-500 text-center">—</span>
              : <XCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />}
            <div>
              <span className="text-slate-200 font-medium">{label}</span>
              <span className="text-slate-400 ml-2">{c.provider}/{c.model}</span>
              {c.latency_ms != null && <span className="text-slate-500 ml-2">{c.latency_ms}ms</span>}
              {c.error && <p className="text-red-400 mt-0.5">{c.error}</p>}
            </div>
          </div>
        );
      })}
      <div className="pt-2 border-t border-slate-700 text-xs">
        <span className="text-slate-400">Overall: </span>
        <span className={result.overall === 'pass' ? 'text-green-400' : result.overall === 'partial' ? 'text-yellow-400' : 'text-red-400'}>
          {result.overall.toUpperCase()} ({passed}/3 passed)
        </span>
      </div>
    </div>
  );
}

function LLMConfigEditor({
  cfg, active, providers, styles,
  onSave, onDelete, onClone, onSanityCheck,
}: {
  cfg:            HybridConfigFull;
  active:         boolean;
  providers:      string[];
  styles:         string[];
  onSave:         (updated: HybridConfigFull) => Promise<void>;
  onDelete:       () => Promise<void>;
  onClone:        (newName: string) => Promise<void>;
  onSanityCheck:  () => Promise<SanityCheckResult>;
}) {
  const [draft,       setDraft]       = useState<HybridConfigFull>({ ...cfg });
  const [saving,      setSaving]      = useState(false);
  const [toast,       setToast]       = useState<string | null>(null);
  const [cloneInput,  setCloneInput]  = useState('');
  const [showClone,   setShowClone]   = useState(false);
  const [sanity,      setSanity]      = useState<SanityCheckState>({ loading: false, result: null, error: null });
  const [confirming,  setConfirming]  = useState(false);

  // Sync when a different config is selected
  useEffect(() => { setDraft({ ...cfg }); setToast(null); setSanity({ loading: false, result: null, error: null }); }, [cfg.name]);

  const set = <K extends keyof HybridConfigFull>(k: K, v: HybridConfigFull[K]) =>
    setDraft(p => ({ ...p, [k]: v }));

  const save = async () => {
    setSaving(true);
    try { await onSave(draft); setToast('Saved'); setTimeout(() => setToast(null), 2000); }
    catch (e) { setToast(`Error: ${e instanceof Error ? e.message : String(e)}`); }
    finally { setSaving(false); }
  };

  const runSanity = async () => {
    setSanity({ loading: true, result: null, error: null });
    try { const r = await onSanityCheck(); setSanity({ loading: false, result: r, error: null }); }
    catch (e) { setSanity({ loading: false, result: null, error: e instanceof Error ? e.message : String(e) }); }
  };

  const doDelete = async () => {
    if (!confirming) { setConfirming(true); return; }
    await onDelete();
    setConfirming(false);
  };

  const doClone = async () => {
    if (!cloneInput.trim()) return;
    await onClone(cloneInput.trim());
    setCloneInput(''); setShowClone(false);
  };

  const SlotSection = ({ title, pKey, mKey }: { title: string; pKey: keyof HybridConfigFull; mKey: keyof HybridConfigFull }) => (
    <div className="border border-slate-700 rounded-lg p-3 space-y-2">
      <p className="text-xs font-semibold text-slate-400">{title}</p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-slate-500 mb-1 block">Provider</label>
          <select value={draft[pKey] as string} onChange={e => set(pKey, e.target.value as HybridConfigFull[typeof pKey])}
            className="w-full px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-slate-200">
            {providers.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-slate-500 mb-1 block">Model</label>
          <input value={draft[mKey] as string} onChange={e => set(mKey, e.target.value as HybridConfigFull[typeof mKey])}
            className="w-full px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-white mono focus:outline-none focus:border-blue-500" />
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <h4 className="text-sm font-semibold text-white">{cfg.name}</h4>
        {active && <span className="px-1.5 py-0.5 text-xs bg-blue-700 text-blue-100 rounded">active</span>}
      </div>

      <SlotSection title="Tool Calling"     pKey="tool_provider"            mKey="tool_model" />
      <SlotSection title="Reasoning Quick"  pKey="reasoning_quick_provider" mKey="reasoning_quick_model" />
      <SlotSection title="Reasoning Deep"   pKey="reasoning_deep_provider"  mKey="reasoning_deep_model" />

      <div className="border border-slate-700 rounded-lg p-3 space-y-2">
        <p className="text-xs font-semibold text-slate-400">Enhancement</p>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={draft.enhance_local} onChange={e => set('enhance_local', e.target.checked)} className="w-3.5 h-3.5" />
            <span className="text-slate-300">Enhance local</span>
          </label>
          <div>
            <label className="text-xs text-slate-500 mb-1 block">Style</label>
            <select value={draft.enhance_style} onChange={e => set('enhance_style', e.target.value)}
              className="w-full px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-slate-200">
              {styles.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={draft.enhance_deep} onChange={e => set('enhance_deep', e.target.checked)} className="w-3.5 h-3.5" />
            <span className="text-slate-300">Enhance deep</span>
          </label>
          <div>
            <label className="text-xs text-slate-500 mb-1 block">Deep style</label>
            <select value={draft.enhance_deep_style} onChange={e => set('enhance_deep_style', e.target.value)}
              className="w-full px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-slate-200">
              {styles.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-2 flex-wrap">
        <button onClick={runSanity} disabled={sanity.loading}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-200 rounded">
          {sanity.loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FlaskConical className="w-3.5 h-3.5" />}
          Sanity Check
        </button>
        <button onClick={() => setShowClone(s => !s)}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 rounded">
          <Copy className="w-3.5 h-3.5" /> Clone
        </button>
        <button onClick={save} disabled={saving}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded font-medium">
          <Save className="w-3.5 h-3.5" /> Save
        </button>
        <button onClick={doDelete} disabled={active}
          title={active ? 'Cannot delete active config' : (confirming ? 'Click again to confirm delete' : 'Delete')}
          className={clsx(
            'flex items-center gap-1.5 px-3 py-1.5 text-xs rounded',
            active ? 'opacity-40 cursor-not-allowed text-slate-500 bg-slate-700' :
            confirming ? 'bg-red-600 hover:bg-red-500 text-white' : 'bg-slate-700 hover:bg-red-900/50 text-red-400',
          )}>
          <Trash2 className="w-3.5 h-3.5" /> {confirming ? 'Confirm Delete' : 'Delete'}
        </button>
        {toast && (
          <span className={clsx('text-xs self-center', toast.startsWith('Error') ? 'text-red-400' : 'text-green-400')}>
            {toast}
          </span>
        )}
      </div>

      {showClone && (
        <div className="flex gap-2">
          <input value={cloneInput} onChange={e => setCloneInput(e.target.value)} placeholder="new_config_name"
            className="px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-white mono w-48 focus:outline-none focus:border-blue-500" />
          <button onClick={doClone} className="px-3 py-1 text-xs bg-green-700 hover:bg-green-600 text-white rounded">Clone</button>
          <button onClick={() => { setShowClone(false); setCloneInput(''); }} className="px-2 py-1 text-xs text-slate-400 hover:text-slate-200">Cancel</button>
        </div>
      )}

      {sanity.result && <SanityCheckCard result={sanity.result} />}
      {sanity.error && <p className="text-xs text-red-400">Sanity check failed: {sanity.error}</p>}
    </div>
  );
}

// ── LLM Config Panel (full) ───────────────────────────────────────────────────

function LLMConfigsPanel() {
  const [response,  setResponse]  = useState<HybridConfigsResponse | null>(null);
  const [selected,  setSelected]  = useState<string | null>(null);
  const [showNew,   setShowNew]   = useState(false);
  const [newName,   setNewName]   = useState('');
  const [creating,  setCreating]  = useState(false);
  const [toast,     setToast]     = useState<string | null>(null);

  const load = () =>
    apiGet<HybridConfigsResponse>('/config/hybrid-configs').then(r => {
      setResponse(r);
      if (!selected && r.configs.length) setSelected(r.configs[0].name);
    }).catch(() => {});

  useEffect(() => { load(); }, []);

  const cfg         = response?.configs.find(c => c.name === selected) ?? null;
  const providers   = response?.providers   ?? [];
  const styles      = response?.enhance_styles ?? [];
  const active      = response?.active ?? null;

  const handleSave = async (updated: HybridConfigFull) => {
    await apiPut(`/config/hybrid-configs/${updated.name}`, updated);
    await load();
  };

  const handleDelete = async () => {
    if (!selected) return;
    await apiDelete(`/config/hybrid-configs/${selected}`);
    setSelected(null);
    await load();
  };

  const handleClone = async (newName: string) => {
    if (!selected) return;
    await apiPost(`/config/hybrid-configs/${selected}/clone`, { new_name: newName });
    await load();
    setSelected(newName);
    setToast(`Cloned to '${newName}'`);
    setTimeout(() => setToast(null), 2000);
  };

  const handleSanityCheck = async (): Promise<SanityCheckResult> => {
    if (!selected) throw new Error('No config selected');
    return apiPost<SanityCheckResult>(`/config/hybrid-configs/${selected}/sanity-check`);
  };

  const createNew = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await apiPost('/config/hybrid-configs', { name: newName, ...EMPTY_CFG });
      await load();
      setSelected(newName);
      setShowNew(false);
      setNewName('');
    } catch (e) {
      setToast(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700">
      <div className="flex items-center gap-2 px-5 py-3 border-b border-slate-700">
        <h3 className="text-sm font-semibold text-slate-200">LLM Configurations</h3>
        <InfoTooltip text={SETTINGS_HELP['llm_configs.section']} />
        <span className="text-xs text-slate-500">({response?.configs.length ?? 0} configs)</span>
      </div>

      <div className="p-5 flex gap-5">
        {/* Config list */}
        <div className="w-52 shrink-0 space-y-1">
          {response?.configs.map(c => (
            <button key={c.name} onClick={() => { setSelected(c.name); setShowNew(false); }}
              className={clsx('w-full text-left px-3 py-2 rounded text-xs transition-colors flex items-center justify-between gap-1',
                selected === c.name ? 'bg-slate-700 text-white' : 'text-slate-400 hover:bg-slate-700/60 hover:text-slate-200')}>
              <span className="truncate">{c.name}</span>
              {c.name === active && <span className="px-1 py-0.5 rounded text-xs bg-blue-700 text-blue-100 shrink-0">●</span>}
            </button>
          ))}
          <button onClick={() => { setShowNew(true); setSelected(null); }}
            className="w-full flex items-center gap-1.5 px-3 py-2 rounded text-xs text-blue-400 hover:bg-slate-700/60 hover:text-blue-300">
            <Plus className="w-3.5 h-3.5" /> New Config
          </button>
          {toast && <p className={clsx('text-xs px-2', toast.startsWith('Error') ? 'text-red-400' : 'text-green-400')}>{toast}</p>}
        </div>

        {/* Editor area */}
        <div className="flex-1 min-w-0">
          {showNew && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-300">New Configuration</p>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Name (alphanumeric + underscores)</label>
                <input value={newName} onChange={e => setNewName(e.target.value)}
                  placeholder="my_new_config"
                  className="w-full max-w-xs px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded text-white mono focus:outline-none focus:border-blue-500" />
              </div>
              <p className="text-xs text-slate-500">All 10 fields will default to sensible values. Edit after creation.</p>
              <button onClick={createNew} disabled={creating || !newName.trim()}
                className="px-4 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded font-medium">
                {creating ? 'Creating…' : 'Create Config'}
              </button>
            </div>
          )}
          {cfg && !showNew && (
            <LLMConfigEditor
              cfg={cfg}
              active={cfg.name === active}
              providers={providers}
              styles={styles}
              onSave={handleSave}
              onDelete={handleDelete}
              onClone={handleClone}
              onSanityCheck={handleSanityCheck}
            />
          )}
          {!cfg && !showNew && (
            <p className="text-xs text-slate-500">Select a config from the list to edit.</p>
          )}
        </div>
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
      <LLMConfigsPanel />
    </div>
  );
}
