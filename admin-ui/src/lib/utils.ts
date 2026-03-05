// ── Date / time helpers ───────────────────────────────────────────────────────

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('en-US', {
      month: 'short',
      day:   'numeric',
      hour:  '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  } catch {
    return iso;
  }
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleTimeString('en-US', {
      hour:   '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return iso;
  }
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('en-US', {
      month: 'short',
      day:   'numeric',
      year:  'numeric',
    });
  } catch {
    return iso;
  }
}

export function formatUptime(seconds: number | null | undefined): string {
  if (seconds == null) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatCountdown(isoFuture: string | null | undefined): string {
  if (!isoFuture) return '—';
  const diff = new Date(isoFuture).getTime() - Date.now();
  if (diff <= 0) return 'now';
  const h = Math.floor(diff / 3_600_000);
  const m = Math.floor((diff % 3_600_000) / 60_000);
  const s = Math.floor((diff % 60_000) / 1_000);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatElapsed(seconds: number | null | undefined): string {
  if (seconds == null) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${(seconds / 60).toFixed(1)}m`;
}

// ── Number formatters ─────────────────────────────────────────────────────────

export function formatCurrency(value: number | null | undefined, decimals = 3): string {
  if (value == null) return '—';
  return `$${value.toFixed(decimals)}`;
}

export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (value == null) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

export function formatScore(value: number | null | undefined): string {
  if (value == null) return '—';
  return value.toFixed(1);
}

export function formatPrice(value: number | null | undefined): string {
  if (value == null) return '—';
  return `$${value.toFixed(2)}`;
}

// ── Decision colors ───────────────────────────────────────────────────────────

export function decisionColor(decision: string | null | undefined): string {
  if (!decision) return 'text-slate-400';
  switch (decision.toUpperCase()) {
    case 'BUY':  return 'text-green-400';
    case 'SELL': return 'text-red-400';
    default:     return 'text-slate-400';
  }
}

export function decisionBg(decision: string | null | undefined): string {
  if (!decision) return 'bg-slate-600 text-slate-200';
  switch (decision.toUpperCase()) {
    case 'BUY':  return 'bg-green-700 text-green-100';
    case 'SELL': return 'bg-red-700 text-red-100';
    default:     return 'bg-slate-600 text-slate-200';
  }
}

export function priorityColor(priority: string | null | undefined): string {
  switch ((priority ?? '').toLowerCase()) {
    case 'high':   return 'text-red-400';
    case 'medium': return 'text-yellow-400';
    default:       return 'text-slate-400';
  }
}

// ── Misc ──────────────────────────────────────────────────────────────────────

export function clsx(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ');
}

export function today(): string {
  return new Date().toISOString().slice(0, 10);
}
