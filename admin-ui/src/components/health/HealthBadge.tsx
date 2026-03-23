import { StatusDot } from '../shared/StatusDot';
import { formatUptime } from '../../lib/utils';
import type { HealthResponse } from '../../api/types';

interface Props {
  health: HealthResponse | null;
}

const COLOR_MAP = {
  green:  'green',
  yellow: 'yellow',
  red:    'red',
  blue:   'blue',
} as const;

const STATUS_LABEL: Record<string, string> = {
  healthy:    'Healthy',
  degraded:   'Degraded',
  unhealthy:  'Unhealthy',
  standalone: 'Standalone',
};

export function HealthBadge({ health }: Props) {
  const color = health ? COLOR_MAP[health.color] ?? 'gray' : 'gray';
  const rawStatus = health ? health.status : 'connecting…';
  const label = STATUS_LABEL[rawStatus] ?? rawStatus;

  return (
    <div className="flex items-center gap-2 text-sm">
      <StatusDot
        color={color as 'green' | 'yellow' | 'red' | 'gray' | 'blue'}
        pulse={color === 'green'}
        size="sm"
      />
      <span className="text-slate-300 capitalize">{label}</span>
      {health?.uptime_seconds != null && (
        <span className="text-slate-500 text-xs">
          up {formatUptime(health.uptime_seconds)}
        </span>
      )}
      {health?.mode === 'standalone' && (
        <span className="text-blue-600 text-xs">(dev)</span>
      )}
    </div>
  );
}
