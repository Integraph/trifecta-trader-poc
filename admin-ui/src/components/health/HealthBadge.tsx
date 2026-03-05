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
} as const;

export function HealthBadge({ health }: Props) {
  const color = health ? COLOR_MAP[health.color] : 'gray';
  const label = health ? health.status : 'connecting…';

  return (
    <div className="flex items-center gap-2 text-sm">
      <StatusDot
        color={color}
        pulse={color === 'green'}
        size="sm"
      />
      <span className="text-slate-300 capitalize">{label}</span>
      {health?.uptime_seconds != null && (
        <span className="text-slate-500 text-xs">
          up {formatUptime(health.uptime_seconds)}
        </span>
      )}
      {health?.pid && (
        <span className="text-slate-600 text-xs">pid:{health.pid}</span>
      )}
    </div>
  );
}
