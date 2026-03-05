import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, CalendarClock, Inbox, Target,
  FlaskConical, ClipboardList, Settings, ScrollText,
} from 'lucide-react';
import { StatusDot } from '../shared/StatusDot';
import { clsx } from '../../lib/utils';
import type { HealthResponse } from '../../api/types';

interface Props {
  health: HealthResponse | null;
}

interface NavItem {
  to:     string;
  label:  string;
  Icon:   React.ElementType;
  badge?: (h: HealthResponse) => React.ReactNode;
}

const NAV: NavItem[] = [
  { to: '/',          label: 'Dashboard',     Icon: LayoutDashboard },
  {
    to: '/scheduler', label: 'Scheduler',      Icon: CalendarClock,
    badge: (h) => {
      const ok = h.subsystems.scheduler.status === 'running';
      return <StatusDot color={ok ? 'green' : 'red'} size="sm" />;
    },
  },
  {
    to: '/queue',     label: 'Queue',          Icon: Inbox,
    badge: (h) => {
      const n = h.subsystems.queue_reader.pending_count;
      if (!n) return null;
      return (
        <span className="ml-auto text-xs bg-yellow-500 text-black px-1.5 py-0.5 rounded-full font-medium">
          {n}
        </span>
      );
    },
  },
  {
    to: '/accuracy',  label: 'Accuracy',       Icon: Target,
    badge: (h) => {
      const n = h.subsystems.accuracy_updater.pending_outcomes;
      if (!n) return null;
      return (
        <span className="ml-auto text-xs bg-slate-600 text-slate-200 px-1.5 py-0.5 rounded-full">
          {n}
        </span>
      );
    },
  },
  { to: '/test-run',  label: 'Test Run',       Icon: FlaskConical },
  { to: '/analyses',  label: 'Analyses',       Icon: ClipboardList },
  { to: '/config',    label: 'Configuration',  Icon: Settings },
  { to: '/logs',      label: 'Logs',           Icon: ScrollText },
];

export function Sidebar({ health }: Props) {
  return (
    <aside className="w-60 shrink-0 bg-slate-900 border-r border-slate-700/60 flex flex-col h-full">
      <div className="px-5 py-4 border-b border-slate-700/60">
        <h1 className="text-base font-semibold text-white tracking-tight">
          Trifecta Trader
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">Admin Dashboard</p>
      </div>

      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
        {NAV.map(({ to, label, Icon, badge }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
                isActive
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50',
              )
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            <span className="flex-1">{label}</span>
            {health && badge?.(health)}
          </NavLink>
        ))}
      </nav>

      <div className="px-4 py-3 border-t border-slate-700/60">
        <p className="text-xs text-slate-600">API: localhost:8420</p>
      </div>
    </aside>
  );
}
