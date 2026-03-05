import { useLocation } from 'react-router-dom';
import { HealthBadge } from '../health/HealthBadge';
import type { HealthResponse } from '../../api/types';

interface Props {
  health: HealthResponse | null;
}

const PAGE_TITLES: Record<string, string> = {
  '/':           'Dashboard',
  '/scheduler':  'Scheduler',
  '/queue':      'Queue',
  '/accuracy':   'Accuracy',
  '/test-run':   'Test Run',
  '/analyses':   'Analyses',
  '/config':     'Configuration',
  '/logs':       'Logs',
};

export function Header({ health }: Props) {
  const { pathname } = useLocation();
  const base  = '/' + pathname.split('/')[1];
  const title = PAGE_TITLES[base] ?? 'Admin';

  return (
    <header className="h-12 shrink-0 px-6 flex items-center justify-between border-b border-slate-700/60 bg-slate-900/80 backdrop-blur-sm">
      <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
      <HealthBadge health={health} />
    </header>
  );
}
