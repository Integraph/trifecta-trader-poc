import { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { EmptyState } from './EmptyState';
import { clsx } from '../../lib/utils';

export interface Column<T> {
  key:        string;
  label:      string;
  sortable?:  boolean;
  className?: string;
  render?:    (row: T) => React.ReactNode;
}

interface Props<T> {
  columns:    Column<T>[];
  rows:       T[];
  keyFn:      (row: T) => string | number;
  onRowClick?: (row: T) => void;
  loading?:   boolean;
  emptyMsg?:  string;
}

export function DataTable<T extends Record<string, unknown>>({
  columns,
  rows,
  keyFn,
  onRowClick,
  loading  = false,
  emptyMsg = 'No data',
}: Props<T>) {
  const [sortKey, setSortKey]   = useState<string | null>(null);
  const [sortDir, setSortDir]   = useState<'asc' | 'desc'>('asc');

  const toggleSort = (key: string) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };

  const sorted = [...rows].sort((a, b) => {
    if (!sortKey) return 0;
    const av = a[sortKey]; const bv = b[sortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = String(av).localeCompare(String(bv), undefined, { numeric: true });
    return sortDir === 'asc' ? cmp : -cmp;
  });

  if (loading) {
    return (
      <div className="animate-pulse space-y-2 p-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-8 bg-slate-700 rounded" />
        ))}
      </div>
    );
  }

  if (!rows.length) return <EmptyState message={emptyMsg} />;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700">
            {columns.map(col => (
              <th
                key={col.key}
                className={clsx(
                  'px-4 py-2 text-left text-xs font-medium text-slate-400 uppercase tracking-wider',
                  col.sortable && 'cursor-pointer select-none hover:text-slate-200',
                  col.className,
                )}
                onClick={() => col.sortable && toggleSort(col.key)}
              >
                <span className="flex items-center gap-1">
                  {col.label}
                  {col.sortable && sortKey === col.key && (
                    sortDir === 'asc'
                      ? <ChevronUp className="w-3 h-3" />
                      : <ChevronDown className="w-3 h-3" />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(row => (
            <tr
              key={keyFn(row)}
              onClick={() => onRowClick?.(row)}
              className={clsx(
                'border-b border-slate-800 hover:bg-slate-800/50 transition-colors',
                onRowClick && 'cursor-pointer',
              )}
            >
              {columns.map(col => (
                <td key={col.key} className={clsx('px-4 py-2 text-slate-300', col.className)}>
                  {col.render ? col.render(row) : String(row[col.key] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
