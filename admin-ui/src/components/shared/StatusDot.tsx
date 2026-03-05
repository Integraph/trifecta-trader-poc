import { clsx } from '../../lib/utils';

interface Props {
  color: 'green' | 'yellow' | 'red' | 'gray';
  pulse?: boolean;
  size?:  'sm' | 'md' | 'lg';
}

const SIZE = { sm: 'w-2 h-2', md: 'w-3 h-3', lg: 'w-4 h-4' };
const COLOR = {
  green:  'bg-green-500',
  yellow: 'bg-yellow-400',
  red:    'bg-red-500',
  gray:   'bg-slate-500',
};

export function StatusDot({ color, pulse = false, size = 'md' }: Props) {
  return (
    <span className={clsx('relative inline-flex rounded-full', SIZE[size], COLOR[color])}>
      {pulse && (
        <span
          className={clsx(
            'absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping',
            COLOR[color],
          )}
        />
      )}
    </span>
  );
}
