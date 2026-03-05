import { Inbox } from 'lucide-react';

interface Props {
  message?: string;
  action?:  React.ReactNode;
}

export function EmptyState({ message = 'No data yet', action }: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-slate-500 gap-3">
      <Inbox className="w-10 h-10 opacity-40" />
      <p className="text-sm">{message}</p>
      {action}
    </div>
  );
}
