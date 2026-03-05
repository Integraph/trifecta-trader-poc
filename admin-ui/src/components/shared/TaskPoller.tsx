import { Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { useTaskPoller } from '../../api/hooks';
import { JsonViewer } from './JsonViewer';

interface Props {
  taskId:       string | null;
  pollInterval?: number;
  onComplete?:  (result: unknown) => void;
  renderResult?: (result: unknown) => React.ReactNode;
  label?:       string;
}

export function TaskPoller({
  taskId,
  pollInterval = 2000,
  onComplete,
  renderResult,
  label = 'Task',
}: Props) {
  const { status, result, error } = useTaskPoller(taskId, pollInterval);

  if (!taskId) return null;

  if (status === 'running') {
    return (
      <div className="flex items-center gap-3 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
        <span className="text-slate-300 text-sm">{label} running…</span>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="flex items-start gap-3 p-4 bg-red-900/20 rounded-lg border border-red-800">
        <XCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-red-300">{label} failed</p>
          <p className="text-xs text-red-400 mt-1 mono">{error}</p>
        </div>
      </div>
    );
  }

  if (status === 'complete') {
    if (onComplete) onComplete(result);
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-green-400 text-sm">
          <CheckCircle2 className="w-4 h-4" />
          <span>{label} complete</span>
        </div>
        {renderResult ? renderResult(result) : <JsonViewer data={result} defaultOpen label="Result" />}
      </div>
    );
  }

  return null;
}
