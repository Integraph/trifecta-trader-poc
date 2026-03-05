import { useState } from 'react';
import { ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';

interface Props {
  data:          unknown;
  defaultOpen?:  boolean;
  label?:        string;
}

function highlight(json: string): React.ReactNode[] {
  const parts = json.split(/("(?:[^"\\]|\\.)*":\s*|"(?:[^"\\]|\\.)*"|[-\d.]+|true|false|null)/g);
  return parts.map((part, i) => {
    if (/^"[^"]*":\s*$/.test(part)) return <span key={i} className="text-slate-400">{part}</span>;
    if (/^"/.test(part))             return <span key={i} className="text-green-400">{part}</span>;
    if (/^[-\d.]/.test(part))        return <span key={i} className="text-blue-400">{part}</span>;
    if (part === 'true')             return <span key={i} className="text-yellow-400">{part}</span>;
    if (part === 'false')            return <span key={i} className="text-red-400">{part}</span>;
    if (part === 'null')             return <span key={i} className="text-slate-500">{part}</span>;
    return <span key={i}>{part}</span>;
  });
}

export function JsonViewer({ data, defaultOpen = false, label = 'Raw JSON' }: Props) {
  const [open, setOpen]     = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(data, null, 2);

  const copy = () => {
    navigator.clipboard.writeText(json).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="border border-slate-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 w-full px-4 py-2 bg-slate-800 text-sm text-slate-300 hover:bg-slate-700 transition-colors"
      >
        {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        <span className="font-medium">{label}</span>
        <span className="ml-auto text-slate-500 text-xs">{json.length} chars</span>
      </button>
      {open && (
        <div className="relative">
          <button
            onClick={copy}
            className="absolute top-2 right-2 p-1.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-400 hover:text-slate-200"
          >
            {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
          </button>
          <pre className="overflow-auto p-4 text-xs mono bg-slate-900 max-h-96 leading-relaxed">
            {highlight(json)}
          </pre>
        </div>
      )}
    </div>
  );
}
