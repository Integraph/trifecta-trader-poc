import { useEffect, useRef, useState } from 'react';
import { Info } from 'lucide-react';

interface Props {
  text: string;
}

export function InfoTooltip({ text }: Props) {
  const [open, setOpen]     = useState(false);
  const iconRef             = useRef<HTMLButtonElement>(null);
  const popoverRef          = useRef<HTMLDivElement>(null);
  const [flipLeft, setFlip] = useState(false);

  // Close on click-outside or Escape
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    const onClickOut = (e: MouseEvent) => {
      if (
        !iconRef.current?.contains(e.target as Node) &&
        !popoverRef.current?.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onClickOut);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onClickOut);
    };
  }, [open]);

  // Flip popover to left if it would overflow viewport
  useEffect(() => {
    if (!open || !popoverRef.current) return;
    const rect = popoverRef.current.getBoundingClientRect();
    setFlip(rect.right > window.innerWidth - 16);
  }, [open]);

  return (
    <span className="relative inline-flex items-center">
      <button
        ref={iconRef}
        type="button"
        onClick={() => setOpen(o => !o)}
        className="text-slate-500 hover:text-slate-300 transition-colors focus:outline-none"
        aria-label="Show help"
        aria-expanded={open}
      >
        <Info className="w-3.5 h-3.5" />
      </button>

      {open && (
        <div
          ref={popoverRef}
          role="tooltip"
          className={`absolute z-50 top-5 ${flipLeft ? 'right-0' : 'left-0'} w-72 bg-slate-700 border border-slate-600 rounded-lg shadow-xl p-3`}
        >
          <p className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">{text}</p>
        </div>
      )}
    </span>
  );
}
