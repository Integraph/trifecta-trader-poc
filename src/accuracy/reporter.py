"""
Accuracy Reporter — generates human-readable and machine-readable summaries
of pipeline signal accuracy from the signal_outcomes table.

Key insight: if high-quality signals (score 8-10) are significantly more
accurate than low-quality ones (0-6), the quality scoring is working.
If there's no correlation, the scoring formula needs recalibration.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

QUALITY_TIERS = [
    ("high (8-10)",   8.0, 10.1),
    ("medium (6-8)",  6.0,  8.0),
    ("low (0-6)",     0.0,  6.0),
]


class AccuracyReporter:
    """Generates accuracy reports from the signal_outcomes and analyses tables."""

    def __init__(self, db):
        """
        Args:
            db: PortfolioDatabase instance.
        """
        self._db = db

    def summary(self, days: int = 30) -> dict:
        """Aggregate accuracy metrics over the last N days.

        Only includes outcomes with status='complete' (all T+10 data scored).

        Returns:
            Dict with total_signals, by_decision, by_quality_tier,
            best_signals, worst_signals.
        """
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=days)).isoformat()

        sql = """
            SELECT so.*, a.quality_score
              FROM signal_outcomes so
              JOIN analyses a ON a.id = so.analysis_id
             WHERE so.status = 'complete'
               AND so.signal_date >= ?
             ORDER BY so.signal_date DESC
        """
        with self._db._conn() as conn:
            rows = [dict(r) for r in conn.execute(sql, (cutoff,)).fetchall()]

        total = len(rows)
        result = {
            "period_days":   days,
            "total_signals": total,
            "by_decision":   {},
            "by_quality_tier": {},
            "best_signals":  [],
            "worst_signals": [],
        }

        if total == 0:
            return result

        # ── By decision ───────────────────────────────────────────────────────
        for decision in ("BUY", "SELL", "HOLD"):
            group = [r for r in rows if r["decision"].upper() == decision]
            if not group:
                continue
            result["by_decision"][decision] = _aggregate(group)

        # ── By quality tier ───────────────────────────────────────────────────
        for label, lo, hi in QUALITY_TIERS:
            group = [r for r in rows if r.get("quality_score") is not None
                     and lo <= r["quality_score"] < hi]
            if not group:
                continue
            result["by_quality_tier"][label] = _aggregate(group)

        # ── Best / worst signals ──────────────────────────────────────────────
        scored = [r for r in rows if r.get("return_t10_pct") is not None]
        scored_sorted = sorted(scored, key=lambda r: r["return_t10_pct"] or 0, reverse=True)
        result["best_signals"]  = _format_top(scored_sorted[:5])
        result["worst_signals"] = _format_top(scored_sorted[-5:][::-1])

        return result

    def ticker_report(self, ticker: str) -> dict:
        """Accuracy report for a specific ticker across all signals.

        Returns signal history with outcomes for trend analysis.
        """
        sql = """
            SELECT so.*, a.quality_score
              FROM signal_outcomes so
              JOIN analyses a ON a.id = so.analysis_id
             WHERE so.ticker = ?
             ORDER BY so.signal_date DESC
        """
        with self._db._conn() as conn:
            rows = [dict(r) for r in conn.execute(sql, (ticker.upper(),)).fetchall()]

        complete = [r for r in rows if r["status"] == "complete"]
        return {
            "ticker":         ticker.upper(),
            "total_signals":  len(rows),
            "complete":       len(complete),
            "pending":        len(rows) - len(complete),
            "summary":        _aggregate(complete) if complete else {},
            "signals":        rows,
        }

    def print_summary(self, days: int = 30) -> None:
        """Print a formatted accuracy summary to stdout."""
        data   = self.summary(days=days)
        total  = data["total_signals"]
        by_dec = data["by_decision"]
        by_qt  = data["by_quality_tier"]
        best   = data["best_signals"]
        worst  = data["worst_signals"]

        SEP = "═" * 55
        print(f"\n{SEP}")
        print(f"TRIFECTA TRADER — Signal Accuracy Report ({days} days)")
        print(SEP)

        # Count pending outcomes separately
        with self._db._conn() as conn:
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM signal_outcomes WHERE status IN ('pending','partial')"
            ).fetchone()[0]
        complete_count = total

        print(f"Total signals: {total + pending_count} | "
              f"Complete: {complete_count} | Pending: {pending_count}")

        if total == 0:
            print("\nNo complete signals yet.")
            print(SEP)
            return

        # Direction accuracy
        print("\nDirection Accuracy:")
        for dec in ("BUY", "SELL", "HOLD"):
            if dec not in by_dec:
                continue
            g = by_dec[dec]
            n = g["count"]
            if dec == "HOLD":
                print(f"  {dec:<4} ({n:>2} signals): —")
            else:
                t1  = _pct(g.get("direction_correct_t1"))
                t5  = _pct(g.get("direction_correct_t5"))
                t10 = _pct(g.get("direction_correct_t10"))
                print(f"  {dec:<4} ({n:>2} signals): T+1: {t1} | T+5: {t5} | T+10: {t10}")

        # Target/stop performance
        print("\nTarget/Stop Performance:")
        for dec in ("BUY", "SELL"):
            if dec not in by_dec:
                continue
            g = by_dec[dec]
            tgt  = _pct(g.get("target_hit_rate"))
            stp  = _pct(g.get("stop_hit_rate"))
            tfs  = _pct(g.get("target_before_stop_rate"))
            print(f"  {dec}: Target hit: {tgt} | Stop hit: {stp} | Target first: {tfs}")

        # Return by quality tier
        print("\nReturn by Quality Tier (T+5 avg):")
        for label, _, _ in QUALITY_TIERS:
            if label not in by_qt:
                continue
            g = by_qt[label]
            n   = g["count"]
            r5  = g.get("avg_return_t5_pct")
            r5s = f"{r5:+.1f}%" if r5 is not None else "N/A"
            print(f"  {label:<16}: {r5s:>7}  ({n} signals)")

        # Best / worst
        if best:
            best_str = " | ".join(
                f"{s['ticker']} {s['return_t10_pct']:+.1f}%" for s in best
            )
            print(f"\nTop {len(best)} Best:  {best_str}")
        if worst:
            worst_str = " | ".join(
                f"{s['ticker']} {s['return_t10_pct']:+.1f}%" for s in worst
            )
            print(f"Top {len(worst)} Worst: {worst_str}")

        print(SEP + "\n")


# ── Private helpers ───────────────────────────────────────────────────────────

def _avg(values) -> Optional[float]:
    """Average of a list of numbers, skipping None values."""
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def _rate(rows, field: str) -> Optional[float]:
    """Fraction of rows where field == 1 (ignoring None)."""
    vals = [r.get(field) for r in rows if r.get(field) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _aggregate(rows: list) -> dict:
    """Compute aggregate accuracy metrics for a group of outcome rows."""
    n = len(rows)
    return {
        "count":                    n,
        "direction_correct_t1":     _rate(rows, "direction_correct_t1"),
        "direction_correct_t5":     _rate(rows, "direction_correct_t5"),
        "direction_correct_t10":    _rate(rows, "direction_correct_t10"),
        "target_hit_rate":          _rate(rows, "target_hit"),
        "stop_hit_rate":            _rate(rows, "stop_hit"),
        "target_before_stop_rate":  _rate(rows, "target_hit_first"),
        "avg_return_t1_pct":        _avg([r.get("return_t1_pct") for r in rows]),
        "avg_return_t5_pct":        _avg([r.get("return_t5_pct") for r in rows]),
        "avg_return_t10_pct":       _avg([r.get("return_t10_pct") for r in rows]),
        "avg_max_favorable_pct":    _avg([r.get("max_favorable_pct") for r in rows]),
        "avg_max_adverse_pct":      _avg([r.get("max_adverse_pct") for r in rows]),
    }


def _format_top(rows: list) -> list:
    """Format top/worst signals for the summary dict."""
    result = []
    for r in rows:
        result.append({
            "ticker":           r.get("ticker"),
            "decision":         r.get("decision"),
            "return_t10_pct":   r.get("return_t10_pct"),
            "quality_score":    r.get("quality_score"),
            "signal_date":      r.get("signal_date"),
        })
    return result


def _pct(value: Optional[float]) -> str:
    """Format a fraction (0-1) as a percentage string, or '—' if None."""
    if value is None:
        return "  —  "
    return f"{value * 100:.0f}%"
