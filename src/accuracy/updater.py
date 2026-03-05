"""
Daily Accuracy Updater — fetches actual prices and scores outcomes.

Designed to run as a scheduled job in the daemon (Task 014), or
standalone via CLI.

Usage:
    python -m src.accuracy.updater                   # One update cycle
    python -m src.accuracy.updater --backfill 30     # Backfill last 30 days
    python -m src.accuracy.updater --ticker AAPL     # Update one ticker
    python -m src.accuracy.updater --report          # Print accuracy summary
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from src.accuracy.price_tracker import PriceTracker, fetch_outcome_prices
from src.accuracy.scorer import AccuracyScorer

logger = logging.getLogger(__name__)


class AccuracyUpdater:
    """Orchestrates daily price fetching and accuracy scoring."""

    def __init__(self, db=None):
        """
        Args:
            db: PortfolioDatabase instance. If None, a default instance is created.
        """
        if db is None:
            from src.portfolio.database import PortfolioDatabase
            db = PortfolioDatabase()
        self.tracker = PriceTracker(db)
        self.scorer  = AccuracyScorer()

    def run_update(self, ticker: Optional[str] = None) -> dict:
        """Full update cycle: fetch prices → score complete outcomes.

        Args:
            ticker: If provided, update only outcomes for this ticker.

        Returns:
            Summary dict:
            {
                "total_pending": int,
                "updated": int,
                "newly_complete": int,
                "errors": int,
                "skipped": int,
            }
        """
        pending = self.tracker.get_pending_outcomes()
        if ticker:
            pending = [o for o in pending if o["ticker"].upper() == ticker.upper()]

        total     = len(pending)
        updated   = 0
        complete  = 0
        errors    = 0
        skipped   = 0

        logger.info("Accuracy update starting: %d pending outcomes", total)

        for outcome in pending:
            outcome_id  = outcome["id"]
            tick        = outcome["ticker"]
            signal_date = outcome["signal_date"]

            try:
                prices = fetch_outcome_prices(tick, signal_date)
            except Exception as e:
                logger.error("Price fetch failed for %s %s: %s", tick, signal_date, e)
                self.tracker.mark_error(outcome_id, str(e))
                errors += 1
                continue

            if not prices:
                logger.warning("No price data for %s %s — skipping", tick, signal_date)
                skipped += 1
                continue

            # Update stored price checkpoints
            self.tracker.update_prices(outcome_id, prices)
            updated += 1

            # If T+10 is now available, score the outcome and mark complete
            if "price_t10" in prices:
                merged = {**outcome, **prices}
                daily_highs = prices.get("daily_highs")
                daily_lows  = prices.get("daily_lows")

                scores = self.scorer.score_outcome(merged, daily_highs, daily_lows)
                if scores:
                    self.tracker.apply_scores(outcome_id, scores)
                self.tracker.mark_complete(outcome_id)
                complete += 1
                logger.info(
                    "Scored %s %s: dir_t5=%s  ret_t5=%.2f%%  target_hit=%s",
                    tick, signal_date,
                    scores.get("direction_correct_t5"),
                    scores.get("return_t5_pct", 0.0) or 0.0,
                    scores.get("target_hit"),
                )

        summary = {
            "total_pending":   total,
            "updated":         updated,
            "newly_complete":  complete,
            "errors":          errors,
            "skipped":         skipped,
        }
        logger.info("Accuracy update complete: %s", summary)
        return summary

    def backfill(self, days_back: int = 30) -> dict:
        """Create and populate outcome records for historical analyses.

        Scans the analyses table for entries within the last N days that
        don't yet have an outcome record, creates them, then immediately
        fetches prices.

        Args:
            days_back: How many calendar days of history to backfill.

        Returns:
            Summary dict with created/updated/errored counts.
        """
        from src.portfolio.database import PortfolioDatabase

        db: PortfolioDatabase = self.tracker._db
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()

        sql = """
            SELECT a.id, a.ticker, a.trade_date, a.decision,
                   a.entry_price, a.stop_loss, a.price_target
              FROM analyses a
         LEFT JOIN signal_outcomes so ON so.analysis_id = a.id
             WHERE a.trade_date >= ?
               AND so.id IS NULL
             ORDER BY a.trade_date ASC
        """
        with db._conn() as conn:
            rows = [dict(r) for r in conn.execute(sql, (cutoff,)).fetchall()]

        created = 0
        for row in rows:
            try:
                self.tracker.create_outcome(
                    analysis_id=row["id"],
                    ticker=row["ticker"],
                    signal_date=row["trade_date"],
                    decision=row["decision"],
                    entry_price=row.get("entry_price"),
                    stop_loss=row.get("stop_loss"),
                    price_target=row.get("price_target"),
                )
                created += 1
            except Exception as e:
                logger.warning("Backfill create_outcome failed for analysis %d: %s", row["id"], e)

        logger.info("Backfill: created %d new outcome records (last %d days)", created, days_back)

        # Now run a normal update cycle to populate prices
        update_summary = self.run_update()
        return {"created": created, **update_summary}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accuracy updater — fetch prices and score pipeline signals."
    )
    parser.add_argument("--backfill", metavar="DAYS", type=int,
                        help="Backfill outcomes for the last N days of analyses")
    parser.add_argument("--ticker", metavar="TICKER",
                        help="Limit update to a single ticker")
    parser.add_argument("--report", action="store_true",
                        help="Print accuracy summary and exit")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history for --report (default: 30)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    updater = AccuracyUpdater()

    if args.report:
        from src.accuracy.reporter import AccuracyReporter
        reporter = AccuracyReporter(updater.tracker._db)
        reporter.print_summary(days=args.days)
        return

    if args.backfill:
        result = updater.backfill(days_back=args.backfill)
        print(f"Backfill complete: {result}")
        return

    result = updater.run_update(ticker=args.ticker)
    print(f"Update complete: {result}")


if __name__ == "__main__":
    main()
