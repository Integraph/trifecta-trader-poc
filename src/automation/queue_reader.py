"""
Queue Reader — polls a file-based queue directory for Scanner candidate JSON files
and runs deep analysis on each one.

Lifecycle per file:
    queue/pending/   → (atomic rename) → queue/processing/
                     → run_analysis pipeline
                     → queue/completed/  (on success)
                     → queue/pending/    (on failure, retry_count incremented)

The reader blocks the calling thread (use from the main thread in daemon mode, or
from a dedicated thread if you want it non-blocking).
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pytz

logger = logging.getLogger(__name__)

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def create_analyze_fn(hybrid_config: str, publish: bool = True) -> Callable:
    """Return an (ticker, scanner_context) -> dict analysis function.

    The scanner_context dict (opportunity_score, catalysts, signal_scores, key_data)
    is merged into the portfolio_context under the key 'scanner_context' so the
    pipeline can reference it without modifying run_analysis.py.

    Args:
        hybrid_config: Name of the hybrid LLM config to use.
        publish: Publish the resulting signal to Supabase after analysis.

    Returns:
        Callable with signature: fn(ticker: str, scanner_context: dict) -> dict
    """
    def _analyze_fn(ticker: str, scanner_context: dict) -> dict:
        from src.run_analysis import (
            run_analysis,
            build_portfolio_context,
            _publish_signal,
        )

        trade_date = datetime.now(pytz.utc).strftime("%Y-%m-%d")

        # Merge scanner context into portfolio_context (informational, not decisional)
        try:
            portfolio_context = build_portfolio_context(ticker)
        except Exception as e:
            logger.warning("Could not fetch portfolio context for %s: %s", ticker, e)
            portfolio_context = {"available": False, "error": str(e)}

        portfolio_context["scanner_context"] = scanner_context

        result = run_analysis(
            ticker=ticker,
            trade_date=trade_date,
            hybrid=hybrid_config,
            use_cache=True,
            cost_breakdown=False,
            portfolio_context=portfolio_context,
            batch_mode=True,
            debug=False,
        )

        if publish:
            try:
                from src.execution.trade_params import extract_trade_params_dual
                trade_params = extract_trade_params_dual(
                    ticker=ticker,
                    decision=result["decision"],
                    quality_score=result.get("quality_score", {}).get("composite", 0.0),
                    final_decision_text=result.get("final_trade_decision_text", ""),
                    trader_plan_text=result.get("trader_investment_plan", ""),
                    current_price=scanner_context.get("key_data", {}).get("current_price"),
                )
            except Exception as e:
                logger.warning("Trade param extraction failed for %s: %s", ticker, e)
                trade_params = None
            _publish_signal(result, trade_params)

        return result

    return _analyze_fn


class QueueReader:
    """Reads Scanner candidates from a file-based queue and runs analysis on each.

    The reader polls queue/pending/ for JSON files, processes them sequentially,
    and writes results to queue/completed/.  Failed analyses are moved back to
    queue/pending/ with an incremented retry_count.
    """

    def __init__(
        self,
        queue_dir: str,
        analyze_fn: Callable,
        target_trader: str = "trifecta-trader",
        poll_interval: int = 30,
        max_retries: int = 2,
        cooldown_seconds: int = 60,
        event_callback: Optional[Callable] = None,
    ):
        """
        Args:
            queue_dir: Path to the queue root directory (must contain
                       pending/, processing/, completed/ subdirs).
            analyze_fn: fn(ticker, scanner_context) -> dict
            target_trader: Only process files whose target_trader matches.
            poll_interval: Seconds between polls of pending/.
            max_retries: Files with retry_count >= max_retries are skipped.
            cooldown_seconds: Minimum gap between analyses (LLM rate limit guard).
            event_callback: Optional fn(event_type: str, data: dict) called at
                lifecycle points. Pass the admin event bus when the API is running.
        """
        self._queue_dir   = Path(queue_dir)
        self._pending     = self._queue_dir / "pending"
        self._processing  = self._queue_dir / "processing"
        self._completed   = self._queue_dir / "completed"
        self._analyze_fn  = analyze_fn
        self._target      = target_trader
        self._poll_interval  = poll_interval
        self._max_retries    = max_retries
        self._cooldown       = cooldown_seconds
        self._event_callback = event_callback
        self._running         = False
        self._stop_requested  = False
        self._last_poll: Optional[datetime]     = None
        self._last_analysis: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the polling loop. Blocks the calling thread until stop() is called."""
        self._ensure_queue_dirs()
        self._running = True
        self._stop_requested = False
        logger.info(
            "Queue reader started: dir=%s  target=%s  poll=%ds  cooldown=%ds",
            self._queue_dir, self._target, self._poll_interval, self._cooldown,
        )
        while not self._stop_requested:
            self._last_poll = datetime.now(timezone.utc)
            try:
                self.poll_once()
            except Exception as e:
                logger.error("Unexpected error in poll cycle: %s", e)
            if not self._stop_requested:
                time.sleep(self._poll_interval)
        self._running = False

    def stop(self) -> None:
        """Signal the polling loop to stop after the current cycle completes."""
        logger.info("Queue reader stop requested.")
        self._stop_requested = True
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the reader is actively polling."""
        return self._running

    @property
    def pending_count(self) -> int:
        """Number of JSON files currently in pending/."""
        try:
            return len(list(self._pending.glob("*.json")))
        except Exception:
            return 0

    @property
    def last_poll(self) -> Optional[datetime]:
        return self._last_poll

    def poll_once(self) -> list:
        """Single poll cycle: find, sort, and process all pending candidates.

        Returns:
            List of result dicts for successfully processed candidates.
        """
        candidates = sorted(
            self._pending.glob("*.json"),
            key=self._sort_key,
        )

        if not candidates:
            logger.debug("Queue empty.")
            return []

        logger.info("Poll found %d candidate(s).", len(candidates))
        results = []
        for path in candidates:
            result = self._process_candidate(path)
            if result is not None:
                results.append(result)
                # Cooldown between analyses (daemon mode only — skipped in direct poll_once calls)
                if self._cooldown > 0 and not self._stop_requested:
                    logger.debug("Cooldown: sleeping %ds", self._cooldown)
                    time.sleep(self._cooldown)
            # Honour stop signal between candidates (daemon mode stop)
            if self._stop_requested:
                break
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _emit(self, event_type: str, data: dict) -> None:
        """Publish a lifecycle event to the admin event bus (if connected)."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception:
                pass

    def _process_candidate(self, pending_path: Path) -> Optional[dict]:
        """Move, analyze, and file a single candidate.

        Returns the analysis result dict on success, None on failure.
        """
        # Read the candidate file
        try:
            message = json.loads(pending_path.read_text())
        except Exception as e:
            logger.error("Failed to read %s: %s", pending_path.name, e)
            return None

        if self._should_skip(message):
            logger.debug("Skipping %s (filtered).", pending_path.name)
            return None

        ticker = message.get("ticker", "UNKNOWN").upper()
        logger.info("Processing candidate: %s  priority=%s", ticker, message.get("priority", "?"))
        self._emit("queue.candidate_picked", {
            "ticker":   ticker,
            "priority": message.get("priority", "medium"),
            "filename": pending_path.name,
        })

        # Move to processing/
        processing_path = self._processing / pending_path.name
        try:
            shutil.move(str(pending_path), str(processing_path))
        except Exception as e:
            logger.error("Could not move %s to processing: %s", pending_path.name, e)
            return None

        # Run analysis
        scanner_context = {
            "scanner_id":       message.get("scanner_id"),
            "opportunity_score": message.get("opportunity_score"),
            "catalysts":        message.get("catalysts", []),
            "signal_scores":    message.get("signal_scores", {}),
            "key_data":         message.get("key_data", {}),
            "priority":         message.get("priority", "medium"),
        }

        try:
            result = self._analyze_fn(ticker, scanner_context)
            self._last_analysis = datetime.now(timezone.utc)
            self._write_completed(processing_path, message, result)
            self._emit("queue.analysis_completed", {
                "ticker":    ticker,
                "decision":  result.get("decision"),
                "quality":   result.get("quality_score", {}).get("composite") if isinstance(result.get("quality_score"), dict) else None,
            })
            return result

        except Exception as e:
            logger.error("Analysis failed for %s: %s", ticker, e)
            self._handle_failure(processing_path, message, str(e))
            self._emit("queue.analysis_failed", {"ticker": ticker, "error": str(e)})
            return None

    def _should_skip(self, message: dict) -> bool:
        """Return True if this candidate should be skipped."""
        if message.get("target_trader") != self._target:
            return True
        if message.get("retry_count", 0) >= self._max_retries:
            logger.warning(
                "Skipping %s: reached max_retries=%d",
                message.get("ticker", "?"), self._max_retries,
            )
            return True
        if message.get("asset_type") == "crypto":
            return True
        return False

    def _write_completed(
        self, processing_path: Path, original: dict, result: dict
    ) -> None:
        """Write the enriched completion JSON to queue/completed/."""
        qs = result.get("quality_score", {})
        tp = result.get("trade_params", {}) or {}

        # Check if a signal was published and get its id
        signal_id = result.get("published_signal_id")  # may be None

        completed = {
            **original,
            "status":    "completed",
            "analysis_result": {
                "decision":       result.get("decision"),
                "quality_score":  qs.get("composite"),
                "entry_price":    tp.get("entry_price"),
                "stop_loss":      tp.get("stop_loss"),
                "price_target":   tp.get("price_target"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "cost_usd":       result.get("cost_breakdown", {}).get("total_usd"),
                "signal_published": signal_id is not None,
                "signal_id":      signal_id,
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        completed_path = self._completed / processing_path.name
        try:
            completed_path.write_text(json.dumps(completed, indent=2, default=str))
            processing_path.unlink(missing_ok=True)
            logger.info(
                "Completed: %s  decision=%s  quality=%.1f",
                processing_path.name,
                result.get("decision", "?"),
                qs.get("composite", 0),
            )
        except Exception as e:
            logger.error("Failed to write completed file for %s: %s", processing_path.name, e)

    def _handle_failure(
        self, processing_path: Path, message: dict, error: str
    ) -> None:
        """Move file back to pending with incremented retry_count."""
        message["retry_count"] = message.get("retry_count", 0) + 1
        message["last_error"]  = error
        message["last_retry_at"] = datetime.now(timezone.utc).isoformat()

        retry_path = self._pending / processing_path.name
        try:
            retry_path.write_text(json.dumps(message, indent=2, default=str))
            processing_path.unlink(missing_ok=True)
            logger.warning(
                "Moved %s back to pending (retry %d/%d): %s",
                processing_path.name, message["retry_count"], self._max_retries, error,
            )
        except Exception as e:
            logger.error("Failed to move %s back to pending: %s", processing_path.name, e)

    def _sort_key(self, path: Path) -> tuple:
        """Sort by priority (high first), then by filename (oldest timestamp first)."""
        try:
            message = json.loads(path.read_text())
            priority_val = PRIORITY_ORDER.get(message.get("priority", "medium"), 1)
        except Exception:
            priority_val = 1
        return (priority_val, path.name)

    def _ensure_queue_dirs(self) -> None:
        """Create queue subdirectories if they don't exist."""
        for sub in (self._pending, self._processing, self._completed):
            sub.mkdir(parents=True, exist_ok=True)
