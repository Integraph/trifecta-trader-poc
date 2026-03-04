"""
SQLite database layer for portfolio state tracking.

Database location: data/portfolio.db (relative to repo root).
Tables: analyses, orders, portfolio_snapshots.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT    NOT NULL,
    trade_date        TEXT    NOT NULL,
    run_timestamp     TEXT    NOT NULL,
    config            TEXT    NOT NULL,
    decision          TEXT    NOT NULL,
    quality_score     REAL    NOT NULL,
    cost_usd          REAL,
    elapsed_seconds   REAL,
    stop_loss         REAL,
    price_target      REAL,
    entry_price       REAL,
    position_size_pct REAL,
    risk_reward       REAL,
    actionable        INTEGER,
    portfolio_equity  REAL,
    held_at_analysis  INTEGER,
    held_shares       INTEGER,
    held_avg_cost     REAL,
    result_file       TEXT,
    UNIQUE(ticker, trade_date, config)
);

CREATE TABLE IF NOT EXISTS orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id       INTEGER REFERENCES analyses(id),
    ticker            TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    side              TEXT    NOT NULL,
    qty               INTEGER NOT NULL,
    entry_price       REAL,
    stop_loss         REAL,
    take_profit       REAL,
    approved          INTEGER NOT NULL,
    rejection_reasons TEXT,
    action            TEXT    NOT NULL,
    alpaca_order_id   TEXT,
    alpaca_status     TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date    TEXT    NOT NULL,
    account_equity   REAL    NOT NULL,
    buying_power     REAL    NOT NULL,
    cash             REAL    NOT NULL,
    positions_json   TEXT    NOT NULL,
    total_positions  INTEGER,
    UNIQUE(snapshot_date)
);
"""


class PortfolioDatabase:
    """Low-level SQLite wrapper.  Use PortfolioTracker for business logic."""

    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── connection helper ──────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    # ── analyses ───────────────────────────────────────────────────────────

    def upsert_analysis(self, row: dict) -> int:
        """INSERT OR REPLACE an analysis row; return its rowid."""
        sql = """
            INSERT OR REPLACE INTO analyses (
                ticker, trade_date, run_timestamp, config, decision,
                quality_score, cost_usd, elapsed_seconds,
                stop_loss, price_target, entry_price, position_size_pct,
                risk_reward, actionable, portfolio_equity,
                held_at_analysis, held_shares, held_avg_cost, result_file
            ) VALUES (
                :ticker, :trade_date, :run_timestamp, :config, :decision,
                :quality_score, :cost_usd, :elapsed_seconds,
                :stop_loss, :price_target, :entry_price, :position_size_pct,
                :risk_reward, :actionable, :portfolio_equity,
                :held_at_analysis, :held_shares, :held_avg_cost, :result_file
            )
        """
        with self._conn() as conn:
            cur = conn.execute(sql, row)
            # After REPLACE the lastrowid reflects the new rowid
            return cur.lastrowid

    def get_analysis_id(self, ticker: str, trade_date: str, config: str) -> Optional[int]:
        sql = """SELECT id FROM analyses
                 WHERE ticker=? AND trade_date=? AND config=?"""
        with self._conn() as conn:
            row = conn.execute(sql, (ticker, trade_date, config)).fetchone()
            return row["id"] if row else None

    def get_recent_analyses(self, ticker: str, limit: int = 10) -> List[dict]:
        sql = """SELECT * FROM analyses WHERE ticker=?
                 ORDER BY trade_date DESC, run_timestamp DESC LIMIT ?"""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, (ticker, limit)).fetchall()]

    def get_decision_history(self, ticker: str) -> List[dict]:
        sql = """SELECT trade_date, decision, quality_score, config
                 FROM analyses WHERE ticker=?
                 ORDER BY trade_date DESC"""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, (ticker,)).fetchall()]

    def get_date_summary(self, trade_date: str) -> List[dict]:
        sql = """SELECT * FROM analyses WHERE trade_date=?
                 ORDER BY run_timestamp"""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, (trade_date,)).fetchall()]

    # ── orders ─────────────────────────────────────────────────────────────

    def insert_order(self, row: dict) -> int:
        sql = """
            INSERT INTO orders (
                analysis_id, ticker, timestamp, side, qty,
                entry_price, stop_loss, take_profit,
                approved, rejection_reasons, action,
                alpaca_order_id, alpaca_status
            ) VALUES (
                :analysis_id, :ticker, :timestamp, :side, :qty,
                :entry_price, :stop_loss, :take_profit,
                :approved, :rejection_reasons, :action,
                :alpaca_order_id, :alpaca_status
            )
        """
        with self._conn() as conn:
            cur = conn.execute(sql, row)
            return cur.lastrowid

    def get_recent_orders(self, limit: int = 20) -> List[dict]:
        sql = """SELECT * FROM orders ORDER BY timestamp DESC LIMIT ?"""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]

    # ── snapshots ──────────────────────────────────────────────────────────

    def upsert_snapshot(self, row: dict) -> None:
        sql = """
            INSERT OR REPLACE INTO portfolio_snapshots (
                snapshot_date, account_equity, buying_power, cash,
                positions_json, total_positions
            ) VALUES (
                :snapshot_date, :account_equity, :buying_power, :cash,
                :positions_json, :total_positions
            )
        """
        with self._conn() as conn:
            conn.execute(sql, row)

    def get_snapshots(self, days: int = 30) -> List[dict]:
        sql = """SELECT * FROM portfolio_snapshots
                 ORDER BY snapshot_date DESC LIMIT ?"""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, (days,)).fetchall()]
