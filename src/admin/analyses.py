"""
Analysis History Endpoints — /analyses/*

GET /analyses/recent         Recent analyses with outcome status
GET /analyses/{id}           Single analysis with full outcome data
GET /analyses/stats          Aggregate statistics
"""

import logging
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.admin.dependencies import get_db

logger = logging.getLogger(__name__)

analyses_router = APIRouter()


@analyses_router.get("/stats")
async def get_analyses_stats():
    """Return aggregate statistics across all recorded analyses."""
    db  = get_db()
    sql = """
        SELECT
            COUNT(*)                          AS total_analyses,
            COALESCE(SUM(cost_usd), 0)        AS total_cost_usd,
            COALESCE(AVG(quality_score), 0)   AS avg_quality_score,
            COALESCE(AVG(elapsed_seconds), 0) AS avg_elapsed_seconds,
            SUM(CASE decision WHEN 'BUY'  THEN 1 ELSE 0 END) AS buy_count,
            SUM(CASE decision WHEN 'SELL' THEN 1 ELSE 0 END) AS sell_count,
            SUM(CASE decision WHEN 'HOLD' THEN 1 ELSE 0 END) AS hold_count,
            COUNT(DISTINCT ticker)            AS unique_tickers,
            SUM(CASE WHEN trade_date = DATE('now') THEN 1 ELSE 0 END) AS analyses_today
          FROM analyses
    """
    with db._conn() as conn:
        row = dict(conn.execute(sql).fetchone())

    return {
        "total_analyses":     row["total_analyses"],
        "total_cost_usd":     round(row["total_cost_usd"] or 0, 4),
        "avg_quality_score":  round(row["avg_quality_score"] or 0, 2),
        "decision_breakdown": {
            "BUY":  row["buy_count"],
            "SELL": row["sell_count"],
            "HOLD": row["hold_count"],
        },
        "avg_elapsed_seconds": round(row["avg_elapsed_seconds"] or 0, 1),
        "analyses_today":      row["analyses_today"],
        "unique_tickers":      row["unique_tickers"],
    }


@analyses_router.get("/recent")
async def get_recent_analyses(
    days:   int           = Query(default=7, ge=1, le=90),
    ticker: Optional[str] = Query(default=None),
    limit:  int           = Query(default=50, ge=1, le=500),
):
    """Return recent analyses with accuracy outcome_status from LEFT JOIN."""
    db     = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    params       = [cutoff]
    count_params = [cutoff]
    ticker_clause_a = ""        # for queries aliasing analyses as 'a'
    ticker_clause   = ""        # for queries without alias
    if ticker:
        ticker_clause_a = "AND a.ticker = ?"
        ticker_clause   = "AND ticker = ?"
        params.append(ticker.upper())
        count_params.append(ticker.upper())

    sql = f"""
        SELECT a.id, a.ticker, a.trade_date, a.decision,
               a.quality_score, a.entry_price, a.stop_loss, a.price_target,
               a.cost_usd, a.elapsed_seconds, a.config, a.run_timestamp,
               so.status AS outcome_status
          FROM analyses a
          LEFT JOIN signal_outcomes so ON so.analysis_id = a.id
         WHERE a.trade_date >= ? {ticker_clause_a}
         ORDER BY a.trade_date DESC, a.run_timestamp DESC
         LIMIT ?
    """
    params.append(limit)

    with db._conn() as conn:
        rows  = [dict(r) for r in conn.execute(sql, params).fetchall()]
        total = conn.execute(
            f"SELECT COUNT(*) FROM analyses WHERE trade_date >= ? {ticker_clause}",
            count_params,
        ).fetchone()[0]

    return {"analyses": rows, "total": total}


@analyses_router.get("/{analysis_id}")
async def get_analysis_detail(analysis_id: int):
    """Return full detail for a single analysis including its signal outcome."""
    db  = get_db()
    sql = """
        SELECT a.*, so.price_at_signal, so.price_t1, so.price_t5, so.price_t10,
               so.direction_correct_t5, so.direction_correct_t10,
               so.target_hit, so.stop_hit, so.return_t5_pct, so.return_t10_pct,
               so.max_favorable_pct, so.max_adverse_pct,
               so.status AS outcome_status
          FROM analyses a
          LEFT JOIN signal_outcomes so ON so.analysis_id = a.id
         WHERE a.id = ?
    """
    with db._conn() as conn:
        row = conn.execute(sql, (analysis_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail={"error": "analysis_not_found"})

    return dict(row)
