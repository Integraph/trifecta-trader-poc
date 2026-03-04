# CURSOR TASK 012: Portfolio-Aware Execution & Watchlist Batch Mode

## Context

The Stock Trader is an excellent single-ticker analysis tool — Tasks 001-011 built a production-quality pipeline with hybrid LLM routing (85% cost savings), analyst caching, dual-source extraction, and paper trading execution. But it operates in a vacuum: it doesn't know what we hold, can't process multiple tickers, and can't make portfolio-level decisions.

The TSLA live test (Task 009) demonstrated this gap: the trader recommended SELL but didn't know we don't own TSLA. The rejection was correct safety behavior, but a portfolio-aware system would have surfaced "no position to sell" *before* spending $0.15 on deep analysis.

This task bridges the gap from "analysis tool" to "trading system."

**Three deliverables, in order:**
1. **Portfolio context injection** — query positions and buying power before analysis, inject into the pipeline
2. **Watchlist batch mode** — process multiple tickers sequentially with portfolio-level orchestration
3. **Portfolio state tracking** — SQLite database for analysis history, decision log, and P&L tracking

---

## Step 1: Portfolio Context Injection

### 1a: Pre-Analysis Portfolio Query

**File:** `src/run_analysis.py` (modify)

Before the analysis pipeline runs, query Alpaca for current portfolio state. This happens whether or not `--execute` is passed — the portfolio context informs the *analysis*, not just the execution.

```python
# New: query portfolio before analysis
from src.execution.position_manager import PositionManager

pm = PositionManager()
account = pm.get_account_state()
positions = pm.get_positions()
current_position = pm.get_position(ticker)

portfolio_context = {
    "account_equity": account.equity,
    "buying_power": account.buying_power,
    "cash": account.cash,
    "total_positions": len(positions),
    "current_position": {
        "held": current_position is not None,
        "shares": current_position.qty if current_position else 0,
        "avg_cost": current_position.avg_entry_price if current_position else None,
        "unrealized_pnl": current_position.unrealized_pl if current_position else None,
        "current_value": current_position.market_value if current_position else None,
    } if current_position else {"held": False},
    "portfolio_allocation": {
        pos.symbol: {
            "pct": (pos.market_value / account.equity * 100) if account.equity > 0 else 0,
            "shares": pos.qty,
        }
        for pos in positions.values()
    },
}
```

Print a portfolio summary before analysis begins:

```
══════════════════════════════════════════════════════
PORTFOLIO CONTEXT
══════════════════════════════════════════════════════
  Account equity:   $100,234.50
  Buying power:     $45,120.00
  Cash:             $22,560.00
  Positions:        4 (AAPL 12%, MSFT 8%, NVDA 10%, GOOGL 5%)

  TSLA:             NOT HELD
  Recommendation context: BUY analysis only (no position to sell)
══════════════════════════════════════════════════════
```

### 1b: Smart Pre-Filtering

**File:** `src/run_analysis.py` (modify)

Add a pre-analysis check that can skip expensive analysis when the answer is obvious:

- If decision would be SELL but we don't hold the ticker → warn the user: "No position in {ticker}. SELL recommendations won't be actionable. Run analysis anyway? [Y/n]" (auto-proceed in batch mode, just log the warning)
- If we already hold the maximum position (15%) in the ticker → warn: "Already at max allocation for {ticker}."
- If buying power is below $1,000 → warn: "Insufficient buying power for new positions."

These warnings are **informational only** — they don't block analysis. The user or batch runner can still choose to proceed for informational purposes.

### 1c: Portfolio Context in Results

**File:** `src/run_analysis.py` (modify)

Add the `portfolio_context` to the saved result JSON:

```json
{
  "ticker": "AAPL",
  "portfolio_context": {
    "account_equity": 100234.50,
    "buying_power": 45120.00,
    "held": true,
    "shares": 50,
    "avg_cost": 182.30,
    "unrealized_pnl": 4120.00,
    "portfolio_pct": 12.1
  },
  "decision": "HOLD",
  "quality_score": { ... },
  ...
}
```

This lets us see the portfolio state at the time each decision was made.

---

## Step 2: Watchlist Batch Mode

### 2a: Watchlist File Format

**Directory:** `config/`

Watchlists are simple YAML files:

```yaml
# config/watchlists/default.yaml
name: "Default Watchlist"
description: "Core positions and candidates"
tickers:
  - AAPL
  - MSFT
  - NVDA
  - GOOGL
  - TSLA
  - AMD
  - META
  - AMZN
```

```yaml
# config/watchlists/small_cap.yaml
name: "Small-Cap Candidates"
description: "High-growth small caps from manual screening"
tickers:
  - DCMM
  - IONQ
  - RKLB
  - AEHR
  - SOUN
```

### 2b: Batch Runner

**New file:** `src/run_batch.py`

A new CLI entry point that processes a watchlist sequentially:

```bash
# Process entire watchlist
python -m src.run_batch --watchlist config/watchlists/default.yaml --hybrid hybrid_haiku_tools

# Process with execution enabled (will still require order approval per-ticker)
python -m src.run_batch --watchlist config/watchlists/default.yaml --hybrid hybrid_haiku_tools --execute

# Process with dry-run (calculate orders, don't submit)
python -m src.run_batch --watchlist config/watchlists/default.yaml --hybrid hybrid_haiku_tools --dry-run

# Process specific tickers (no watchlist file needed)
python -m src.run_batch --tickers AAPL,MSFT,NVDA --hybrid hybrid_haiku_tools
```

**CLI Arguments:**

```
--watchlist PATH         # Path to YAML watchlist file
--tickers AAPL,MSFT      # Comma-separated tickers (alternative to --watchlist)
--hybrid CONFIG          # Hybrid LLM config (default: hybrid_haiku_tools)
--date DATE              # Analysis date (default: today)
--execute                # Enable order execution for approved trades
--dry-run                # Calculate orders without submitting
--no-cache               # Disable analyst caching
--no-cost-breakdown      # Skip per-run cost reports
--max-concurrent 1       # Reserved for future parallelism (always 1 for now)
--skip-held              # Skip tickers already at max allocation
--priority-sort          # Analyze high-priority tickers first (see below)
```

**Batch Execution Flow:**

```python
def run_batch(watchlist, config, date, execute, dry_run):
    # 1. Load watchlist
    tickers = load_watchlist(watchlist)

    # 2. Query portfolio state ONCE (not per-ticker)
    pm = PositionManager()
    account = pm.get_account_state()
    positions = pm.get_positions()

    # 3. Print portfolio summary
    print_portfolio_summary(account, positions)

    # 4. Process each ticker
    results = []
    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] Analyzing {ticker}...")

        # Pre-filter warnings
        position = positions.get(ticker)
        if skip_held and position and position_pct >= MAX_POSITION_PCT:
            print(f"  SKIPPED: Already at max allocation ({position_pct:.1f}%)")
            continue

        # Run single-ticker analysis (reuse existing run_analysis logic)
        result = run_single_analysis(ticker, config, date, execute, dry_run, portfolio_context)
        results.append(result)

        # Print running cost total
        print(f"  Cost so far: ${sum(r.cost for r in results):.4f}")

    # 5. Print batch summary
    print_batch_summary(results, account)
```

### 2c: Batch Summary Report

After all tickers are processed, print a consolidated summary:

```
══════════════════════════════════════════════════════
BATCH ANALYSIS COMPLETE
══════════════════════════════════════════════════════
  Watchlist:        default.yaml (8 tickers)
  Config:           hybrid_haiku_tools
  Date:             2026-03-04
  Total time:       48m 23s
  Total cost:       $0.87

  RESULTS
  ──────────────────────────────────────────────────
  AAPL    BUY    9.4/10   $0.11   Target: $215   (HELD: 50 shares)
  MSFT    HOLD   9.1/10   $0.10   —              (HELD: 30 shares)
  NVDA    BUY    9.7/10   $0.12   Target: $950   (NOT HELD)
  GOOGL   HOLD   8.8/10   $0.09   —              (HELD: 25 shares)
  TSLA    SELL   9.1/10   $0.11   Target: $295   (NOT HELD — no action)
  AMD     BUY    9.3/10   $0.11   Target: $185   (NOT HELD)
  META    HOLD   8.9/10   $0.10   —              (NOT HELD)
  AMZN    BUY    9.5/10   $0.13   Target: $225   (NOT HELD)

  ACTIONABLE SIGNALS
  ──────────────────────────────────────────────────
  BUY:   AAPL (add), NVDA (new), AMD (new), AMZN (new)
  SELL:  TSLA (skipped — not held)
  HOLD:  MSFT, GOOGL, META

  ORDERS (--dry-run)
  ──────────────────────────────────────────────────
  NVDA   BUY   12 shares @ ~$890   $10,680   APPROVED (R/R: 2.1)
  AMD    BUY   45 shares @ ~$162   $7,290    APPROVED (R/R: 1.8)
  AMZN   BUY   40 shares @ ~$210   $8,400    APPROVED (R/R: 1.9)
  AAPL   BUY   10 shares @ ~$195   $1,950    REJECTED (already at 12%)
══════════════════════════════════════════════════════
```

### 2d: Save Batch Results

**File:** `results/batch/batch_YYYYMMDD_HHMMSS.json`

Save a consolidated batch result file with all ticker results, portfolio context, and the batch summary. This feeds the portfolio tracker (Step 3).

---

## Step 3: Portfolio State Tracking

### 3a: SQLite Database

**New file:** `src/portfolio/database.py`

Create a SQLite database at `data/portfolio.db` with these tables:

```sql
-- Analysis history
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    config TEXT NOT NULL,
    decision TEXT NOT NULL,          -- BUY/HOLD/SELL
    quality_score REAL NOT NULL,
    cost_usd REAL,
    elapsed_seconds REAL,
    stop_loss REAL,
    price_target REAL,
    entry_price REAL,
    position_size_pct REAL,
    risk_reward REAL,
    actionable BOOLEAN,
    portfolio_equity REAL,           -- Account equity at time of analysis
    held_at_analysis BOOLEAN,        -- Did we hold this ticker?
    held_shares INTEGER,             -- How many shares?
    held_avg_cost REAL,              -- Average cost basis
    result_file TEXT,                -- Path to full result JSON
    UNIQUE(ticker, trade_date, config)
);

-- Order log
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER REFERENCES analyses(id),
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    side TEXT NOT NULL,               -- buy/sell
    qty INTEGER NOT NULL,
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    approved BOOLEAN NOT NULL,
    rejection_reasons TEXT,           -- JSON array
    action TEXT NOT NULL,             -- EXECUTED/REJECTED/DRY_RUN
    alpaca_order_id TEXT,
    alpaca_status TEXT
);

-- Daily portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    account_equity REAL NOT NULL,
    buying_power REAL NOT NULL,
    cash REAL NOT NULL,
    positions_json TEXT NOT NULL,     -- JSON blob of all positions
    total_positions INTEGER,
    UNIQUE(snapshot_date)
);
```

### 3b: Database Integration

**New file:** `src/portfolio/tracker.py`

```python
class PortfolioTracker:
    def __init__(self, db_path="data/portfolio.db"):
        ...

    def log_analysis(self, result: dict, portfolio_context: dict):
        """Insert analysis result into the analyses table."""
        ...

    def log_order(self, analysis_id: int, order_calc: OrderCalculation, action: str):
        """Insert order attempt into the orders table."""
        ...

    def take_snapshot(self):
        """Capture current portfolio state from Alpaca."""
        ...

    def get_analysis_history(self, ticker: str, limit: int = 10) -> list:
        """Get recent analyses for a ticker."""
        ...

    def get_decision_history(self, ticker: str) -> list:
        """Get all BUY/SELL decisions for a ticker."""
        ...

    def get_daily_pnl(self, days: int = 30) -> list:
        """Calculate P&L from portfolio snapshots."""
        ...

    def get_batch_summary(self, date: str) -> dict:
        """Summarize all analyses run on a given date."""
        ...
```

### 3c: Wire Into Pipeline

**File:** `src/run_analysis.py` (modify) and `src/run_batch.py` (new)

After each analysis completes:
1. Call `tracker.log_analysis(result, portfolio_context)` to record the analysis
2. If execution was attempted, call `tracker.log_order(analysis_id, order_calc, action)`
3. At the start of each batch run, call `tracker.take_snapshot()` to capture the portfolio state

### 3d: Portfolio Summary CLI

Add a `--portfolio` flag to `run_analysis.py` that prints a portfolio summary from the database without running any analysis:

```bash
python -m src.run_analysis --portfolio
```

Output:
```
══════════════════════════════════════════════════════
PORTFOLIO SUMMARY (2026-03-04)
══════════════════════════════════════════════════════
  Account equity:   $100,234.50
  Buying power:     $45,120.00
  Positions:        4

  HOLDINGS
  ──────────────────────────────────────────────────
  AAPL    50 shares   $182.30 avg   +$4,120   +4.5%   12.1% of portfolio
  MSFT    30 shares   $410.50 avg   -$320     -0.3%   8.2% of portfolio
  NVDA    15 shares   $880.00 avg   +$1,050   +8.0%   10.5% of portfolio
  GOOGL   25 shares   $175.00 avg   +$225     +0.5%   5.1% of portfolio

  RECENT ANALYSES (last 7 days)
  ──────────────────────────────────────────────────
  2026-03-04  AAPL   BUY    9.4/10   (held: yes)
  2026-03-04  TSLA   SELL   9.1/10   (held: no — skipped)
  2026-03-03  AAPL   SELL   9.1/10   (held: yes)
  2026-03-03  TSLA   SELL   9.4/10   (held: no — skipped)

  ORDERS (last 7 days)
  ──────────────────────────────────────────────────
  2026-03-04  NVDA   BUY   12 shares   APPROVED → EXECUTED
  2026-03-03  TSLA   SELL  0 shares    REJECTED (no position)
══════════════════════════════════════════════════════
```

---

## Step 4: Tests

**New test file:** `tests/test_portfolio.py`

- Test watchlist YAML loading (valid file, missing file, empty file)
- Test portfolio context generation (mock Alpaca API)
- Test pre-filter warnings (no position for SELL, max allocation, low buying power)
- Test batch summary calculation
- Test SQLite database creation and schema
- Test `log_analysis()` with mock result data
- Test `log_order()` with approved and rejected orders
- Test `take_snapshot()` with mock Alpaca data
- Test `get_analysis_history()` returns correct records
- Test `get_daily_pnl()` calculation
- Test batch results file writing

**Extend existing tests:**
- Verify existing `run_analysis.py` single-ticker mode still works unchanged
- Verify `--portfolio` flag works without analysis

**Run full test suite** and confirm no regressions.

---

## Step 5: Document Results

**New file:** `docs/TASK_012_REPORT.md`

Include:
- Summary of all three deliverables
- Example output from a batch run (at least 3 tickers)
- Example output from `--portfolio` summary
- Database schema documentation
- Any issues with Alpaca API integration
- Test results

---

## Exit Criteria

- [ ] Portfolio context queried and printed before every analysis run
- [ ] Portfolio context saved in result JSON files
- [ ] Pre-filter warnings displayed for non-actionable scenarios
- [ ] Watchlist YAML format works with batch runner
- [ ] `run_batch.py` processes multiple tickers sequentially
- [ ] Batch summary printed after all tickers complete
- [ ] Batch results saved to `results/batch/`
- [ ] SQLite database created with analyses, orders, and snapshots tables
- [ ] Analysis results logged to database automatically
- [ ] Order attempts logged to database automatically
- [ ] `--portfolio` flag prints portfolio summary from database
- [ ] Analyst cache benefits carry across tickers in the same batch (shared cache, not cleared between tickers)
- [ ] All existing tests pass, new tests added
- [ ] TASK_012_REPORT.md written

---

## Important Notes

- **Zero vendor modifications.** All changes go in `src/` and `config/`. Do not modify anything in `vendor/TradingAgents/`.
- **Do not change the existing single-ticker flow.** `run_analysis.py` with a single `--ticker` must work exactly as before. The batch runner is a new entry point alongside it.
- **Alpaca API rate limits.** The Alpaca paper API has rate limits. Query positions and account state ONCE at the start of a batch run, not per-ticker. Cache the portfolio state for the duration of the batch.
- **Sequential processing only.** Do not parallelize ticker analysis. The LLM pipeline is not thread-safe and shared resources (Ollama, cache) would conflict. The `--max-concurrent 1` flag is a placeholder for future work.
- **Cache sharing across batch tickers.** Do NOT clear the analyst cache between tickers in a batch run. Different tickers have different cache keys, so they won't collide. But news/sentiment data might overlap if tickers are in the same sector — that's fine, the cache TTLs handle freshness.
- **Database location:** `data/portfolio.db` at repo root. Add `data/` to `.gitignore` (it likely already is for results storage).
- **Watchlist directory:** `config/watchlists/`. Include a `default.yaml` example with 5-8 tickers.
- **Don't over-engineer the portfolio tracker.** This is an MVP — SQLite, simple queries, no ORM. Keep it under 200 lines. The Scanner will eventually feed candidates via the JSON queue (not the watchlist), so the watchlist is a stopgap for manual operation.
