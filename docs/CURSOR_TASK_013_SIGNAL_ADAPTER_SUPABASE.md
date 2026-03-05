# Task 013: Signal Adapter & Supabase Writer

**Priority:** HIGH — Unblocks platform UI integration
**Depends on:** Task 012 (complete)
**Blocked by:** Nothing — can start immediately

---

## Objective

Build a signal adapter that transforms POC pipeline output into the platform UI's `AISignal` schema and writes it to Supabase. This is the bridge between the Python analysis pipeline and the React/TypeScript platform frontend.

The UI agent is building against seed data right now. When this task ships, real pipeline signals will flow into the same Supabase table the UI already reads from.

---

## Background

The POC pipeline produces structured JSON results (saved to `results/{TICKER}/analysis_{DATE}_{CONFIG}.json`). The platform UI expects signals in a specific `AISignal` schema (documented below). Task 013 bridges the gap with three deliverables:

1. **Signal Adapter** — Python module that maps pipeline fields to AISignal fields
2. **Supabase Writer** — Python module that inserts transformed signals into Supabase
3. **Pipeline Integration** — Hook the adapter into `run_analysis.py` and `run_batch.py` so signals are written automatically after each analysis

---

## Deliverable 1: Signal Adapter (`src/integration/signal_adapter.py`)

### Input: Pipeline Result Dict

The pipeline produces this structure (from `run_analysis.py`):

```python
result = {
    "ticker": "AAPL",
    "decision": "SELL",                           # BUY|HOLD|SELL|UNKNOWN
    "quality_score": {
        "composite": 9.4,                         # 0-10
        "reasoning_depth": 8,
        "data_grounding": 9,
        "risk_awareness": 10,
        "decision_consistent": True,
        "has_stop_loss": True,
        "has_price_target": True,
        "has_position_sizing": True
    },
    "final_trade_decision_text": "## Summary...",  # Full markdown analysis
    "trader_investment_plan": "## KEY METRICS...",
    "hybrid_config": "hybrid_haiku_tools",
    "run_timestamp": "2026-03-03T15:05:25.374611",
    "elapsed_seconds": 1056.0,
    "cost_breakdown": {
        "total_usd": 0.0626,
        "by_model": { ... },
        "cache_hits": 2,
        "cache_hit_rate_pct": 50.0
    },
    "portfolio_context": {
        "account_equity": 100000.0,
        "buying_power": 200000.0,
        "held": False,
        "shares": 0,
        "avg_cost": None,
        "unrealized_pnl": None
    }
}
```

Trade parameters are extracted separately via `TradeParams` (from `src/execution/trade_params.py`):

```python
trade_params = TradeParams(
    ticker="AAPL",
    decision="SELL",
    quality_score=9.4,
    entry_price=264.72,
    stop_loss=238.0,
    price_target=245.0,
    position_pct=2.0,
    risk_reward_ratio=1.5,
    confidence="medium-high",
    ...
)
```

### Output: AISignal Dict

The platform UI expects this exact schema:

```python
ai_signal = {
    "id": "sig_AAPL_20260303_143025",            # Unique ID
    "symbol": "AAPL",                              # Ticker symbol (1-10 chars)
    "strategy": "sell",                            # Lowercase decision
    "alphaScore": 0.94,                            # quality_score.composite / 10
    "confidence": 0.80,                            # reasoning_depth / 10
    "confidenceInterval": {                        # Derived from quality spread
        "lower": 0.75,                             # confidence - spread
        "upper": 0.89,                             # confidence + spread (capped at 1.0)
        "level": "95%"
    },
    "entryPrice": 264.72,                          # From trade_params
    "stopLoss": 238.0,                             # From trade_params
    "takeProfit": 245.0,                           # From trade_params (price_target)
    "riskFlags": ["high_volatility"],              # Extracted from analysis
    "reasoning": "Strong technical...",            # Condensed reasoning text
    "reasoningPath": ["technical_analysis", ...],  # Analysis step names
    "promptVersion": "v1.0.0",                     # Semantic version
    "evidenceIds": [                               # Source evidence links
        {
            "type": "analysis",
            "id": "market_report_AAPL_20260303",
            "timestamp": "2026-03-03T15:05:25Z"
        },
        {
            "type": "analysis",
            "id": "fundamentals_report_AAPL_20260303",
            "timestamp": "2026-03-03T15:05:25Z"
        }
    ],
    "createdAt": "2026-03-03T15:05:25.374611Z",   # ISO 8601
    "expiresAt": "2026-03-04T15:05:25.374611Z"    # 24h default TTL
}
```

### Transformation Rules

Create a function `transform_to_signal(result: dict, trade_params: TradeParams) -> dict`:

| Platform Field | Source | Transform |
|---|---|---|
| `id` | Generated | `f"sig_{ticker}_{date}_{time}"` — must be unique |
| `symbol` | `result["ticker"]` | Direct copy |
| `strategy` | `result["decision"]` | Lowercase: `"BUY"` → `"buy"` |
| `alphaScore` | `result["quality_score"]["composite"]` | Divide by 10, clamp to [0, 1] |
| `confidence` | `result["quality_score"]["reasoning_depth"]` | Divide by 10, clamp to [0, 1] |
| `confidenceInterval` | Derived | `lower = max(0, confidence - 0.08)`, `upper = min(1, confidence + 0.08)`, `level = "95%"` |
| `entryPrice` | `trade_params.entry_price` | Direct, or `None` if not extracted |
| `stopLoss` | `trade_params.stop_loss` | Direct, or `None` if not extracted |
| `takeProfit` | `trade_params.price_target` | Direct, or `None` if not extracted |
| `riskFlags` | Extracted | See risk flag extraction rules below |
| `reasoning` | `result["final_trade_decision_text"]` | First 2000 chars of the summary section |
| `reasoningPath` | Generated | List of agent names that ran: `["market_analyst", "social_media_analyst", "news_analyst", "fundamentals_analyst", "bull_researcher", "bear_researcher", "risk_judge"]` |
| `promptVersion` | Config | Start at `"v1.0.0"`, bump on prompt changes |
| `evidenceIds` | Generated | One entry per agent report in the analysis |
| `createdAt` | `result["run_timestamp"]` | Parse to ISO 8601 with timezone |
| `expiresAt` | Calculated | `createdAt + 24 hours` (configurable TTL) |

### Risk Flag Extraction

Extract risk flags from the analysis text and trade parameters:

```python
def extract_risk_flags(result: dict, trade_params: TradeParams) -> list[str]:
    flags = []
    text = result.get("final_trade_decision_text", "").lower()

    # Volatility flags
    if "high volatility" in text or "volatile" in text:
        flags.append("high_volatility")

    # Earnings proximity
    if "earnings" in text and ("upcoming" in text or "approaching" in text or "next week" in text):
        flags.append("earnings_approaching")

    # Low confidence
    if trade_params.confidence and "low" in trade_params.confidence.lower():
        flags.append("low_confidence")

    # Poor risk/reward
    if trade_params.risk_reward_ratio and trade_params.risk_reward_ratio < 1.5:
        flags.append("poor_risk_reward")

    # Decision inconsistency
    if not result.get("quality_score", {}).get("decision_consistent", True):
        flags.append("decision_inconsistent")

    # Missing execution params
    if not trade_params.has_bracket_params:
        flags.append("missing_bracket_params")

    return flags
```

---

## Deliverable 2: Supabase Writer (`src/integration/supabase_writer.py`)

### Setup

```python
# Uses supabase-py client
# pip install supabase

from supabase import create_client

class SupabaseWriter:
    def __init__(self, url: str = None, key: str = None):
        """Initialize with env vars SUPABASE_URL and SUPABASE_SERVICE_KEY."""

    def write_signal(self, signal: dict) -> dict:
        """Insert a single AISignal into the signals table. Returns inserted row."""

    def write_signals_batch(self, signals: list[dict]) -> list[dict]:
        """Insert multiple signals. Used by run_batch.py."""

    def get_latest_signal(self, ticker: str) -> dict | None:
        """Get the most recent signal for a ticker. For deduplication."""

    def cleanup_expired(self) -> int:
        """Delete signals past their expiresAt. Returns count deleted."""
```

### Configuration

Add to `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...  # Service role key (NOT anon key) for server-side writes
```

Add to `config/settings.yaml` (or create):
```yaml
supabase:
  signal_ttl_hours: 24        # Default signal expiry
  write_enabled: true          # Toggle for dry-run mode
  table_name: "signals"        # Target table
```

### Supabase Table Schema

The platform UI repo already defines a `signals` table in `backend/src/db/schema.ts`. The adapter must write rows that match this exact schema. The key columns are:

```
id              UUID PRIMARY KEY
symbol          VARCHAR(10)
strategy        VARCHAR(20)
alpha_score     NUMERIC(28,10)
confidence      NUMERIC(28,10)
confidence_interval  JSONB          -- {lower, upper, level}
entry_price     NUMERIC(28,10)
stop_loss       NUMERIC(28,10)
take_profit     NUMERIC(28,10)
risk_flags      JSONB              -- string array
reasoning       TEXT
reasoning_path  JSONB              -- string array
prompt_version  VARCHAR(20)
evidence_ids    JSONB              -- array of {type, id, url?, timestamp?}
created_at      TIMESTAMP
expires_at      TIMESTAMP
```

**Important:** The Supabase table uses `snake_case` column names. The frontend API layer converts to `camelCase` for JavaScript. The adapter should write `snake_case` to Supabase:

```python
supabase_row = {
    "id": signal["id"],
    "symbol": signal["symbol"],
    "strategy": signal["strategy"],
    "alpha_score": signal["alphaScore"],          # camelCase → snake_case
    "confidence": signal["confidence"],
    "confidence_interval": signal["confidenceInterval"],
    "entry_price": signal["entryPrice"],
    "stop_loss": signal["stopLoss"],
    "take_profit": signal["takeProfit"],
    "risk_flags": signal["riskFlags"],
    "reasoning": signal["reasoning"],
    "reasoning_path": signal["reasoningPath"],
    "prompt_version": signal["promptVersion"],
    "evidence_ids": signal["evidenceIds"],
    "created_at": signal["createdAt"],
    "expires_at": signal["expiresAt"],
}
```

### Deduplication

Before writing, check if a signal already exists for the same `(symbol, strategy, DATE(created_at))`. If it does, use `upsert` to replace the older signal. This prevents duplicate signals from re-runs.

### Error Handling

- If Supabase is unreachable, log the error and continue (don't crash the pipeline)
- If `write_enabled` is `false`, log the signal that would have been written but skip the actual write
- Always log: signal ID, symbol, strategy, alphaScore, and whether the write succeeded

---

## Deliverable 3: Pipeline Integration

### Hook into `run_analysis.py`

After a successful analysis, add signal writing:

```python
# At the end of the analysis, after result is built:
if supabase_enabled:
    from src.integration.signal_adapter import transform_to_signal
    from src.integration.supabase_writer import SupabaseWriter

    signal = transform_to_signal(result, trade_params)
    writer = SupabaseWriter()
    writer.write_signal(signal)
```

**Make this opt-in** via a `--publish` flag:

```
python -m src.run_analysis --ticker AAPL --config hybrid_haiku_tools --publish
```

Without `--publish`, the pipeline behaves exactly as before. The flag is additive.

### Hook into `run_batch.py`

Same pattern — after all tickers are analyzed, write signals in batch:

```python
if args.publish:
    signals = [transform_to_signal(r, tp) for r, tp in zip(results, trade_params_list)]
    writer = SupabaseWriter()
    writer.write_signals_batch(signals)
```

### Backward Compatibility

- Zero changes to existing pipeline logic
- Zero vendor modifications
- `--publish` is optional; omitting it preserves current behavior
- Local result JSON files are still written regardless of `--publish`
- The SQLite portfolio database (`data/portfolio.db`) is still written regardless of `--publish`

---

## New Files

```
src/integration/
    __init__.py
    signal_adapter.py          # Deliverable 1
    supabase_writer.py         # Deliverable 2
tests/
    test_signal_adapter.py     # Unit tests for adapter
    test_supabase_writer.py    # Unit tests for writer (mock Supabase)
config/
    supabase.yaml              # Supabase configuration (optional, can use .env)
```

## Modified Files

```
src/run_analysis.py            # Add --publish flag and signal writing hook
src/run_batch.py               # Add --publish flag and batch signal writing
requirements.txt               # Add supabase dependency
.env.example                   # Add SUPABASE_URL and SUPABASE_SERVICE_KEY
```

---

## Exit Criteria

1. `signal_adapter.py` transforms a pipeline result dict into a valid AISignal dict
2. All required AISignal fields are populated (id, symbol, strategy, alphaScore, confidence, reasoning, createdAt)
3. Optional fields are populated when available (entryPrice, stopLoss, takeProfit, confidenceInterval, evidenceIds)
4. Risk flags are extracted from analysis text
5. `supabase_writer.py` can write a signal to Supabase
6. `supabase_writer.py` can write a batch of signals
7. Deduplication prevents duplicate signals for the same ticker+date
8. `--publish` flag works on `run_analysis.py`
9. `--publish` flag works on `run_batch.py`
10. Without `--publish`, pipeline behavior is unchanged
11. Graceful error handling: Supabase failures don't crash the pipeline
12. Unit tests pass for adapter transformation logic
13. Unit tests pass for writer (with mocked Supabase client)
14. Zero vendor modifications
15. All existing tests still pass

---

## Dependencies

```
supabase>=2.0.0
```

Install: `pip install supabase --break-system-packages`

---

## Notes

- The UI agent is building the platform backend to read from the same Supabase signals table. They're using seed data for now. Once this task ships, real signals will appear in the UI automatically.
- The `promptVersion` field should start at `"v1.0.0"`. When we modify pipeline prompts in the future, we bump this version. The UI displays it for EU AI Act audit compliance.
- The `evidenceIds` field links to source data. For now, generate entries for each agent report (market, sentiment, news, fundamentals). In the future, we can add specific SEC filing IDs and news article URLs.
- The signal `expiresAt` defaults to 24 hours after creation. The UI can filter on this to show only fresh signals.
