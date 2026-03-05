# Task 013 Report — Signal Adapter & Supabase Integration

**Date:** 2026-03-05  
**Status:** ✅ Complete  
**Spec:** `docs/CURSOR_TASK_013_SIGNAL_ADAPTER_SUPABASE.md`

---

## Objective

Build a signal adapter that transforms POC pipeline output (`run_analysis.py` result dict) into the platform UI's `AISignal` schema and writes it to Supabase. Add an opt-in `--publish` flag to both `run_analysis.py` and `run_batch.py`.

---

## Deliverables

### 1. `src/integration/signal_adapter.py`

Core transformation layer. No external dependencies at import time.

**`transform_to_signal(result, trade_params, ttl_hours)`**  
Maps pipeline fields to `AISignal`:

| Pipeline field | AISignal field | Transformation |
|---|---|---|
| `uuid.uuid4()` | `id` | Fresh UUID4 per run |
| `sig_TICKER_YYYYMMDD_HHMMSS` | `signal_ref` (in `evidenceIds`) | Human-readable slug |
| `ticker` | `symbol` | Uppercased |
| `decision.lower()` | `strategy` | `"buy"` / `"sell"` / `"hold"` |
| `quality_score.composite / 10` | `alphaScore` | Clamped `[0, 1]` |
| `quality_score.reasoning_depth / 10` | `confidence` | Clamped `[0, 1]` |
| `confidence ± 0.08` | `confidenceInterval` | Capped at `[0, 1]`, 95% level |
| `trade_params.entry_price` | `entryPrice` | `None` if extraction failed |
| `trade_params.stop_loss` | `stopLoss` | `None` if extraction failed |
| `trade_params.price_target` | `takeProfit` | `None` if extraction failed |
| `extract_risk_flags()` | `riskFlags` | See below |
| `final_trade_decision_text[:2000]` | `reasoning` | Summary section preferred |
| `REASONING_PATH` | `reasoningPath` | Static list of 7 agent names |
| `"v1.0.0"` | `promptVersion` | Bumped on prompt changes |
| `_build_evidence_ids()` | `evidenceIds` | Internal slug + 4 analyst entries |
| `run_timestamp` | `createdAt` | ISO 8601 with UTC tz |
| `createdAt + ttl_hours` | `expiresAt` | Default 24 h |

**`extract_risk_flags(result, trade_params)`**  

Flags extracted from analysis text:
- `high_volatility` — "high volatility" or "volatile" in decision text
- `earnings_approaching` — "earnings" + proximity keyword
- `macro_risk` — recession / Fed / interest rate keywords
- `sector_headwinds` — sector headwind / regulatory / antitrust

Flags extracted from trade params:
- `low_confidence` — `trade_params.confidence` contains "low"
- `poor_risk_reward` — `risk_reward_ratio < 1.5`
- `missing_bracket_params` — `has_bracket_params` is falsy

Flags from quality score:
- `decision_inconsistent` — `quality_score.decision_consistent == False`

**`to_supabase_row(signal)`**  
Converts camelCase AISignal → snake_case Supabase row for insert.

---

### 2. `src/integration/supabase_writer.py`

**`SupabaseWriter`** class:

| Method | Description |
|---|---|
| `write_signal(signal)` | Upsert one AISignal to Supabase; returns row or `None` on error |
| `write_signals_batch(signals)` | Sequential batch write; returns list of successes |
| `get_latest_signal(ticker)` | Most recent row for ticker; `None` on error |
| `cleanup_expired()` | Delete rows where `expires_at < now`; returns count |

**Key behaviours:**
- `write_enabled=False` (from config or arg) logs but makes zero network calls
- All Supabase errors are caught and logged; the pipeline continues
- Deduplication via upsert `on_conflict="symbol,created_at"` prevents re-run duplicates
- Lazy client initialisation — no import-time network call
- Credentials loaded from env vars (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`) or passed directly

---

### 3. `config/supabase.yaml`

```yaml
supabase:
  signal_ttl_hours: 24
  write_enabled: true
  table_name: "signals"
```

Set `write_enabled: false` for dry-run mode (logs signal, no Supabase write).

---

### 4. Pipeline Integration

**`run_analysis.py`**

New `--publish` flag. Trade params are extracted whenever `--publish`, `--execute`, or `--dry-run` is active (graceful fallback to `None` if extraction fails):

```
python -m src.run_analysis --ticker AAPL --hybrid hybrid_haiku_tools --publish
```

The `_publish_signal(result, trade_params)` helper is called after the main pipeline result. It never raises — errors are logged and discarded.

**`run_batch.py`**

New `--publish` flag. Each ticker gets trade params extracted and a signal written per-ticker (additive to existing batch behaviour). Batch signal write is sequential within the existing loop.

```
python -m src.run_batch --tickers AAPL,MSFT,NVDA --hybrid hybrid_haiku_tools --publish
```

---

### 5. Supporting Files

| File | Change |
|---|---|
| `src/integration/__init__.py` | Empty package marker |
| `.env` | Added commented `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` placeholders |
| `.env.example` | Created with all env vars documented including Supabase |
| `pyproject.toml` | Added `supabase>=2.0.0` and `pyyaml` to dependencies |

---

## Test Results

### New Task 013 tests: 59/59 passed ✅

```
tests/test_signal_adapter.py     40 passed
tests/test_supabase_writer.py    19 passed
Total                            59 passed in 0.07s
```

**Test coverage:**
- Required AISignal fields present
- UUID4 generation and format
- alphaScore / confidence clamping and scaling (edge cases: 0, 10, 12, -1)
- confidenceInterval width, capping at [0, 1]
- `strategy` lowercase mapping for BUY / SELL / HOLD
- All risk flag types (volatility, earnings, R/R, bracket, consistency, confidence)
- `None` trade params → null price fields (no crash)
- `signal_ref` slug format and presence in evidenceIds
- 4+ analyst report entries in evidenceIds
- ISO 8601 timestamps
- 24h default TTL and custom TTL
- Invalid timestamp fallback to current UTC time
- `REASONING_PATH` completeness and `promptVersion` semver format
- `to_supabase_row()` camelCase→snake_case conversion, value preservation
- `SupabaseWriter` init from args and env vars
- `write_enabled=False` produces no network call and returns `None`
- Upsert called with correct snake_case row and `on_conflict` param
- Error handling: `write_signal`, `get_latest_signal`, `cleanup_expired` all return gracefully
- Batch write returns only successes, handles empty list

### Full suite: 224 passed, 40 pre-existing failures, 8 skipped

Pre-existing failures (unrelated to Task 013):
- `langchain_google_genai` not installed in test environment (17 test files affected)
- `mistral-small:22b` local tool calling (2 tests, pre-existing from Task 011)

---

## Exit Criteria Verification

| # | Criterion | Status |
|---|---|---|
| 1 | `transform_to_signal()` maps all fields correctly | ✅ |
| 2 | `alphaScore` = composite/10, clamped to [0,1] | ✅ |
| 3 | `confidence` = reasoning_depth/10, clamped to [0,1] | ✅ |
| 4 | `confidenceInterval` = ±0.08 band, capped at [0,1] | ✅ |
| 5 | `id` is UUID4; `signal_ref` slug in evidenceIds | ✅ |
| 6 | `expiresAt` = createdAt + 24h (configurable) | ✅ |
| 7 | `extract_risk_flags()` identifies all 8 flag types | ✅ |
| 8 | `SupabaseWriter.write_signal()` upserts on `(symbol, created_at)` | ✅ |
| 9 | `write_enabled=False` skips network call, logs intent | ✅ |
| 10 | Supabase errors caught, logged, pipeline continues | ✅ |
| 11 | `--publish` flag in `run_analysis.py` (opt-in) | ✅ |
| 12 | `--publish` flag in `run_batch.py` (opt-in) | ✅ |
| 13 | Trade params extracted when `--publish` active; graceful fallback to None | ✅ |
| 14 | Zero changes to pipeline behaviour without `--publish` | ✅ |
| 15 | 59 new tests pass, 0 regressions introduced | ✅ |

---

## Supabase Table Schema

The `signals` table should have these columns (snake_case):

```sql
CREATE TABLE signals (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol              TEXT NOT NULL,
  strategy            TEXT NOT NULL,        -- 'buy' | 'sell' | 'hold'
  alpha_score         NUMERIC(6,4),
  confidence          NUMERIC(6,4),
  confidence_interval JSONB,               -- {lower, upper, level}
  entry_price         NUMERIC(10,2),
  stop_loss           NUMERIC(10,2),
  take_profit         NUMERIC(10,2),
  risk_flags          TEXT[],
  reasoning           TEXT,
  reasoning_path      TEXT[],
  prompt_version      TEXT,
  evidence_ids        JSONB,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at          TIMESTAMPTZ
);
```

Deduplication is handled at the application layer via upsert `on_conflict="symbol,created_at"`. For date-level deduplication via a unique index:

```sql
CREATE UNIQUE INDEX signals_symbol_date_uidx
  ON signals (symbol, DATE(created_at));
```

---

## Known Issues / Future Work

1. **Supabase credentials not yet provisioned** — `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are placeholder comments in `.env`. The pipeline will log a warning and skip the write if `--publish` is used without valid credentials.
2. **`promptVersion` is static** — currently hardcoded to `v1.0.0` in `signal_adapter.py`. Should be bumped and tracked in a config file when prompts change materially.
3. **Date-level deduplication** — the unique index `(symbol, DATE(created_at))` must be created in Supabase for the upsert to correctly replace same-day signals. Without the index, upserts will insert duplicates.
4. **`evidenceIds` analyst report IDs** — currently generated as synthetic strings (`market_report_AAPL_20260303`). Future: link to actual cache file paths or content hashes.
