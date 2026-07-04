# TASK TRI-69 — Lightweight Edge Check (pre-registered) · DEVELOP Report

> **⚡ LIVE STATUS** (updated at each completed step)
>
> | Step | Status |
> |---|---|
> | 1. Build Config A (temp=0 all three slots) | 🔄 IN PROGRESS — plumbing + tests done, live smoke running |
> | 2. Point-in-time / no-leak mode | 🔄 IN PROGRESS — module + tests done, live leak proof pending |
> | 3. Determinism battery (burned points) | ⬜ pending |
> | 4. Pre-registration commit | ⬜ pending (BEFORE any eval run) |
> | 5. Eval runs | ⬜ pending |
> | 6. Scoring | ⬜ pending |
> | 7. Verdict | ⬜ pending |
>
> Checkpoint triggers fired: **none**

**Issue:** TRI-69 · **Branch:** `jeff/tri-69-edge-check` (from TRI-70 tip 769be7d) · **Spec:** `docs/TASK_TRI-69_EDGE_CHECK.md` (v3, LOCKED) · **Integrity doc:** `docs/TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md` · **Paper only; --execute forbidden.**

---

## Pre-registration

*(To be completed and committed as its own commit BEFORE any eval run — Step 4. Placeholder until then; nothing below this line in this section may change after that commit.)*

---

## Build log

### Step 1 — Config A (TEST-ONLY): `tri69_config_a`

- **Slots (pinned by pre-reg):** tool = `ollama/qwen3-coder:30b` · quick = `ollama/qwen3.5:9b` · deep = `anthropic/claude-sonnet-4-5-20250929` · **temperature = 0.0 on all three.**
- **New plumbing:** `HybridLLMConfig` gained an optional `temperature` field (absent from YAML for legacy configs → byte-identical round-trip); `create_hybrid_llms` now threads it into **all three** `create_llm_client` calls (`src/hybrid_llm.py`). Both vendor client classes already forward `temperature` via `_PASSTHROUGH_KWARGS` (verified: `anthropic_client.py:9-12`, `openai_client.py:166-169` — the Ollama path is `OpenAIClient`, so nothing is silently dropped).
- **DEVELOP-latitude choice:** enhancement flags OFF (`enhance_local=False`, `enhance_deep=False`) — matches the TRI-70 `bench_deep_*` pattern that shares the same tool+quick slots (measure the raw pipeline; no prompt-prefix wrappers). Made before pre-registration commit.
- **Tests:** `tests/test_tri69_point_in_time.py::TestTemperaturePlumbing` — temp=0 reaches all three slot LLM objects; None leaves provider defaults (negative); YAML round-trip. 3/3 pass.
- Builder: `scripts/build_tri69_config.py` (idempotent).

### Step 2 — Point-in-time / no-leak mode (`src/point_in_time.py`, env-gated `TRIFECTA_POINT_IN_TIME=1`)

Vendor tree stays zero-mod; everything is patched at our layer via the vendor's `VENDOR_METHODS` registry + module namespaces.

Leak surfaces found (date-bounding audit of every analyst tool, 2026-07-04) and closures:

| # | Surface | Verdict | Closure |
|---|---|---|---|
| 1 | `get_fundamentals` (yfinance `Ticker.info` + alpha_vantage OVERVIEW) | 🔴 pre-registered leak — 52wk hi/lo, 50/200-day avg, TTM ratios, all today-relative | **Neutralized** → pointer message to date-bounded statement tools |
| 2 | `get_prediction_markets` (Polymarket) | 🔴 pre-registered — live odds, no point-in-time path | **Disabled** (stub message) |
| 3 | `get_insider_transactions` | 🔴 NEW (audit) — no date param at all; returns filings as of real now | **Disabled** (stub message) |
| 4 | Reddit + StockTwits pre-fetch in sentiment analyst (`sentiment_analyst.py:70-71`) | 🔴 live feeds, no date bound (the "live social" of the pre-reg) | **Disabled** — stubs patched into both dataflow modules AND the analyst's import namespace |
| 5 | Statement tools with `curr_date=None` | 🟠 vendor filter silently no-ops | **curr_date forced to as-of date** when omitted |
| 6 | Any date arg > as-of (hallucinated future window) | 🟠 defensive | **Generic clamp** on every vendor-routed method + `build_verified_market_snapshot` |
| — | `resolve_instrument_identity` (`Ticker.info`) | ✅ SAFE — extracts only name/sector/industry/exchange/quote_type; no price/mcap fields | none needed |
| — | get_stock_data / get_indicators / get_verified_market_snapshot / get_news / get_global_news / get_macro_indicators / statements-with-date | ✅ SAFE (date-bounded in vendor, verified with citations) | leak guard still scans outputs |

**Leak guard (fail-loud):** every vendor-routed output is scanned for raw info-keys (`fiftyTwoWeek*`, `fiftyDayAverage`, `twoHundredDayAverage`, `trailing*`); fundamentals-category outputs additionally for formatted labels (`52 Week High:` …). A hit raises `LookAheadLeakError` — the run dies rather than scoring a contaminated decision. News prose ("hit a 52-week high") does NOT false-positive (tested).

**Defense-in-depth:** eval/battery runs save the four analyst reports (`TRIFECTA_SAVE_REPORTS=1`) and the runner re-scans them post-hoc (`_leak_audit` in `scripts/run_tri69_edge_check.py`).

**Memory isolation:** every run gets a fresh `TRADINGAGENTS_MEMORY_LOG_PATH` — no cross-run reflection/decision contamination, no run-order dependence.

- **Tests:** 8/8 PIT tests pass (neutralization, disables, curr_date forcing, date clamp, leak-guard rejection [negative test], prose non-false-positive, analyst-namespace patch, idempotency).
- Live leak proof (PIT off → fields present; PIT on → absent): recorded below.

### Live leak proof (2026-07-04)

Same live vendor call (`route_to_vendor("get_fundamentals", "MSFT", "2026-03-13")`), PIT off vs on:

- **PIT OFF** — output contains ALL six forbidden formatted labels, with values that are unambiguously today-relative for a 2026-03-13 decision date: `PE Ratio (TTM): 23.257294`, `EPS (TTM): 16.79`, `52 Week High: 555.45`, `52 Week Low: 349.2`, `50 Day Average: 407.5954`, `200 Day Average: 445.4361`. The leak is real; the guard tests are not tautological.
- **PIT ON** — same call returns the neutralization message; zero forbidden info-keys, zero forbidden labels. `get_prediction_markets` and `get_insider_transactions` return their disabled-mode stubs.

### Step 3 — Determinism battery

*(pending)*

---

## Results

*(pending — nothing here until after the pre-registration commit)*

## Compute / $ spent

*(running total; see result JSONs)*

## Verdict

*(pending)*
