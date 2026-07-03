# TRI-66 UAT Result — Headless Paper-Smoke Behavioral Acceptance

**Runner:** Independent UAT operator (Cursor session, M3 Max)  
**Script:** `docs/TASK_TRI-66_UAT.md` v3  
**Branch:** `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030`  
**Reviewed code tip:** `135b366`  
**UAT start marker:** `/tmp/tri66_uat_start` (created before first smoke)  
**Run date (UTC):** 2026-07-02T21:57Z – 2026-07-02T22:38Z  
**Wall time:** ~95 min (C1 ~27m, D1 ~19m, G2 ~41m, B1 ~3m + ~22m re-verify)

---

## Commands executed (F2 evidence)

All smoke commands used **`--dry-run`** only. **No `--execute`** anywhere.

```bash
touch /tmp/tri66_uat_start

# B1
pytest --ignore=tests/test_reasoning_comparison.py --ignore=tests/test_prompt_engineering.py --ignore=tests/test_alpaca_connection.py -q

# B2
pytest tests/test_signal_processing.py -q

# C1 — cloud paper smoke
python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run

# D1 — local / Ollama paper smoke
python -m src.run_batch --tickers AAPL --hybrid hybrid_aggressive_qwen --dry-run

# G1 — invalid symbol (observational)
python -m src.run_batch --tickers ZZZZQQ --hybrid hybrid_haiku_tools --dry-run

# G2 — variance re-run (observational)
python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run
```

**F2 paper hardcode:** `grep -n "paper=True" src/execution/executor.py` → present at lines 45 and 49 (`# ← HARDCODED. NEVER CHANGE THIS.`).

---

## Preserved evidence files (§0.5)

| Run | Timestamped copy |
|-----|-------------------|
| C1 | `results/AAPL/analysis_2026-07-02_hybrid_haiku_tools.uat_c1_1783028200.json` |
| D1 | `results/AAPL/analysis_2026-07-02_hybrid_aggressive_qwen.uat_d1_1783029356.json` |
| G2 | `results/AAPL/analysis_2026-07-02_hybrid_haiku_tools.uat_g2_1783031909.json` |

---

## Sign-off table

| Test | Type | Result | Evidence / notes |
|------|------|--------|------------------|
| A1 branch/reviewed code state | Mandatory | **PASS** | Branch `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030`; HEAD `135b366`; `git diff --stat 135b366 -- src vendor pyproject.toml tests config` empty |
| A2 zero-mod | Mandatory | **PASS** | Submodule `85946c2` at tag v0.3.0; porcelain empty; `git diff v0.3.0 \| wc -l` = 0 |
| A3 deps/4 pins | Mandatory | **PASS** | langgraph 1.0.10, langchain-core 1.2.16, langgraph-checkpoint 4.1.1, langgraph-checkpoint-sqlite 3.1.0, yfinance 1.5.1, stockstats 0.6.8 |
| A4 import gate | Mandatory | **PASS** | `python -c "import src.run_analysis"` exit 0 |
| A5 environment | Mandatory | **PASS** | `qwen2.5:14b` in Ollama; ANTHROPIC + Alpaca keys in `.env`; smokes completed without credit-balance 400 |
| B1 suite baseline (8, 0 new) | Mandatory | **PASS** | `8 failed, 610 passed, 2 skipped` — failure set exactly: 5× `test_accuracy_reporter`, 1× `test_admin_scheduler`, 2× `mistral-small:22b` tool-calling |
| B2 signal tests (42) | Mandatory | **PASS** | `42 passed in 0.02s` |
| C1 cloud smoke → valid decision | Mandatory | **PASS** | **BUY** 8.5/10; 27.2m; $0.0000 (100% cache hits); exit 0; `[DRY RUN — no order submitted]` |
| D1 local smoke → valid + offline mem | Mandatory | **PASS** | **HOLD** 5.1/10 (low score recorded, not failed); 19.0m; $0.5723; Ollama `qwen2.5:14b` on reasoning_quick + reasoning_deep; memory.py grep empty |
| E1–E8 extraction cases | Mandatory | **PASS** | SUMMARY: ALL PASS (13/13 cases) |
| E9 documented residual | Observational | **RECORDED** | Returns `SELL` for `**Rating**: Hold\n\nRecommendation: Sell` — accepted per TRI-66 |
| F1 no order placed | Mandatory | **PASS** | No `order_id=` / `Order EXECUTED` in C1/D1/G2 logs; `find audit results/audit … -exec grep EXECUTED` → empty |
| F2 paper/dry-run enforced | Mandatory | **PASS** | All commands above use `--dry-run`; `paper=True` hardcoded in executor |
| G1 invalid-data integrity | Observational (advisory) | **PASS** | `ERROR analysing ZZZZQQ: No market data for 'ZZZZQQ': Yahoo Finance returned no rows`; batch row `ERROR 0.0/10`; no confident BUY/SELL |
| G2 variance re-run | Observational | **PASS** | Run 1 (C1): **BUY** 8.5/10; Run 2 (G2): **BUY** 8.5/10 — both valid; same direction this session (variance still possible on other runs per TRI-78) |
| G3 fail-loud | Observational | **NOT-TESTED** | Skipped (optional); no missing-model probe run |

---

## Smoke run details

### C1 — `hybrid_haiku_tools` / AAPL / 2026-07-02

- **Decision:** BUY (upstream `Overweight` → corrected to BUY)
- **Quality:** 8.5/10 (reasoning 5, data 10, risk 10, consistent Yes)
- **Elapsed:** 1629.2s (~27.2m batch)
- **Cost:** $0.0000 (4/4 analyst cache hits)
- **Routing:** tool_calling=haiku, reasoning_quick/deep=sonnet + ollama qwen on quick path per config
- **Exit:** 0

### D1 — `hybrid_aggressive_qwen` / AAPL / 2026-07-02

- **Decision:** HOLD (upstream `Hold` → HOLD)
- **Quality:** 5.1/10 — **below 8.0 execution gate; recorded for TRI-70, not a UAT failure**
- **Elapsed:** 1138.9s (~19.0m batch)
- **Cost:** $0.5723
- **Routing:** `reasoning_quick` + `reasoning_deep` = `ollama/qwen2.5:14b`
- **Exit:** 0

### G1 — `ZZZZQQ` invalid symbol

- Graceful `NoMarketDataError`-class message; batch completed with ERROR row; exit 0

### G2 — C1 re-run

- **Decision:** BUY 8.5/10; 40.7m wall (cache cold vs C1); exit 0

---

## F1 verification output

```bash
find audit results/audit -type f -name '*.json' -newer /tmp/tri66_uat_start -exec grep -l '"action": "EXECUTED"' {} +
# (no output — PASS)

grep -E "order_id=|Order EXECUTED" /tmp/tri66_c1_smoke.log /tmp/tri66_d1_smoke.log /tmp/tri66_g2_smoke.log
# (no output — PASS)
```

All three smokes printed `[DRY RUN — no order submitted]`.

---

## B1 failure set (exactly 8)

```
FAILED tests/test_accuracy_reporter.py::TestSummary::test_aggregates_by_decision
FAILED tests/test_accuracy_reporter.py::TestSummary::test_best_and_worst_signals
FAILED tests/test_accuracy_reporter.py::TestSummary::test_counts_complete_outcomes
FAILED tests/test_accuracy_reporter.py::TestSummary::test_direction_accuracy_aggregation
FAILED tests/test_accuracy_reporter.py::TestSummary::test_quality_tier_breakdown
FAILED tests/test_admin_scheduler.py::TestSchedulerHistory::test_history_returns_runs
FAILED tests/test_local_tool_calling.py::test_tool_calling_basic[mistral-small:22b]
FAILED tests/test_local_tool_calling.py::test_tool_calling_multi_tool[mistral-small:22b]
```

Summary line: `8 failed, 610 passed, 2 skipped`.

---

## Overall UAT verdict

### **PASS**

Every **Mandatory** test passed. Observational items recorded; E9/G1/G2 do not block. G3 not run.

**UAT does not declare Done.** This result is handed to the **Arbiter** for independent re-verification and final sign-off.

### Carry-forward (out of TRI-66 scope)

- Local-first signal quality on `hybrid_aggressive_qwen` remains below execution gate (TRI-70).
- Run-to-run decision variance is expected (TRI-78); both C1 and G2 returned BUY this session.
- E9 extraction residual accepted pending structured `PortfolioDecision` extraction (TRI-70).
- Five `test_accuracy_reporter` stale-date failures remain (TRI-71).
