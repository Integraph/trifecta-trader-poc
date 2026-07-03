# TRI-66 — UAT Test Script (headless paper-smoke behavioral acceptance) · v3

**Stage:** UAT (independent runtime acceptance) · **Repo:** `trifecta-trader-poc` · **Branch:** `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030` · **Reviewed code tip:** **`135b366`** (see A1 — later *docs-only* commits are fine)
**Runner:** an **independent** operator/Cursor session on the **M3 Max** — *not* the DEVELOP or QA agent. **Record results in** `docs/TASK_TRI-66_UAT_RESULT.md`, then hand to the Arbiter. **You do not declare Done.**
**Scope:** the **engine only**, and it is **headless** — CLI behavioral acceptance, **no UI**. (Admin-UI behavior is TRI-75, separate; it does not gate this.)

> **v3 (after DEVELOP + QA review of v2):** 🔴 fixed the **F1 safety check** — it greps the *actual* audit path `results/audit/` (was `audit/` only, which would have passed vacuously even if a real order fired — `run_analysis.py:551/568`) and is now **time-scoped** to the UAT window. Also: A1 now allows committing this doc (pins the *reviewed code state*, not literal HEAD); B1 calls out dynamic Ollama-model parametrization as noise; F2 requires the exact commands be pasted into the result doc; grep checks judge by **empty output, not exit code**; G1 downgraded to advisory; E2c label fixed.

---

## 0. Read before running — acceptance principles

1. **The engine is non-deterministic (TRI-78).** Same ticker/day/config can return BUY on one run and SELL on the next. **PASS = a _valid_ decision (`BUY`/`HOLD`/`SELL`, not `UNKNOWN`) with the pipeline completing cleanly. NEVER fail a test because the decision differs from a prior run or from this script's example.**
2. **"Runs correctly" ≠ "signal is good."** UAT accepts the engine *functions* on v0.3.0. Signal quality/profitability is **out of scope** (TRI-69/70). A low quality score (~3.6–5.1/10, below the 8.0 execution gate) is **not** a UAT failure.
3. **Paper only. `--execute` is FORBIDDEN.** Every run uses `--dry-run`. Any real/live order = **immediate FAIL — stop the UAT** and escalate.
4. **Start marker (for F1):** run `touch /tmp/tri66_uat_start` **before your first smoke** — F1 uses it to scope the order check to this UAT run.
5. **Preserve evidence (TRI-79).** Result files (`results/<ticker>/analysis_<date>_<config>.json`) overwrite on re-run. **Immediately after every run (including the last), copy its result JSON to a timestamped name** — e.g. `cp results/AAPL/analysis_<date>_<config>.json results/AAPL/analysis_<date>_<config>.uat_$(date +%s).json`. (Copy *after* each run, not just before the next — removes the "forgot before G2" trap.)
6. **Prereqs:** Ollama running with `qwen2.5:14b` pulled; `.env` has `ANTHROPIC_API_KEY` + Alpaca **paper** keys; Anthropic paper **credits available**; the repo's venv active.
7. **grep-based checks judge by _empty output_, not exit code** — a clean `grep` with no match exits non-zero; that is a PASS here. Don't wire these with `&&`/`$?` logic.
8. Each test is **MANDATORY** (gates UAT) or **OBSERVATIONAL** (record; does not block). Overall UAT = **PASS** only if every **Mandatory** test passes.
9. **Budget & patience.** C1 + D1 + the G2 re-run are ~3 full pipeline runs ≈ **50–75 min** wall time and ≈ **$1.20–2.10** Anthropic spend; B1 adds ~2–8 min. Local (Ollama) runs are slow (~16–24 min) — **do not interrupt a prolonged Ollama wait** (that caused a false-negative in an earlier round).

---

## Section A — Preconditions (MANDATORY)

### A1 — Correct branch & reviewed code state
- **Steps:** `git rev-parse --abbrev-ref HEAD` ; then confirm **no code changed since the reviewed tip `135b366`**:
  `git diff --stat 135b366 -- src vendor pyproject.toml tests config`  *(must be empty)*
- **Expected:** branch `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030`; the diff is **empty**. (HEAD may be `135b366` **or** a later **docs-only** commit designated by the Arbiter — committing this UAT script itself is fine, since it doesn't touch `src/ vendor/ pyproject.toml tests/ config/`.)
- **PASS:** on the branch **and** zero diff vs `135b366` across those paths. **FAIL:** wrong branch, or **any code/vendor/deps/test change since `135b366`** — unreviewed code must not enter UAT; void the run and escalate.

### A2 — Vendor at v0.3.0, zero-mod
- **Steps:** `git submodule status vendor/TradingAgents` ; `git -C vendor/TradingAgents status --porcelain` ; `git -C vendor/TradingAgents diff v0.3.0 | wc -l`
- **Expected:** gitlink `85946c2f…` at tag `v0.3.0`; porcelain **empty**; diff line count **0**.
- **PASS:** submodule at `85946c2` **and** tree clean **and** diff-vs-tag = 0. **FAIL:** any modification vs the v0.3.0 tag.

### A3 — Required deps installed at the pinned versions
- **Steps:**
  ```
  python -c "import langgraph.checkpoint.sqlite, stockstats, yfinance; import importlib.metadata as m; \
  print('langgraph', m.version('langgraph')); print('langchain-core', m.version('langchain-core')); \
  print('langgraph-checkpoint', m.version('langgraph-checkpoint')); print('langgraph-checkpoint-sqlite', m.version('langgraph-checkpoint-sqlite')); \
  print('yfinance', m.version('yfinance')); print('stockstats', m.version('stockstats'))"
  ```
  *(`langgraph` has no `__version__` attribute — use `importlib.metadata.version()`.)*
- **Expected:** no ImportError; `langgraph==1.0.10`, `langchain-core==1.2.16`, `langgraph-checkpoint==4.1.1`, `langgraph-checkpoint-sqlite==3.1.0`, `yfinance>=1.4.1`, `stockstats>=0.6.5`.
- **PASS:** imports succeed and **all four langgraph-stack pins match exactly** + floors met. **FAIL:** any ImportError or any pinned version off.

### A4 — Import gate
- **Steps:** `python -c "import src.run_analysis"` ; `echo $?`
- **Expected:** exit `0`, no traceback.
- **PASS:** exit 0. **FAIL:** any error. *(`import src.hybrid_graph` failing standalone is EXPECTED — TRI-72, out of scope.)*

### A5 — Environment ready
- **Steps:** `ollama list | grep qwen2.5:14b` ; confirm `.env` keys; confirm Anthropic credits (first smoke starts without `400 credit balance too low`).
- **PASS:** model present, keys present, credits available. **FAIL:** any missing.

---

## Section B — Test suite baseline (MANDATORY)

### B1 — Full suite: exactly the known 8 failures, zero new
- **Steps (use `--ignore`, the form the baseline was measured with):**
  ```
  pytest --ignore=tests/test_reasoning_comparison.py --ignore=tests/test_prompt_engineering.py --ignore=tests/test_alpaca_connection.py -q
  ```
- **Expected:** `8 failed, 610 passed, 2 skipped`. The 8 failures are **only**: 5× `test_accuracy_reporter` (stale-date, TRI-71), 1× `test_admin_scheduler` (port-8420 env), 2× `mistral-small:22b` tool-calling.
- **PASS:** exactly those 8 fail and **no new failure**. **FAIL:** any failure outside the known 8, or a previously-passing test now failing.
- **Judge only the _failure set_, not count noise.** `tests/test_local_tool_calling.py` **parametrizes over whatever Ollama models are installed**, so on a machine with extra local models the passed/failed/skip counts can grow. A new failure counts **only if** it involves the required `qwen2.5:14b` or the known-baseline `mistral-small:22b`; **extra-model parametrizations are environment noise**, as is the `deselected` count if you use `--deselect`.

### B2 — Decision-extraction unit tests
- **Steps:** `pytest tests/test_signal_processing.py -q`
- **Expected:** `42 passed` (pinned to `135b366`).
- **PASS:** 42/42. **FAIL:** any failure.

---

## Section C — Cloud paper smoke (exit-5) (MANDATORY)

### C1 — `hybrid_haiku_tools` produces a valid decision
- **Steps:** `python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run` → copy the result JSON aside (§0.5).
- **PASS (all must hold):**
  1. Decision ∈ {`BUY`,`HOLD`,`SELL`} — **not** `UNKNOWN` (do **not** require a specific direction).
  2. Pipeline reached the **Portfolio Manager** (no `KeyError`/routing error).
  3. A numeric quality score is present.
  4. Exit 0, no unhandled exception. *(Reddit 429s / Polymarket / FRED optional-source notes are handled fallbacks, not failures.)*
  5. **No order placed** — verify per **F1**.
- **FAIL:** `UNKNOWN`; error before the Portfolio Manager; no quality score; any order placed; non-zero exit.
- **Record:** decision, quality, cost, elapsed (TRI-78 variance log).

---

## Section D — Local / Ollama paper smoke (exit-6) (MANDATORY)

### D1 — `hybrid_aggressive_qwen` produces a valid decision on the local path
- **Steps:** `python -m src.run_batch --tickers AAPL --hybrid hybrid_aggressive_qwen --dry-run` → copy the result JSON aside.
- **PASS (all must hold):**
  1. Decision ∈ {`BUY`,`HOLD`,`SELL`} — not `UNKNOWN`.
  2. Ollama/local path exercised (local `qwen2.5:14b` handled reasoning; no routing crash).
  3. **Offline memory — concrete probe (judge by empty output):**
     `grep -niE "openai|embed|chromadb|faiss|requests\.(get|post)" vendor/TradingAgents/tradingagents/agents/utils/memory.py` → **no output** (v0.3.0's memory is a file journal). Logs are supporting evidence only.
  4. Exit 0; no order placed (per F1).
- **Explicitly NOT a fail:** a **low quality score** (~3.6–5.1/10). That's the local-first *viability* question (TRI-70) — **record, don't fail.**
- **FAIL:** `UNKNOWN`; Ollama routing crash; the memory grep finds an embeddings/API dependency; any order placed; non-zero exit.
- **Record:** decision, quality, cost, elapsed.

---

## Section E — Decision-extraction edge cases (the fixed seam)

Deterministic probes against `extract_decision` (fast, no pipeline). Paste:

```
python - <<'PY'
from src.signal_processing import extract_decision as d
cases = {
 "E1a **Recommendation**: Overweight -> BUY":       ("**Recommendation**: Overweight", "BUY"),
 "E1b **Rating**: Underweight -> SELL":             ("**Rating**: Underweight", "SELL"),
 "E1c **Recommendation**: Hold -> HOLD":            ("**Recommendation**: Hold", "HOLD"),
 "E1d out-of-vocab 'Neutral' -> UNKNOWN":           ("**Rating**: Neutral", "UNKNOWN"),
 "E2a **Action**: Underweight (qwen) -> SELL":      ("**Action**: Underweight", "SELL"),
 "E2b **Final Transaction Proposal: Buy** -> BUY":  ("**Final Transaction Proposal: Buy**", "BUY"),
 "E2c **Investment Recommendation: Sell** -> SELL": ("**Investment Recommendation: Sell**", "SELL"),
 "E3 legacy FINAL TRANSACTION PROPOSAL: HOLD":      ("FINAL TRANSACTION PROPOSAL: HOLD", "HOLD"),
 "E4 prose-safety (no header) -> UNKNOWN":          ("We think an underweighted position looks attractive here.", "UNKNOWN"),
 "E5 blockquote stale after real -> HOLD":          ("**Rating**: Hold\n\n> Recommendation: Underweight", "HOLD"),
 "E6 fenced stale after real -> BUY":               ("**Rating**: Buy\n\n```\nRecommendation: Sell\n```", "BUY"),
 "E7 last real decision-line wins -> SELL":         ("**Rating**: Buy\n... more analysis ...\n**Recommendation**: Sell", "SELL"),
 "E8 no decision at all -> UNKNOWN":                ("The market is uncertain; no clear call.", "UNKNOWN"),
}
fails=0
for name,(text,exp) in cases.items():
    got=d(text); ok = "PASS" if got==exp else "FAIL"
    if got!=exp: fails+=1
    print(f"[{ok}] {name}: got={got} expected={exp}")
print("SUMMARY:", "ALL PASS" if fails==0 else f"{fails} FAIL")
PY
```

- **E1–E8 — PASS:** every case prints `[PASS]` (SUMMARY: ALL PASS). **FAIL:** any `[FAIL]`.
  - E1 = the v0.3.0 **5-tier** rating (Buy/Overweight/Hold/Underweight/Sell) → 3-level map (Overweight→BUY, Underweight→SELL). **`Neutral` is NOT in v0.3.0's `PortfolioRating`** → correct behavior is loud `UNKNOWN`. *(A `Neutral`→HOLD mapping would be a code change via DEVELOP→QA — TRI-70/TRI-74 — not a UAT assertion.)*
  - E2 = local-model label variants. E3 = legacy format. E4 = prose does not false-match.
  - **E5/E6 = the QA round-2 fixes** — a quoted/fenced stale decision must NOT override the real one.
  - E7 = last genuine decision-line wins. E8 = no decision → loud `UNKNOWN`.

### E9 — Documented residual (OBSERVATIONAL — record, do NOT fail)
- **Steps:** `python -c "from src.signal_processing import extract_decision as d; print(d('**Rating**: Hold\n\nRecommendation: Sell'))"`
- **Known behavior:** returns `SELL` — the residual a pure regex can't disambiguate. **Accepted for TRI-66**; durable fix = structured `PortfolioDecision` extraction in **TRI-70**. **Record; do NOT fail on E9.**

---

## Section F — Safety (MANDATORY) — concrete verification

### F1 — No order placed on any dry-run
- **Steps (judge by _empty output_):**
  1. CLI output shows dry-run per-ticker lines with **no `order_id=`** and **no `Order EXECUTED for …`** line.
  2. **No executed-order audit record written during this UAT run** — check **both** audit locations (the pipeline writes to `results/audit/`; `audit/` is the executor default), scoped to since the start marker:
     ```
     find audit results/audit -type f -name '*.json' -newer /tmp/tri66_uat_start -exec grep -l '"action": "EXECUTED"' {} +
     ```
     → returns **nothing**. *(A real fill would tag `"action": "EXECUTED"` — `src/execution/executor.py:91`. A dry-run must produce none. `results/audit/` is the real path via `run_analysis.py:551/568`.)*
- **PASS:** no `order_id=` in output **and** the `find | grep` prints nothing. **FAIL:** any `EXECUTED` audit entry or order id → **immediate overall FAIL**.

### F2 — Paper / `--dry-run` enforced
- **Steps:**
  1. **Paste the exact smoke commands you ran into the result doc** (`docs/TASK_TRI-66_UAT_RESULT.md`) — every one must contain `--dry-run` and none `--execute`. (Shell history is not sufficient evidence.)
  2. Confirm the executor is still hard-pinned to paper: `grep -n "paper=True" src/execution/executor.py` → present at ~line 49 (`# ← HARDCODED. NEVER CHANGE THIS.`).
- **PASS:** all pasted commands are `--dry-run` **and** `paper=True` is still hardcoded (live trading deferred — TRI-31). **FAIL:** any `--execute`, or the paper hardcode removed/altered.

---

## Section G — Robustness / edge behavior

### G1 — Invalid/sparse market data does not yield a confident signal (OBSERVATIONAL — advisory; Arbiter may promote to mandatory)
- **Rationale:** measurement integrity — v0.3.0 adds stale-OHLCV rejection / a verified data contract (`NoMarketDataError`-class; the TRI-57 class). But invalid-symbol handling is broader than the vendor-upgrade acceptance surface, so this is **advisory for TRI-66** unless the Arbiter opts to gate on it.
- **Steps:** run an obviously invalid symbol, `--dry-run`: `python -m src.run_batch --tickers ZZZZQQ --hybrid hybrid_haiku_tools --dry-run`
- **PASS:** the engine surfaces the data problem — an `ERROR analysing ZZZZQQ: NoMarketDataError…`-class row / graceful error, or a `UNKNOWN`/low-confidence result — and does **not** emit a confident directional decision. *(An **ERROR row on the invalid symbol is the graceful/expected path** — do not misread it as a pipeline failure.)* **FAIL (if promoted):** a confident BUY/SELL on the invalid symbol.
- **Record** the outcome. If not run, mark **NOT-TESTED**.

### G2 — Variance observation (OBSERVATIONAL — do NOT fail on a different decision)
- **Steps:** re-run C1 (`hybrid_haiku_tools`) a second time (evidence already copied per §0.5).
- **PASS:** completes and yields a valid decision. A **different** decision than run 1 is **expected, not a failure** — **record both** (TRI-78). **FAIL:** the run errors or returns `UNKNOWN`.

### G3 — Missing-model / missing-dep fails loud (OBSERVATIONAL)
- **Steps (optional):** point a config at an un-pulled Ollama model, `--dry-run`.
- **PASS:** a clear, loud error (not a silent `UNKNOWN`/hang). **FAIL:** silent/misleading failure. *(Not-tested if skipped.)*

---

## Sign-off — record in `docs/TASK_TRI-66_UAT_RESULT.md`

Paste the **exact commands run** (for F2) and the **fresh evidence file names** (§0.5), then:

| Test | Type | Result (PASS/FAIL/NOT-TESTED) | Evidence / notes |
|---|---|---|---|
| A1 branch/reviewed code state | Mandatory | | diff-vs-135b366 empty? |
| A2 zero-mod | Mandatory | | |
| A3 deps/4 pins | Mandatory | | versions printed |
| A4 import gate | Mandatory | | |
| A5 environment | Mandatory | | |
| B1 suite baseline (8, 0 new) | Mandatory | | failure set only |
| B2 signal tests (42) | Mandatory | | |
| C1 cloud smoke → valid decision | Mandatory | | decision/quality/cost |
| D1 local smoke → valid + offline mem | Mandatory | | decision/quality/cost |
| E1–E8 extraction cases | Mandatory | | ALL PASS? |
| E9 documented residual | Observational | | record behavior |
| F1 no order placed | Mandatory | | find\|grep output |
| F2 paper/dry-run enforced | Mandatory | | commands pasted |
| G1 invalid-data integrity | Observational (advisory) | | |
| G2 variance re-run | Observational | | both decisions |
| G3 fail-loud | Observational | | |

**Overall UAT verdict:** **PASS** only if every **Mandatory** row is PASS (E9/G1/G2/G3 do not block). Otherwise **FAIL** with specifics.

**UAT does not declare Done.** On PASS → hand this result to the **Arbiter** for independent re-verification and the Done sign-off. Carry forward: local signal below the 8.0 execution gate (TRI-70), run-to-run variance is real (TRI-78), and the extraction residual (E9) is accepted pending structured extraction (TRI-70).
