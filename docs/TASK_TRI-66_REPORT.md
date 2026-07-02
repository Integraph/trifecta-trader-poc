# TASK TRI-66 — DEVELOP Report: Vendored TradingAgents v0.2.0 → v0.3.0

**Stage:** DEVELOP (Claude Code) → QA (Codex) → UAT (paper-smoke)
**Branch:** `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030` (from `main`)
**Work order:** `docs/TASK_TRI-66_ENGINE_UPGRADE.md` (v7)
**Date:** 2026-07-01 · **Status:** DEVELOP complete pending final smoke/test confirmation (see §8–9). **I do not declare Done.**

> Paper/`--dry-run` only; `--execute` never used. Goal = restore true zero-mod. Reported against the tree — trust no line here without re-running it.

---

## 1. Summary

Upgraded the vendored `TradingAgents` submodule from our pinned **v0.2.0 + Task-020 commit (`5de91bc`)** to upstream **v0.3.0 (peeled `85946c2`)**, restoring **true zero-mod** (our 4 local shims are discarded by the tag checkout — no vendor edits remain). Added the three dependencies v0.3.0 requires, pinned the LangGraph stack to the tested resolve, and reworked the engine's vendor-facing code (`src/hybrid_graph.py`) for v0.3.0's refactored graph-construction API.

**The rework was substantive, not a rename** (as the work order anticipated). v0.3.0 (a) removed per-agent memory from every agent factory (centralized into a file-based `TradingMemoryLog`), and (b) made analyst graph nodes **spec-driven**, renaming social→**Sentiment** and the final judge node→**Portfolio Manager**. Both required matching changes in our hand-mirrored `HybridGraphSetup`.

---

## 2. Per-file reconciliation

### 2.1 Vendor submodule — zero-mod restored
- Checked out the v0.3.0 peeled commit `85946c2f60768ab2dae23a5a36cd927662feef94`; the tag tree does **not** contain our `5de91bc` commit, so all **4 Task-020 lazy-import shims are discarded wholesale** (no hand-reverting).
- Verified: `git -C vendor/TradingAgents status` → clean; `git -C vendor/TradingAgents diff 85946c2` → empty; `vendor/TradingAgents/pyproject.toml` → `version = "0.3.0"`.
- Gitlink updated `5de91bc → 85946c2`; `.gitmodules` unchanged. **Zero-mod ✅**

### 2.2 The 4 shims (confirmed then dropped)
`git show 5de91bc` confirmed all four are try/except optional-dep lazy-import shims (rank_bm25 / stockstats / stockstats_utils / google_client) — not logic changes. v0.3.0 supersedes 2 (memory.py dropped rank_bm25; factory.py made provider imports lazy) and re-eager-imports stockstats in the other 2 — resolved by installing the dep, not re-patching.

### 2.3 `pyproject.toml` (dependency additions — required for v0.3.0 to import)
| Change | Reason |
|---|---|
| `langgraph-checkpoint-sqlite==3.1.0` (**added**) | v0.3.0 imports `langgraph.checkpoint.sqlite` at **module level** (`graph/checkpointer.py:14`) → hard-blocks `import src.run_analysis` if absent. Not previously declared. |
| `yfinance>=1.4.1` (floor **raised** from bare) | v0.3.0 rewrote `dataflows/y_finance.py` against the 1.4.x API; venv was at 1.2.0. Resolved to **1.5.1**. |
| `stockstats>=0.6.5` (floor set) | eager-imported in v0.3.0 `dataflows/*`. Resolved to **0.6.8**. |
| `langgraph==1.0.10`, `langchain-core==1.2.16`, `langgraph-checkpoint==4.1.1` (**pinned**) | pin the working LangGraph 1.x stack; no downgrade. |

### 2.4 `src/hybrid_graph.py` (the vendor-facing rework)
The 3 vendor imports are `run_analysis.py:27-28`, `hybrid_graph.py:22-25`, `hybrid_llm.py:378`. Only `hybrid_graph.py` needed changes (the others' symbols — `TradingAgentsGraph`, `GraphSetup`, `ConditionalLogic`, `DEFAULT_CONFIG`, `create_llm_client` — are unchanged in v0.3.0; `TradingAgentsGraph.__init__(selected_analysts, debug, config, callbacks)` matches our `super().__init__`).

Changes in `HybridGraphSetup` / `HybridTradingGraph`:
1. **Risk Judge factory:** `create_risk_manager(llm, memory)` → `create_portfolio_manager(llm)` (removed in v0.3.0; signature dropped `memory`).
2. **All agent factories lost their `memory` arg** in v0.3.0 — dropped it from `create_bull_researcher/bear_researcher/trader/research_manager` too.
3. **Removed the orphaned memory plumbing** — the 5 `*_memory` params + `self.*` assignments in `HybridGraphSetup.__init__`, and the 5 memory kwargs passed in `HybridTradingGraph._rebuild_graph` (v0.3.0's `TradingAgentsGraph` no longer creates `self.bull_memory` etc.; it creates one `self.memory_log`).
4. **Spec-driven node naming:** replaced the naive `f"{key.capitalize()} Analyst"` pattern with `build_analyst_execution_plan(selected_analysts)` + `spec.agent_node/clear_node/tool_node`. Required because v0.3.0 renamed social→**"Sentiment Analyst"** / **"Msg Clear Sentiment"** and `conditional_logic.should_continue_social()` returns those labels. **This was the cause of the first smoke's `KeyError 'Msg Clear Sentiment'`.**
5. **Final node "Risk Judge" → "Portfolio Manager"** (matches `should_continue_risk_analysis`'s return value + the node v0.3.0 registers).

Diff size: `src/hybrid_graph.py` 114 lines (+62 / −67 — net reduction from removing the memory plumbing). Topology otherwise unchanged (same edges, same `ConditionalLogic` methods/signature).

### 2.5 `src/signal_processing.py` (decision-vocabulary mapping — folded in per Jeff, 2026-07-01)
v0.3.0's Portfolio Manager emits a **5-level `PortfolioDecision` rating** (`Buy / Overweight / Hold / Underweight / Sell`), replacing the old Risk Manager's 3-level `FINAL TRANSACTION PROPOSAL: BUY/SELL/HOLD`. `extract_decision()` only recognized BUY/HOLD/SELL, so smokes returned `decision=UNKNOWN` for valid `Overweight`/`Underweight` analyses. Added a highest-priority, **header-anchored** mapping method: **Overweight→BUY, Underweight→SELL** (Buy/Hold/Sell unchanged). Reasoning-prose mentions of the rating words are **not** matched (anchored to the header label, so e.g. "an underweighted position" → UNKNOWN, not SELL).

**Different LLMs label the decision line differently** — this only became visible by running both smokes: cloud Haiku/Sonnet write `**Rating**: Overweight`; the structured render uses `**Recommendation**:`; local `qwen2.5:14b` writes `**Action**: Underweight` and `**Final Transaction Proposal: Underweight**`. The matcher accepts all of these labels. **Surfaced and approved before committing** (standard financial mapping collapsing v0.3.0's 5-level scale onto the engine's 3-level decision).

**Codex QA P1 (CHANGES-REQUESTED, 2026-07-01) — accepted and fixed.** As committed in `c035adc`, this report claimed the matcher was "header-anchored"; **that claim was false**. The regex had no line anchor and did not require the colon, and because method 0 takes the last match, label-shaped *prose* (e.g. `Rationale: prior note wrote Rating: Underweight.`) could override a real `**Rating**: Hold` header → SELL. All four of QA's repro cases reproduced. My original prose-safety tests only covered rating words *without* label words — exactly the missed failure mode.

**Fix (final form):** the label must start a line (markdown heading/list/bold prefixes allowed: `^[\s#>*-]{0,12}` + MULTILINE), the colon separator is **mandatory**, the core label is one of `Recommendation / Rating / Action / Transaction Proposal` with at most one qualifier from a tight allowlist (`Investment / Final / Overall / Portfolio / Trading / Recommended` — added after a post-fix local run emitted `**Investment Recommendation: Underweight**`), and the last *decision line* wins for looping output. All 4 QA repro cases pass, plus adversarial tests for mid-line bold labels, looping headers, markdown decorations, qualified labels, and unlisted-qualifier prose — **37/37 signal tests pass**, and all four preserved real outputs (two cloud, two local) extract correctly. Both smokes were **re-run post-fix and recorded valid decisions live** (§9), per QA's P2.

**Documented residual ceiling:** a prose line that itself *begins* with `<Label>: <rating>` still matches — regex scraping cannot fully disambiguate that, and the local model **changes its label format run to run**, so the allowlist can only chase. The durable fix is structured `PortfolioDecision` extraction, owned by **TRI-70**. (Methods 1–3, the pre-TRI-66 legacy patterns, retain their historical unanchored behavior — unchanged by this task and likewise superseded by TRI-70.)

Files changed: `pyproject.toml`, `src/hybrid_graph.py`, `src/signal_processing.py`, `tests/test_signal_processing.py` (+4 mapping tests), `tests/test_pipeline.py` (2 assertions updated for the "Risk Judge"→"Portfolio Manager" vendor rename), and the `vendor/TradingAgents` gitlink — no other source touched.

---

## 3. 🔴 Surfaced behavior change (per guardrail — do not paper over)

**v0.3.0 replaced per-agent memory with a centralized file-based journal.** In v0.2.0 each reasoning agent (bull/bear researchers, trader, research manager, risk manager) received its own `FinancialSituationMemory` (BM25) object at construction. v0.3.0 removed that entirely; memory is now one `TradingMemoryLog` (a plain file-based decision journal — **no embeddings, no API**), which the vendor's `TradingAgentsGraph.propagate()` reads and injects into state as `past_context` for the Portfolio Manager (the Risk Judge).

**Effect on our hybrid path:** preserved, not lost — our `HybridTradingGraph.propagate()` **delegates to `super().propagate()`**, so it inherits v0.3.0's `memory_log` handling; and our graph uses v0.3.0's `create_portfolio_manager` + `AgentState`, so the Portfolio Manager still receives memory context (via state, not construction). This is **v0.3.0's design, applied to the vendor's own graph too — not a divergence introduced here.** Net: the Risk Judge still gets memory; the per-agent memory the *other* reasoning agents used to get is gone for everyone under v0.3.0.

**For the App Manager / TRI-70:** the memory *mechanism* changed, which could shift Risk-Judge behavior/quality vs. the old benchmark. TRI-66 does not re-score quality (that's TRI-70); flagging so the benchmark accounts for it.

**Second behavior change — decision vocabulary (5-level → 3-level).** v0.3.0's Portfolio Manager emits `Buy/Overweight/Hold/Underweight/Sell` instead of `BUY/SELL/HOLD`. Surfaced and approved before committing; mapped Overweight→BUY, Underweight→SELL in `signal_processing.py` (§2.5). The engine now produces valid decisions on v0.3.0's output (verified: the recorded `Overweight` analysis maps to `BUY`).

Other v0.3.0 behavior notes observed: advisory `RuntimeWarning`s that `claude-sonnet-4-5`/`claude-haiku-4-5` are "not in the known model list" (legacy-but-active; validation is advisory — model-string currency is TRI-70/73, out of scope here). External Reddit RSS `429`s and a Polymarket brotli-decode error during the sentiment analyst (vendor backs off / degrades gracefully; not upgrade-related). One `Trader: structured-output invocation failed; retrying once as free text` (vendor graceful fallback — the PM likewise rendered free-text `**Rating**:`, which §2.5 handles).

---

## 4. Dependencies — before → after

| Package | Before | After (pinned/resolved) |
|---|---|---|
| langgraph | 1.0.10 (unpinned) | **1.0.10 (held, pinned)** |
| langchain-core | 1.2.16 (unpinned) | **1.2.16 (pinned)** |
| langgraph-checkpoint | 4.0.1 (transitive) | **4.1.1 (pinned)** |
| langgraph-checkpoint-sqlite | absent | **3.1.0 (added, pinned)** |
| yfinance | 1.2.0 | **1.5.1** (floor `>=1.4.1`) |
| stockstats | absent (declared, uninstalled) | **0.6.8** (floor `>=0.6.5`) |

`pip install -e .` resolved cleanly (no conflicts; langgraph held at 1.0.10, no downgrade). It also drifted ~52 other packages (unpinned transitive re-resolve: streamlit 1.58, redis 8.0.1, langchain-community 0.4.1, google-genai, etc.) — **full lockfile is TRI-65**, out of scope here. `pip freeze` before/after saved.

---

## 5. Import gate (exit-3)
`python -c "import src.run_analysis"` → **OK** (deps present, shims gone). Confirmed the import gate is exactly `stockstats` + `langgraph-checkpoint-sqlite` (scanned every module-level third-party import in the v0.3.0 tag; `langchain_google_genai` is lazy/google-only and declared).

## 6. LangGraph stack pins (exit-7)
`langgraph 1.0.10` / `langchain-core 1.2.16` / `langgraph-checkpoint 4.1.1` / `langgraph-checkpoint-sqlite 3.1.0` — installed matches pinned; `langgraph` **not** downgraded or floated. (`langgraph-prebuilt` / `langgraph-sdk` are transitive → TRI-65.)

## 7. Offline-memory check (Step 7 / exit-6)
v0.3.0 `agents/utils/memory.py` (`TradingMemoryLog`) has **no** embeddings/OpenAI/API/requests/chromadb references — it's a file-based journal. No new offline-path dependency. ✅

---

## 8. Test baseline vs post-upgrade (exit-4)

**Baseline (before upgrade, main tree):** `8 failed, 589 passed, 2 skipped` — exactly the documented TRI-34 set:
- 5× `test_accuracy_reporter.py::TestSummary` (stale-date time-bomb → TRI-71)
- 1× `test_admin_scheduler.py::test_history_returns_runs` (port-8420 in use → env)
- 2× `test_local_tool_calling.py::[mistral-small:22b]` (local tool-calling)

(Run with `test_reasoning_comparison.py`, `test_prompt_engineering.py`, `test_alpaca_connection.py` excluded — these make real paid/live-broker calls and are redundant with the paper smokes; the **same exclusion set** is used post-upgrade for an apples-to-apples delta.)

**Post-upgrade (raw):** `10 failed, 591 passed, 2 skipped` — the same 8 baseline failures **plus 2 new fallout failures** in `test_pipeline.py` (`test_conditional_logic_risk_round_limit`, `test_conditional_logic_config_not_passed`), both `assert ... == 'Risk Judge'`. These assert the **old vendor node name**; v0.3.0 renamed the final risk node "Risk Judge" → "Portfolio Manager" (same rename adapted in `hybrid_graph.py`). **Fixed** by updating the two assertions to `"Portfolio Manager"` (`test_pipeline.py` now 5/5). No other test references the old node name (remaining "Risk Judge" hits are descriptive docstrings).

**Post-upgrade (after fallout fix):** confirmation re-run → **`8 failed, 593 passed, 2 skipped`** — **exactly the 8 baseline failures, ZERO new** (`test_conditional_logic_*` now pass). Net vs baseline: **0 new failures (exit-4 ✅)**, and **+4 new passing** signal_processing mapping tests (593 vs 589 passed).

---

## 9. Paper smokes (exit-5 / exit-6)

- **Cloud (`hybrid_haiku_tools`) — ✅ PASSED (re-run after the QA P1 fix, on the anchored matcher).**
  `python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run`
  → **`DECISION: SELL`** (PM emitted `Underweight` → SELL), **quality 8.5/10**, **cost $0.074**, target $273, elapsed ~967s; dry-run completed (no order), no errors. Full pipeline on v0.3.0 (4 analysts → 2 researchers → research manager → trader → 3 risk debators → Portfolio Manager); valid 3-level decision (not UNKNOWN).
  _Earlier DEVELOP runs of this config: `Overweight`→BUY (8.5/10, $0.080, ~1001s) and `Overweight` (8.2/10, $0.027, first run). Same config/ticker/day produced `Overweight` and `Underweight` across runs — see §11 run-to-run variance note._
- **Local/Ollama (`hybrid_aggressive_qwen`, `qwen2.5:14b`) — ✅ PASSED (re-run after the QA P1 fix, on the anchored matcher).**
  `python -m src.run_batch --tickers AAPL --hybrid hybrid_aggressive_qwen --dry-run`
  → **`DECISION: SELL`** (PM emitted `Underweight` → SELL), **quality 3.6/10**, **cost $1.036**, elapsed ~1432s; dry-run completed (no order), no errors. Proves the **local/Ollama routing survived the upgrade** and agent memory works offline (file journal, no embeddings API).
  - _Local-format history (why three runs): run 1 → `UNKNOWN` (qwen labeled the decision `**Action**:` / `**Final Transaction Proposal:**`; matcher broadened). Run 2 (post-anchoring) → `UNKNOWN` again — **the same model changed its label to `**Investment Recommendation: Underweight**`**; matcher given a tight qualifier allowlist. Run 3 → valid `SELL` recorded live. **The label varies per RUN, not just per model** — regex can chase this but never close it; structured `PortfolioDecision` extraction (TRI-70) is the durable fix._
  - _Local quality varied 5.1 → 3.6 across runs (cost $0.49–$1.04) — run-to-run variance data for TRI-70's benchmark design._

---

## 10. Exit-criteria status

| # | Criterion | Status |
|---|---|---|
| 1 | Submodule at v0.3.0 `85946c2`, gitlink/`.gitmodules` correct | ✅ |
| 2 | 4 shims dropped, zero-mod restored, 3 deps installed + in `pyproject` | ✅ |
| 3 | `import src.run_analysis` succeeds + a paper analysis runs | ✅ (cloud smoke ran full pipeline) |
| 4 | `pytest` no new failures vs the 8-failure baseline | ✅ (confirm re-run: `8 failed, 593 passed` — exactly the 8 baseline; 2 fallout failures fixed) |
| 5 | `run_batch --dry-run` valid decision on `hybrid_haiku_tools` (cloud) | ✅ post-QA-fix live run: **SELL** (`Underweight`→SELL), 8.5/10, $0.074 |
| 6 | `run_batch --dry-run` valid decision on `hybrid_aggressive_qwen` (local) + offline memory | ✅ post-QA-fix live run: **SELL** (`Underweight`→SELL), 3.6/10, $1.036; memory = offline file journal |
| 7 | LangGraph stack pinned to tested/resolved (no downgrade) | ✅ (`1.0.10 / 1.2.16 / 4.1.1 / 3.1.0`) |
| 8 | Report written; work-order docs committed | Report ✅; commit pending |

**Pending to close:**
- [x] Confirmation full-suite pytest → exactly the 8 baseline failures (`8 failed, 593 passed, 2 skipped`); re-confirmed after the broadened §2.5 matcher. ✅
- [x] **exit-6 local smoke** — credits topped up; fresh run → **SELL**, 5.1/10, $0.945. ✅
- [x] Commit changes + work-order docs. ✅

**Net for the gate: 8 of 8 exit criteria met.** The vendor upgrade is complete, zero-mod, and the engine runs end-to-end with valid decisions on **both** the cloud and local paths. → Hand to Codex QA (`docs/CODEX_TRIFECTA_ENGINE_HANDOFF.md`); then paper-smoke UAT; then Engine App Manager + Arbiter re-verify. **DEVELOP does not declare Done.**

## 11. Known limitations / follow-ups (out of TRI-66 scope)
- Model-string currency (`claude-sonnet-4-5`/`haiku-4-5` not in v0.3.0 catalog — advisory warnings): **TRI-70/73**.
- Cache report-key for social: our `make_cached_analyst` keys by `spec.key="social"`; v0.3.0's report key is `sentiment_report`. Affects *cache-hit* effectiveness for the sentiment analyst only (correctness unaffected on cache miss); flag for a follow-up if cache hit-rate matters.
- `import src.hybrid_graph` standalone still fails (vendor sys.path only wired by entrypoints): **TRI-72**.
- Accuracy stale-date test time-bomb: **TRI-71**. Full dependency lockfile: **TRI-65**.
- **Result files clobber each other** — `results/<TICKER>/analysis_<date>_<config>.json` has no run timestamp, so any re-run (including QA's independent verification) **overwrites the prior run's evidence**. Discovered when QA's re-run replaced the DEVELOP cloud-smoke evidence file. Gate-integrity nuisance; suggest a ticket to add a run timestamp (or run-id) to result filenames.
- **Run-to-run decision variance observed:** same ticker, same day, same config produced `Overweight` (one run) and `Underweight` (a later run) on the cloud path. Not a bug — LLM non-determinism — but TRI-70's benchmark must account for it (repeat runs / variance bars, not single-shot decisions).

---
*Next: Codex QA (`docs/TASK_TRI-66_CODEX_REVIEW.md`) → paper-smoke UAT → Engine App Manager + Arbiter re-verify.*
