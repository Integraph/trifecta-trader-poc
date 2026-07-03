# TRI-66 — Arbiter Sign-off Package (Engine App Manager → Ecosystem Arbiter)

**Issue:** TRI-66 (Urgent) — upgrade vendored TradingAgents v0.2.0 → v0.3.0, restore zero-mod
**Prepared by:** Engine App Manager · **Date:** 2026-07-02 · **Branch:** `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030` · **Reviewed tip:** `135b366`
**Gate status:** DEVELOP ✅ → QA ✅ (2 rounds) → UAT ✅ → **awaiting Arbiter re-verify + Done sign-off**

> **App-Manager recommendation: SIGN OFF Done**, with the carry-forward items (§6) tracked on their tickets. The upgrade is complete, zero-mod is restored, and the engine runs end-to-end on v0.3.0 producing valid paper decisions on both the cloud and local paths. **I do not declare Done — this is a recommendation + evidence for your independent re-verification.**

---

## 1. What TRI-66 delivered (scope: the pure vendor upgrade — nothing more)

- Vendored `TradingAgents` **v0.2.0 + our Task-020 commit → upstream v0.3.0** (peeled `85946c2f…`); the tag checkout **discarded our 4-file local commit wholesale → true zero-mod restored**.
- Added the 3 deps v0.3.0 requires (`stockstats>=0.6.5`, **`langgraph-checkpoint-sqlite==3.1.0`** — module-level import, was undeclared, **hard-blocked the engine import**; **`yfinance>=1.4.1`**), and pinned the direct LangGraph stack (`langgraph 1.0.10 / langchain-core 1.2.16 / langgraph-checkpoint 4.1.1 / langgraph-checkpoint-sqlite 3.1.0`).
- Reworked the 3 vendor-importing source files for the v0.3.0 API. Three real breakages found & fixed: **(a)** `create_risk_manager` removed → `create_portfolio_manager(llm)` (Risk Judge; memory arg dropped, now via v0.3.0's centralized `past_context`); **(b)** spec-driven node names (`KeyError 'Msg Clear Sentiment'`) → re-synced to `build_analyst_execution_plan`; **(c)** the PM's new **5-level** decision vocabulary → a contract-preserving 5→3 mapping in `signal_processing.py` (Overweight→BUY, Underweight→SELL).
- **NO model-string changes and NO new configs** (those are TRI-70 — verified below).

## 2. Commit trail (all on the branch)

| Commit | Purpose |
|---|---|
| `a8fafc8` | Vendor upgrade v0.2.0→v0.3.0; zero-mod restored; deps + pins; Risk-Judge remap |
| `c035adc` | Close exit-5/6 — both paper smokes live; broaden PM decision-label matcher |
| `2a57026` | **QA round-1 P1 fix** — decision matcher genuinely line-anchored |
| `2cffe43` | QA round-1 review artifact (committed verbatim) |
| `f07b3a4` | **QA round-2 fix** — quoted/fenced stale decisions can't override |
| `135b366` | QA round-2 review artifact (current tip) |
| (`da831cb` / `f93c872`) | TRI-32 — prior commits + docs committed/pushed (v2.0.1/2.0.2) |

## 3. Exit criteria — 8/8

| # | Criterion | Status |
|---|---|---|
| 1 | Submodule at v0.3.0 `85946c2`, gitlink/`.gitmodules` correct | ✅ |
| 2 | 4 shims dropped, **zero-mod restored**, 3 deps in `pyproject` | ✅ |
| 3 | `import src.run_analysis` + a paper analysis runs | ✅ |
| 4 | `pytest` no new failures vs the 8-failure baseline | ✅ `8 failed / 610 passed / 2 skipped` |
| 5 | Cloud smoke valid decision (`hybrid_haiku_tools`) | ✅ **BUY 8.5/10** |
| 6 | Local smoke valid decision (`hybrid_aggressive_qwen`) + offline memory | ✅ **HOLD 5.1/10**; file-journal memory |
| 7 | LangGraph stack pinned, not downgraded | ✅ `1.0.10 / 1.2.16 / 4.1.1 / 3.1.0` |
| 8 | Report written + docs committed | ✅ `docs/TASK_TRI-66_REPORT.md` |

## 4. QA & UAT summary

- **QA (Codex), 2 rounds — both returned CHANGES-REQUESTED on real defects, both fixed:** round 1, the decision matcher was *claimed* header-anchored but wasn't (prose could override a real decision line) → fixed + line-anchored. Round 2, quoted/fenced stale decisions could still override (v0.3.0 injects prior decisions as `past_context`, so this was realistic) → fixed (`>` removed from the line class; fenced code stripped; fails safe to loud `UNKNOWN`). DEVELOP honestly corrected a false "header-anchored" claim in its own report — the gate working as intended. Per §10's 2-round cap, escalation resolved: accept the fix; structured `PortfolioDecision` extraction (the durable fix) stays **TRI-70**.
- **UAT (independent operator, M3 Max), headless paper-smoke — verdict PASS.** All mandatory tests passed; observational items recorded; G3 not-run (optional). Full result: `docs/TASK_TRI-66_UAT_RESULT.md`. Highlights: cloud C1 = BUY 8.5/10; local D1 = HOLD 5.1/10 (below the 8.0 gate — recorded per TRI-70, not a fail); F1 safety = no `EXECUTED` audit record in either `audit/` or `results/audit/` (time-scoped) + all smokes printed `[DRY RUN — no order submitted]`; G1 invalid symbol `ZZZZQQ` → graceful `NoMarketDataError`, no confident signal.

## 5. App-Manager independent verification (what *I* re-ran/re-read — not the runner's word)

Confirmed against the tree from the App-Manager environment:

- **Reviewed code state:** `git rev-parse --short HEAD` = `135b366`; `git diff --stat 135b366 -- src vendor pyproject.toml tests config` = **empty** (no code drift since the QA-reviewed tip).
- **Zero-mod (the load-bearing claim):** `git submodule status` → `85946c2… (v0.3.0)`; **`git -C vendor/TradingAgents diff v0.3.0 | wc -l` = 0**. Genuinely upstream, no stray patch.
- **Safety hardcode:** `grep -n "paper=True" src/execution/executor.py` → lines **45, 49** (`# ← HARDCODED. NEVER CHANGE THIS.`). Live trading remains impossible in code (TRI-31 deferred).
- **No config/model-string change (scope integrity):** the `f07b3a4`/`c035adc` diffs touch only `hybrid_graph.py`, `signal_processing.py`, `pyproject.toml`, the two test files, the report, and the vendor gitlink — **no `config/hybrid_llm.yaml` or `_DEFAULT_CONFIGS` model edits.**
- **Extraction seam — I ran the E1–E8 probe myself** (pure-Python, no vendor deps): **SUMMARY: ALL PASS (13/13)**, including the QA round-2 blockquote (E5) and fenced (E6) cases, prose-safety (E4), out-of-vocab `Neutral`→`UNKNOWN` (E1d), and last-line-wins (E7). This is the seam that took two QA rounds — independently confirmed, not relayed.

**What I could NOT independently re-execute** (honest boundary — the App-Manager sandbox has no Ollama, no project venv, no paper credits, and is a different machine than the M3 Max): the full `pytest` run, the C1/D1/G2 pipeline smokes, and F1's `find … -newer` scoped to the live run window. For those I rely on the UAT runner's + DEVELOP's recorded evidence (commands, logs, preserved result JSONs). **Recommend the Arbiter re-run at least: `pytest --ignore=…`, one cloud + one local `--dry-run` smoke, the E-probe, and the zero-mod diff.**

## 6. Carry-forward for the Arbiter (out of TRI-66 scope — tracked)

Accepting TRI-66 does **not** resolve these; they are correctly scoped elsewhere:

- **TRI-70 (High)** — the real question. Local `hybrid_aggressive_qwen` scored **5.1/10, below the 8.0 execution gate** (its signals wouldn't paper-trade), on a **stale qwen2.5** and an economically-mislabeled config (clouds the expensive Sonnet tool slot). TRI-70 owns: benchmark **current** models (Qwen 3.6 — 3.7 is cloud-only), honest config economics, the `hybrid_graph.py` pricing-table refresh, robust **structured `PortfolioDecision` extraction**, and repeat-run design.
- **TRI-78 (High)** — **run-to-run non-determinism**: same ticker/day/config has produced opposite decisions across runs (this session both C1 & G2 = BUY, but the pattern is real). TRI-69/TRI-70 must measure stability, not single shots.
- **E9 residual** — a plain, un-quoted, un-fenced prose line beginning exactly like a decision header can still mis-map; narrow, documented, durable fix = TRI-70 structured extraction.
- **TRI-71** (accuracy stale-date tests — 5 of the 8 baseline failures), **TRI-72** (hybrid_graph standalone import), **TRI-73** (config refresh), **TRI-75** (post-upgrade admin-UI smoke), **TRI-76/77/79** (cache key / deprecated alias / result-file run-id), **TRI-65/67** (lockfiles / langgraph lock — note ~52 transitive packages drifted on the dep re-resolve).

## 7. Bottom line

The vendor upgrade is a **success**: zero-mod restored, engine runs end-to-end on v0.3.0, valid paper decisions on both paths, gate fully passed with two real QA catches fixed. It does **not** answer the viability question — the first local datapoint (5.1/10, sub-gate) is a yellow flag, but that's exactly TRI-70's job on the now-upgraded engine, not a TRI-66 concern.

**Recommended action:** Arbiter independently re-verifies (§5), then **signs TRI-66 Done**; TRI-70 becomes the next runnable item (after TRI-32 push confirmation and, if desired, the TRI-75 UI smoke). *App Manager does not declare Done.*
