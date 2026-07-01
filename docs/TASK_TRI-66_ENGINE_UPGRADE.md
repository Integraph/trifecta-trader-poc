# TRI-66 — DEVELOP Work Order: Upgrade vendored TradingAgents v0.2.0 → v0.3.0 (v7)

**Issue:** TRI-66 (Urgent) · **Stage:** DEVELOP (Claude Code) → QA (Codex) → UAT (paper-smoke) · **Repo:** `trifecta-trader-poc` · **Branch:** `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030`
**Report to:** `docs/TASK_TRI-66_REPORT.md`. **Owner:** Engine App Manager. **Arbiter re-verifies before Done.**

> **v4 — scope NARROWED to the pure vendor upgrade (Arbiter, June 30, after the Manager + QA + Dev convergent review).** All **benchmark-config construction** — the test-only `benchmark_opus_a` (Opus 4.8), the all-local `benchmark_local_b`, the `hybrid_graph.py` pricing row, and model-string currency — **moves to TRI-70.** Rationale: building an all-local config (whose tool slot needs a local function-calling model) and proving qwen tool-calling holds on v0.3.0's routing is **net-new benchmark work**, and local tool-calling is exactly where the baseline is already red (2 of 8 failures). TRI-66 must not stall on work unrelated to the upgrade.
>
> **v5 (Dev + QA review, 2026-07-01, verified vs the v0.3.0 tag):** v4's dependency precondition was **wrong** — `stockstats` is *not* the only gap. v0.3.0 also requires **`langgraph-checkpoint-sqlite>=2.0.0`** (undeclared here; imported at **module level** → hard-blocks `import src.run_analysis`) and **`yfinance>=1.4.1`** (our venv is below floor at `1.2.0`). Both are now added to `pyproject` in Step 3. Also fixed the local-smoke wording (`--hybrid` only accepts **named** configs).
>
> **v6 (Dev + QA review, 2026-07-01):** reordered Steps 3–4 (deps → bump to the v0.3.0 tag, which discards our commit + shims — no manual reverting); pin the **whole langgraph stack** (Step 6) — the checkpoint-sqlite↔langgraph resolution was **verified clean in an isolated resolve** (`langgraph-checkpoint-sqlite 3.1.0` + `langgraph-checkpoint 4.1.1` hold `langgraph 1.0.10`, no downgrade); exact smoke commands in exits 5/6; explicit `stockstats>=0.6.5` floor; removed the throwaway-config carve-out.
>
> **v7 (Dev + QA review, 2026-07-01):** named the one **hard code breakage** in Step 5 — v0.3.0 removed `create_risk_manager` → `create_portfolio_manager(llm)` (the Risk Judge; signature change drops the `memory` arg), which fails both smokes until fixed and is a behavioral change to flag. Fixed the prereq wording (pull `qwen2.5:14b`); relabeled Step 6 the *direct* langgraph stack (prebuilt/sdk are transitive → TRI-65). DEVELOP runs after this.

> **Paper/sandbox only — `--execute` FORBIDDEN.** Vendor changes are deliberate and gated. Goal = **true zero-mod**.

## Objective
Bring vendored `TradingAgents` from **v0.2.0 + our Task-020 commit** to upstream **v0.3.0** (peeled commit `85946c2f60768ab2dae23a5a36cd927662feef94`), reconcile our shims to **restore zero-mod**, pin the LangGraph 1.x we already run, and prove the engine **still runs on existing cloud and local paths**. Nothing more. Unblocks TRI-70 (benchmark setup) → TRI-69.

## Prerequisites (have these ready before Step 1)
- **Machine:** MacBook Pro M3 Max / 128 GB; Python **3.11+**; the repo's venv.
- **Ollama running, with `qwen2.5:14b` pulled** — that's the model `hybrid_aggressive_qwen` names, and it's what the Step-8b local smoke uses. (Model *selection/currency* is TRI-70; TRI-66 just needs this one model present to prove Ollama routing survived.)
- **`.env` populated:** `ANTHROPIC_API_KEY` (cloud smoke + Haiku tools) and Alpaca **paper** keys (the executor initializes even on `--dry-run`); Supabase keys if the pipeline path touches them. **No live keys — `--execute` forbidden.**
- **Submodule tags fetched:** `git -C vendor/TradingAgents fetch --tags` so the v0.3.0 tag / peeled `85946c2…` is available locally.

## Verified preconditions (Arbiter-confirmed against the tree)
- Pin `5de91bc` = v0.2.0 + our commit on `f047f26`; target = v0.3.0 (annotated tag → peeled commit `85946c2f…`).
- **The 4 mods are optional-dep lazy-import shims** (rank_bm25 / stockstats / google client), and **load-bearing**: `import src.run_analysis` succeeds today *only* because of them — `stockstats`/`rank_bm25`/`langchain-google-genai` are absent from the venv. `stockstats` **is a declared dep** (`pyproject.toml`).
  - v0.3.0: `memory.py` dropped rank_bm25; `factory.py` made provider imports lazy → **those 2 shims drop clean.** `dataflows/stockstats_utils.py` + `y_finance.py` **still eager-import `stockstats`** → **need the dep installed, not the shim.**
- **→ Dependency gap is THREE deps, not one** (v0.3.0's own `pyproject` requires them; verified against the tag):
  - `stockstats>=0.6.5` — declared here (bare), eager-imported in `dataflows/*` → install. (Dropping the dataflows shim without it = `ModuleNotFoundError`.)
  - **`langgraph-checkpoint-sqlite>=2.0.0` — NOT declared here, imported MODULE-LEVEL** (`graph/checkpointer.py:14` `from langgraph.checkpoint.sqlite import SqliteSaver`; `graph/trading_graph.py:37` imports `.checkpointer`). Chain: `import src.run_analysis → trading_graph → .checkpointer → langgraph.checkpoint.sqlite`. **Without it, exit-3 fails even after stockstats + dropping all 4 shims.** Add to `pyproject` + install.
  - **`yfinance>=1.4.1` — our declaration is bare and the venv is below floor (`1.2.0`); v0.3.0 rewrote `dataflows/y_finance.py` against the 1.4.x API** (the Step-8 data path). Bump the floor + install.
  - `parsel` / `backtrader` / `questionary` / `chainlit` are absent from our venv but **not** eager-imported by our entry chain → **not** blockers (verified). The import gate is exactly: `stockstats` + `langgraph-checkpoint-sqlite`.
- **LangGraph already 1.x:** unpinned in `pyproject.toml`; installed `1.0.10` / `langchain-core 1.2.16`; pipeline runs on it. Pin to the working 1.x — do NOT downgrade (TRI-67 = the cross-engine lock).
- **Memory retrieval changed:** v0.3.0 removed `rank_bm25` from `memory.py`. Verify the replacement doesn't require an embeddings endpoint/API that would break the offline path (Step 7).
- **Imports:** `import src.hybrid_graph` fails standalone — the vendor sys.path is wired by the **entrypoints `run_analysis.py:21` and `run_batch.py:22`**, not by `hybrid_graph.py` itself. Gate = `import src.run_analysis` (or run the pipeline via an entrypoint). Standalone-import fix = TRI-72 (out of scope).
- **Real test baseline = 8 failures** (TRI-34): 5× accuracy stale-date (TRI-71) + 1× port-8420 (env) + 2× mistral-small tool-calling — all pre-existing.
- **No all-local config exists** (all 13 configs use Anthropic tools) → **building one is TRI-70, not TRI-66.**

## Steps
1. **Capture the REAL baseline.** `pytest`; record the actual fail/skip set (expect the 8 above). Save an existing-config `run_batch --dry-run` output (one ticker) for comparison.
2. **Read the mods** (`git show 5de91bc`) — confirm the 4 optional-dep shims.
3. **Prepare deps (required for v0.3.0 to import).** In `pyproject.toml`: **set `stockstats>=0.6.5`** (currently bare), **add `langgraph-checkpoint-sqlite>=2.0.0`**, and **bump `yfinance>=1.4.1`**. Install (`pip install -e .` / `uv sync`); capture `pip freeze` **before/after**. Verify `python -c "import langgraph.checkpoint.sqlite, stockstats, yfinance"` and yfinance ≥ 1.4.1. **These dep additions are IN TRI-66 scope** (v0.3.0 won't import without them); *locking/pinning all* floating deps stays TRI-65.
4. **Bump submodule to v0.3.0 — this drops the shims.** Checkout the tag (peeled `85946c2…`); it **discards our Task-020 commit and all 4 shims wholesale** (do NOT hand-revert files). Update the gitlink + `.gitmodules`. Then verify **zero-mod**: `git -C vendor/TradingAgents status` clean and `git -C vendor/TradingAgents diff 85946c2…` empty (no stray local edits), and `import src.run_analysis` succeeds (Step-3 deps present).
5. **Update our importing files for the v0.3.0 API — expect real rework here, not a one-line diff.** **The 3 _source_ files that import the vendor** (tests import it directly too — watch for test fallout): `src/run_analysis.py:27-28` (`graph.trading_graph.TradingAgentsGraph`, `default_config.DEFAULT_CONFIG`); `src/hybrid_graph.py:22-25,39` (`graph.trading_graph`, `graph.setup.GraphSetup`, `graph.conditional_logic.ConditionalLogic`, `agents`); `src/hybrid_llm.py:378` (`llm_clients.factory.create_llm_client`, lazy).
   - **🔴 KNOWN HARD BREAKAGE (verified vs the tag) — the Risk Judge:** v0.3.0 **removed `create_risk_manager`** (our `hybrid_graph.py:39` import), replaced by **`create_portfolio_manager(llm)`** — a rename **and** a signature change (the `memory` arg is gone; v0.3.0 moved memory to the file journal). We use it at `:39` (import), `:300-301` (`create_risk_manager(self.reasoning_deep_llm, self.risk_manager_memory)`), `:322` (`add_node("Risk Judge", …)`). It's a **lazy** import (via `run_analysis.py:274`) so exit-3 still passes — but **both Step-8 smokes route through `HybridTradingGraph`, so exit-5/6 hard-fail** with `ImportError` until fixed. v0.3.0 wires it as `create_portfolio_manager(self.deep_thinking_llm)` (`setup.py:76`). **Dropping the `risk_manager_memory` feed also orphans `:217/231/450` — that's a behavioral change on the quality-critical deep slot: SURFACE it / escalate to the App Manager, do NOT silently paper over.**
   - v0.3.0 also **materially refactored graph construction** (spec-driven `analyst_factories` in `setup.py`, changed agent roster, new memory model). `hybrid_graph.py::HybridGraphSetup` hand-mirrors the *old* `GraphSetup`, so matching v0.3.0 is likely **substantive rework** — this is where the real TRI-66 code effort and behavioral risk live.
   - `create_social_media_analyst` (`:29`) survives as a **deprecated back-compat alias** in v0.3.0 — fine for TRI-66; migrate to `create_sentiment_analyst` later, not here.
   - For every other call site (analyst/researcher/trader/debator factories, `GraphSetup`/`ConditionalLogic`/`TradingAgentsGraph` constructors): **diff vs the v0.3.0 API, fix what moved, and surface behavioral changes** rather than silently adapting.
6. **Pin the direct LangGraph stack.** Pin `langgraph`/`langchain-core` to the tested `1.0.10`/`1.2.16`, **and pin the newly-pulled `langgraph-checkpoint-sqlite` + `langgraph-checkpoint` to their resolved versions**. **Verified resolution (isolated resolve, 2026-07-01):** `langgraph-checkpoint-sqlite>=2.0.0` resolves cleanly with `langgraph==1.0.10` → `checkpoint-sqlite 3.1.0` + `langgraph-checkpoint 4.1.1` (langgraph held, no downgrade). The resolve also pulls `langgraph-prebuilt 1.0.10` + `langgraph-sdk 0.3.15` as **transitive** deps — those get locked by TRI-65's lockfile, not hand-pinned here. Confirm your resolve matches; if adding checkpoint-sqlite floated `langgraph` off 1.0.10, bring the stack back to the tested set and re-run the Step-8 smokes on it.
7. **Offline-memory check.** Confirm v0.3.0's memory retrieval works with no embeddings-API dependency on the local path — **surface it if it doesn't** (it affects TRI-70's all-local Config B).
8. **Smoke on EXISTING configs (one ticker each, paper `--dry-run`):**
   - `hybrid_haiku_tools` (existing production, Sonnet deep) → proves the **cloud** path survived.
   - The existing Ollama-deep config **`hybrid_aggressive_qwen`** — **pull its model (`qwen2.5:14b`)** for this smoke → proves the **local/Ollama path survived** (the path the **local-first** product direction, TRI-70 / §6.7, rides on — first-class, not an afterthought). **Purpose = Ollama routing survives the upgrade; the model *version* is irrelevant here** (`--hybrid` accepts named configs only, and TRI-66 adds none — so pull `qwen2.5:14b`, don't invent a config). Model selection/currency = TRI-70.
   - **Do NOT create new configs or change model strings here** — Opus/benchmark/currency work is TRI-70.
9. **Green.** `pytest` = no new failures vs the Step-1 real baseline.
10. **Report** → `docs/TASK_TRI-66_REPORT.md`: reconciliation per file, final submodule state (zero-mod?), deps synced, langgraph pins, the memory-retrieval finding, baseline vs post, both smoke outputs, anything unverified. **Commit the docs** (this work order + the manager/codex prompts) so a fresh worktree has them (fold into TRI-32).

## Exit criteria
1. `vendor/TradingAgents` at v0.3.0 (`85946c2…`); gitlink/`.gitmodules` correct.
2. All 4 shims dropped; **zero-mod restored** — `stockstats` + **`langgraph-checkpoint-sqlite`** + **`yfinance>=1.4.1`** installed and added to `pyproject`, so `import src.run_analysis` works — or any remainder justified.
3. `python -c "import src.run_analysis"` succeeds + a paper analysis runs.
4. `pytest` = **no new failures vs the captured real baseline** (the 8 known are pre-existing).
5. `python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run` → valid decision (cloud path, paper, no order placed).
6. `python -m src.run_batch --tickers AAPL --hybrid hybrid_aggressive_qwen --dry-run` (with `qwen2.5:14b` pulled) → valid decision, proving the **local/Ollama path** survived — and agent memory works **offline** (no new embeddings-API dependency).
7. `langgraph`/`langchain-core` **and** `langgraph-checkpoint`/`langgraph-checkpoint-sqlite` pinned to the tested/resolved versions (whole stack not floating; `langgraph 1.0.10` not downgraded).
8. `docs/TASK_TRI-66_REPORT.md` written with evidence; work-order docs committed.

## Guardrails
- `--execute` forbidden; paper/dry-run only.
- **Explicitly OUT of TRI-66 (→ other tickets):** all benchmark configs + Opus 4.8 + `hybrid_graph.py` pricing + model-string currency (**TRI-70**); LangGraph cross-engine lock (**TRI-67**); lockfiles (**TRI-65**); accuracy time-bomb (**TRI-71**); hybrid_graph standalone-import (**TRI-72**); admin-UI rebuild. **Do not create any new config or change any model string in TRI-66.**
- **Model policy (context, not TRI-66 work):** selection is **local-first, outcome over brand** — governed by **TRI-70 + operating-prompt §6.7**. TRI-66's only job on the model front is to prove the **existing** cloud and local paths still run post-upgrade; it picks/changes nothing.
- **Surface, don't paper over**, any v0.3.0 behavior change (data contract, news windows, memory retrieval, provider routing).

## QA brief for Codex (after DEVELOP)
Assume the report is wrong until proven. Check: (a) submodule at v0.3.0 (`85946c2…`) and zero-mod genuinely restored (diff vs the tag; **`stockstats` + `langgraph-checkpoint-sqlite` + `yfinance>=1.4.1` actually installed and in `pyproject`**, not a leftover shim); (b) no new failures vs the real 8-failure baseline on a clean run; (c) **no new config was added and no model string changed** (that's TRI-70); (d) both existing-config smokes produced real decisions; (e) offline memory works (no embeddings API); (f) the **langgraph stack is pinned consistently** — `langgraph 1.0.10` held (not downgraded or floated), `langgraph-checkpoint`/`checkpoint-sqlite` pinned to resolved. Snippets OK; never commit app code. → `docs/TASK_TRI-66_CODEX_REVIEW.md`.

---

*DEVELOP stage of DEVELOP → QA → UAT (`ECOSYSTEM_CONTEXT.md` §10; headless UAT = paper-smoke). **Done = exit 1–8 AND Codex QA APPROVED AND Engine App Manager + Arbiter re-verify.** Then TRI-70 builds the benchmark configs (Opus test-only + all-local) and runs the Step-0 benchmark on the upgraded engine.*
