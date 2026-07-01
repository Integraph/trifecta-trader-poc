# TRI-66 — DEVELOP Kickoff (Claude Code)

You are **DEVELOP** (Claude Code) executing **TRI-66** on the MacBook Pro M3 Max / 128 GB. Run it end-to-end through to the two paper smokes and a report. We need the smoke + test results **ASAP** to decide whether this MVP is viable — move with urgency, but follow the gate and verify against the tree (trust no report, including this one).

## Canonical spec
- **Work order (read first): `docs/TASK_TRI-66_ENGINE_UPGRADE.md` (v7)** — exact 10 steps + 8 exit criteria.
- **QA brief (for the downstream Codex review): `docs/CODEX_TRIFECTA_ENGINE_HANDOFF.md`.**

## Mission (nothing more)
Upgrade the vendored `TradingAgents` submodule **v0.2.0 → v0.3.0**, restore **true zero-mod**, pin the LangGraph 1.x stack, and prove the engine **still runs on the existing cloud and local paths**. **Do NOT** create configs, change model strings, or touch benchmark/Opus work — that is **TRI-70**.

## Setup
- **Branch:** `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030` from current `main`. First confirm your branch actually contains `docs/TASK_TRI-66_ENGINE_UPGRADE.md` (these docs were uncommitted — TRI-32); if not, get them before starting.
- **Prereqs:** Ollama running with **`qwen2.5:14b` pulled**; `.env` has `ANTHROPIC_API_KEY` + Alpaca **paper** keys; run `git -C vendor/TradingAgents fetch --tags`.

## Verified facts — build on these, don't re-derive
- Target = v0.3.0 **peeled commit `85946c2f60768ab2dae23a5a36cd927662feef94`**.
- The 4 local vendor mods are optional-dependency **lazy-import shims**; checking out the v0.3.0 tag **discards our commit + shims wholesale** (no manual file-reverting).
- **3 deps must be added/installed or v0.3.0 won't import.** In `pyproject.toml`: set `stockstats>=0.6.5`, **add `langgraph-checkpoint-sqlite>=2.0.0`**, bump `yfinance>=1.4.1`. (`langgraph-checkpoint-sqlite` is imported **module-level** in v0.3.0 and hard-blocks `import src.run_analysis` if missing.)
- **Pin the direct langgraph stack** to the verified-clean resolve — no downgrade: `langgraph 1.0.10` / `langchain-core 1.2.16` / `langgraph-checkpoint 4.1.1` / `langgraph-checkpoint-sqlite 3.1.0`. (`langgraph-prebuilt` / `langgraph-sdk` come along transitively → left to the TRI-65 lockfile.)
- **🔴 Known hard breakage — the Risk Judge:** v0.3.0 removed `create_risk_manager`; use **`create_portfolio_manager(llm)`** (the `memory` arg is gone). Fix `hybrid_graph.py:39` (import), `:300-301` (node build), `:322` (`add_node("Risk Judge", …)`), and the now-orphaned `risk_manager_memory` plumbing (`:217/231/450`). It's a lazy import so the import gate still passes, but **both paper smokes fail until this is remapped.** Dropping the per-ticker memory feed to the quality-critical deep slot is a **behavioral change — flag it in the report, do not silently paper over it.** Expect **real rework** in `hybrid_graph.py` (v0.3.0 materially refactored graph construction; `HybridGraphSetup` mirrors the old API).
- **Offline memory is safe:** v0.3.0's `memory.py` is a file-based decision journal (no embeddings / no API) — confirm, but it should not break the local path.
- **Test baseline = 8 known pre-existing failures** (TRI-34): 5× accuracy stale-date, 1× port-8420, 2× mistral-small. **"Green" = no NEW failures vs this baseline** (do not try to fix these here).

## Sequence (Steps 1–10 in the work order)
Capture the real baseline → add/install the 3 deps → checkout the v0.3.0 tag & verify zero-mod (`git -C vendor/TradingAgents status` clean, diff vs tag empty) → rework the 3 source importing files for the v0.3.0 API (incl. the Risk-Judge remap) → pin the langgraph stack → offline-memory check → **the two paper smokes:**
- `python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run`  (cloud path)
- `python -m src.run_batch --tickers AAPL --hybrid hybrid_aggressive_qwen --dry-run`  (local/Ollama path)

## Guardrails
- **Paper / `--dry-run` only — `--execute` is FORBIDDEN.**
- Do not bundle TRI-67 (langgraph lock), TRI-65 (lockfiles), TRI-71 (accuracy time-bomb), or TRI-72 (hybrid_graph packaging).
- **Checkpoint:** surface anything material — the Risk-Judge behavioral change, or any v0.3.0 behavior shift (data contract, news windows, provider routing) — to the Engine App Manager **before** committing around it. Otherwise run straight through.

## Deliverable → `docs/TASK_TRI-66_REPORT.md`
Per-file reconciliation, final submodule / zero-mod state, deps + the exact langgraph pins, the memory finding, **baseline-vs-post test deltas**, and **both smoke outputs** (the decisions we're waiting on), plus anything you couldn't verify. Then: Codex QA → paper-smoke UAT → Arbiter re-verify. **You do not declare Done.**
