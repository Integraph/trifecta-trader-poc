# TRI-66 — QA Kickoff (Codex, independent adversarial review)

You are **QA (Codex)** for **TRI-66** (vendored TradingAgents v0.2.0 → v0.3.0). Independent, adversarial review. **Assume every claim is wrong until you've verified it against the tree yourself.** You write **no committed application code** (fix snippets are fine). Output → `docs/TASK_TRI-66_CODEX_REVIEW.md`.

**Independence (important):** form your own findings **first**. Do **NOT** read `docs/TASK_TRI-66_REPORT.md` until you've independently verified the items below — then you may compare and note discrepancies. Your standing brief is `docs/CODEX_TRIFECTA_ENGINE_HANDOFF.md`.

**Under review:** branch `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030`, commits `a8fafc8` + `c035adc`. Scope was the **pure vendor upgrade** — **no** model-string changes and **no** new configs (those are TRI-70). Paper/`--dry-run` only; `--execute` forbidden.

**Prereqs to run the checks:** M3 Max with Ollama (`qwen2.5:14b` pulled) + Anthropic **paper** credits (topped up) for the smokes.

## Verify each — against the tree / a clean run, not the report
- **(a) Zero-mod genuinely restored.** Submodule at v0.3.0 peeled `85946c2`; `git -C vendor/TradingAgents status` clean **and** `git -C vendor/TradingAgents diff v0.3.0` **empty** (no stray commit, no leftover shim). Confirm the 3 deps (`stockstats>=0.6.5`, `langgraph-checkpoint-sqlite`, `yfinance>=1.4.1`) are actually **installed** — the import must not be secretly relying on a leftover shim.
- **(b) No new config, no model string changed** (that's TRI-70). `hybrid_haiku_tools` + the four `*_aggressive` configs **byte-unchanged**; the only source touched is `hybrid_graph.py`, `signal_processing.py`, `pyproject.toml`, the two test files, + the vendor gitlink.
- **(c) No new test failures vs the real 8-failure baseline** — run `pytest` yourself (exclude the 3 live-cloud/broker files as the baseline did). Expect **8 failed / ~596 passed** (5× accuracy stale-date, 1× port-8420, 2× mistral-small) and **0 new**. Confirm the **+7 new signal tests are real**, not tautological.
- **(d) Both smokes produced real decisions** (not errored output dressed as success), `--dry-run`, no order placed:
  - `python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run` → valid **BUY/HOLD/SELL**.
  - `python -m src.run_batch --tickers AAPL --hybrid hybrid_aggressive_qwen --dry-run` → valid decision (local/Ollama path).
- **(e) Risk-Judge remap correct.** `create_risk_manager` → `create_portfolio_manager(llm)` (memory arg dropped); the "Portfolio Manager" node builds and the graph compiles. **Fault-check the behavioral claim:** does the deep slot still receive memory via the vendor's `propagate()` → `past_context` (v0.3.0's centralized `memory_log`), i.e. is the memory change benign as claimed — or did we silently strip the Risk Judge's memory?
- **(f) Decision mapping sound + prose-safe — probe this hardest (the engine's most fragile seam).** The 5→3 map (`Overweight→BUY`, `Underweight→SELL`) must be **header-anchored** (`Recommendation|Rating|Action|Final Transaction Proposal`), must **NOT** false-match reasoning prose (e.g. "an underweighted position" must stay UNKNOWN, not SELL), and must be **contract-preserving** (output stays BUY/HOLD/SELL). Try to break it with adversarial PM text.
- **(g) LangGraph stack pinned, not downgraded** — `langgraph 1.0.10` held; `langgraph-checkpoint 4.1.1` / `langgraph-checkpoint-sqlite 3.1.0` pinned to the resolved set.
- **(h) Offline memory** — v0.3.0's memory path needs no embeddings endpoint/API (local path safe).

## Rules
- **Escalate, don't fix silently.** Report anything that fails with `file:line` evidence + severity, plus any **measurement-integrity** risk (esp. the decision-extraction fragility across models — already flagged for TRI-70; confirm it's genuinely parked, not silently corrupting).
- Verdict: **APPROVED** or **CHANGES-REQUESTED** with specifics → `docs/TASK_TRI-66_CODEX_REVIEW.md`.
- You do **not** declare Done — the Engine App Manager + Arbiter re-verify after your review. Next stage after QA: paper-smoke UAT → Arbiter.
