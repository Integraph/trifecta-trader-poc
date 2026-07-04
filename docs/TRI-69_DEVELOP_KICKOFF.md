# TRI-69 — DEVELOP Kickoff (Claude Code) · v3

You are **DEVELOP** (Claude Code) on the MacBook Pro M3 Max. This is **the go/no-go for the whole project**: does the engine's decision predict next-horizon direction better than horizon-matched buy-and-hold, **net of cost**? TRI-66 is Done; the engine is on v0.3.0.

## Canonical spec
Read first: **`docs/TASK_TRI-69_EDGE_CHECK.md` (v3)** — it's a **pre-registration**; its Design + Decision-rule are **locked** (the App Manager owns them; you fill the mechanical universe + execute). Also read **`docs/TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md`** (fill model, benchmark, survivorship, exit rule) — if it's not in `docs/` yet, get it copied there before starting; do not run without it.

## What this is / is NOT
- A **historical, out-of-sample, POINT-IN-TIME backtest** on past dates — runnable now, ~1–2 days.
- A **KILL-SWITCH, not a green-light.** Falsifies the thesis (no/negative edge → stop/pivot); a strong result is only *"promising, run bigger,"* never *"confirmed edge."*
- **Not** the weeks-long live-condition stability track. Keep it minimal.

## 🔴 The two things that silently corrupt the result — get them right first
1. **Look-ahead leak in fundamentals (verified in the vendor code).** `get_fundamentals()` (`vendor/.../dataflows/y_finance.py:274`) ignores `curr_date` and returns **today's `Ticker.info`** — 52-week hi/lo, 50/200-day averages, TTM ratios — all future-relative, so for a past decision date they encode where the price went. In point-in-time mode you **must drop/neutralize that tool** (the financial *statements* via `get_balance_sheet`/`get_cashflow` are date-bounded ✓ — keep those). **Prove no `fiftyTwoWeek*/fiftyDayAverage/twoHundredDayAverage/trailing*` field reaches the analyst context.** Without this, any "edge" is partly reading tomorrow's chart.
2. **Statistics: cross-sectional dependence + non-50% base rate.** ~25 large-caps on one date are largely **one macro bet** — a naive binomial over ~100 decisions overstates significance both ways. **Effective N ≈ number of dates.** Score with a **date-clustered** test (date-level sign/permutation, or date-level portfolio excess return vs the benchmark). **Primary test = vs horizon-matched buy-and-hold, net of cost**; "vs chance" is secondary and its null is the **realized base rate**, not 0.5.

## Integrity discipline
1. **Pre-register in its own commit BEFORE running.** The App Manager locks the rules; you write the mechanical universe/dates and commit the full pre-registration first — the git timestamp proves criteria preceded data. Don't change them after.
2. **Out-of-sample, mechanical, no survivorship:** select the universe as **index membership as-of the earliest decision date** (per the integrity doc) from a **named, reproducible source** (point-in-time constituents aren't in yfinance — archived list / as-of-date ETF holdings snapshot), then draw the ~20–30 eval names by a **pre-registered mechanical rule** (top-N by as-of-date market cap, or seeded random sample — state the seed), **not** today's known liquid names or hand-picking. **Exclude AAPL/NVDA/TSLA at ALL dates** (the engine was developed on AAPL).
3. **Point-in-time / look-ahead-safe:** OHLCV up to date + as-of news + date-bounded statements; **live Reddit/Polymarket off; fundamentals-overview neutralized** (per 🔴 #1).

## Build prereqs (first)
- **Config A (stable arm, TEST-ONLY, pinned):** `temperature=0` on **all three slots** — `tool_provider: ollama` (`qwen3-coder:30b`) + `reasoning_quick = ollama qwen3.5:9b` + `reasoning_deep = anthropic claude-sonnet-4-5-20250929`. **Sonnet, not Opus** — Opus-4-8 rejects `temperature`. **temp=0 on the judge alone is NOT enough:** the local tool/quick slots sample at Ollama's ~0.7–0.8 default unless pinned, and if their narratives drift the judge flips even at temp=0 (that's what TRI-70's 0.60 is). **New plumbing required:** the engine doesn't pass temperature today (`DEFAULT_CONFIG["temperature"]=None`; `create_hybrid_llms` threads only provider+model — `src/hybrid_llm.py:382/388/399`), so thread `temperature=0` through all three `create_llm_client` calls and verify each client's `get_llm()` applies it (Ollama/OpenAI-compat path especially). Flag `test-only`; never production by default.
- **Point-in-time / no-leak mode:** add the live-social-off + fundamentals-neutralize toggles. **Determinism proof (tests the whole pipeline, not just the judge):** N=3 runs on the three already-burned TRI-70 points — **`AAPL`/`NVDA`/`TSLA` @ `2026-06-27`** (all excluded from eval) → **decision-identical**. This is where an unpinned local slot shows up: if temp=0 isn't actually plumbed on all three, it fails here (TRI-70's 0.60 predicts it). **A single flip fires the checkpoint — do not loosen to "2 of 3."** If you can't close the leaks or make it deterministic, **STOP and surface it.**

## Run + score + verdict
- One point-in-time run per `(ticker, date)` (deterministic → no repeats), run-id'd.
- Score per the integrity doc: **date-clustered** directional test vs horizon-matched net-of-cost benchmark; report hit-rate, per-date portfolio excess return, **HOLD rate + directional N**, test statistic + α, power caveat.
- **HOLD-dominance:** TRI-70 was modal-HOLD; HOLD = no-trade (excluded). If HOLD 60–80%, ~100 decisions → only ~20–40 directional trials. If underpowered → **expand the universe/dates (not a verdict)**; size to hit the pre-registered min directional N.
- **Verdict (pre-registered):** no edge → STOP/PIVOT · beats benchmark (date-clustered) → PROMISING-run-bigger · underpowered → INCONCLUSIVE + N needed.

## Guardrails & checkpoint
- Paper only; no live. Config A is test-only, never production by default.
- **Surface before committing around it:** unclosable leak (fundamentals/live-social), determinism failure, non-out-of-sample sample, or HOLD-dominance blocking min directional N. Integrity blockers, not workarounds.

## Deliverable → `docs/TASK_TRI-69_REPORT.md`
Pre-registration (committed first), universe/dates, the date-clustered results, HOLD rate + directional N, the go/no-go verdict with the honest power caveat, and compute/$ spent. Then Codex QA → UAT → Arbiter. **You do not declare Done.**
