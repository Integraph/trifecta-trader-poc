# Trifecta MVP — §B Measurement-Integrity Test Checklist

**Date:** June 30, 2026 (rev for v2.2) · **Author:** Ecosystem Arbiter · **Companion to:** `TRIFECTA_MVP_PLAN.md` §B · **Owner to implement:** the engine/scanner (Python) App Manager · **Verified by:** Arbiter re-check

## Why this exists
§B is the **measuring instrument** for the whole thesis test. Everything else in the MVP is throwaway and gets light testing — **§B code does not.** A bug here doesn't crash; it returns a confident **wrong** verdict, and we graduate a fake edge or kill a real one. This repo has a documented history of exactly this: Phase-0 found tautological tests that passed without exercising the real code. This checklist is the acceptance bar each integrity component must clear before any of its numbers are trusted.

## The meta-rule (read first) — defeat the false-green
For **every** test below, a **negative/mutation test is mandatory**: deliberately break the property and prove the test **fails**. A test that can't fail is worthless. Two distinct things, don't conflate them:
- **Negative test (lives in the normal suite, PASSES):** asserts the guard correctly **rejects** bad input (e.g. a stale bar is refused). Green = guard works.
- **Mutation test (run on demand, must go RED):** deliberately break the guard in the code and confirm the test suite **fails**. Green-when-broken = the test is tautological. This is the specific antidote to this repo's false-green history.
Concretely, each integrity check ships with both: a negative case that passes by asserting rejection, and a demonstrated mutation that red-lights. No component is "done" until its mutation is shown to fail the suite.

Also non-negotiable: tests run against the **real** function, on **real-shaped** data — no mock that returns the answer the assertion wants. If a test would pass against an empty/stub implementation, it doesn't count.

---

## 1. Look-ahead / point-in-time — split into two testable properties (CC#4)
*LLMs are not deterministic (sampling, and temp=0 isn't bit-stable across runs/hardware), so "re-run the model and assert the same decision" is flaky. Split it:*
**1a — Look-ahead guard (deterministic, testable):** the stored `input_snapshot` for a decision at *t* contains **no** data stamped > *t*.
- Assert every field's timestamp ≤ the `asof_cutoff`. **Negative:** inject a `t+1` bar/news item → the snapshot builder **raises/rejects** (passes by rejecting).
- **Boundary:** exact cutoff defined and tested (only bars closed before the decision timestamp; intraday news only up to *t*; timezone explicit).
**1b — Deterministic replay of the POST-LLM pipeline:** from the **stored raw model output** (not a re-inference), re-run parse → decision → fill → P&L → benchmark and assert byte-identical results.
- This requires §B.7 to store `raw_model_output` (or its hash). Without it you can't separate "the model produced X" from "our math computed Y."
- **Mutation:** alter the parse/fill/P&L code → replay diverges from the stored result (suite goes red).

## 2. In-sample / out-of-sample separation
**Property:** any optimized scanner weights are frozen and evaluated only out-of-sample.
**Tests:**
- Assert the weights used in a scored run match a **frozen, version-tagged** weight set (hash check), not a live-fitted one.
- **Negative:** point the run at in-sample-fitted weights → an integrity assertion fails (the run must refuse, or at minimum flag the result non-graduating).
- Assert train/test date ranges **do not overlap** for any evaluated window.

## 3. Fill & slippage model (the single biggest lever)
**Property:** simulated fills are realistic, capped, and stress-tested.
**Tests:**
- **Fill rule:** assert fills occur per the frozen rule (e.g. next bar open / next-bar VWAP) — never at the decision-bar close (that's a hidden look-ahead).
- **Slippage applied:** assert realized fill price = reference ± slippage(bps) in the correct direction (buys worse, sells worse). **Negative:** zero the slippage → P&L must change (proves it was actually applied).
- **Volume cap:** assert no fill exceeds the configured fraction of bar volume; oversized orders partial-fill or roll. **Negative:** order > cap that fills fully → test fails.
- **Sensitivity sweep:** run the same trade tape at 1× / 2× / 3× slippage; assert the report emits all three and that **aggregate** P&L is **monotonically non-increasing** as slippage rises. *Per-trade* monotonicity can legitimately break under partial fills or trades skipped by the volume cap at higher slippage — assert at the portfolio level, and assert skipped/partial trades are accounted for, not dropped. An edge that survives only at 1× is flagged.
- **Liquid-universe guard:** assert every traded symbol meets the min-liquidity threshold on its decision date.

## 4. Benchmark (horizon-matched, honest)
**Property:** buy-and-hold comparison is apples-to-apples.
**Tests:**
- **Horizon match:** assert the benchmark holds over the **same period** as the strategy position it's compared against (no comparing a 3-day trade to a 3-month hold).
- **Dividends + cost:** assert total-return benchmark includes dividends and is **net of the same cost/slippage model** as the strategy.
- **Sizing/cash drag:** assert benchmark allocation and any uninvested-cash treatment match the stated rule; assert start point (candidate timestamp vs inception) is the frozen one.
- **Negative:** strip dividends or costs from the benchmark → the excess-return number must move (proves they're in the calc).

## 5. P&L calculation
**Property:** realized/unrealized P&L is arithmetically correct.
**Tests:**
- **Golden fixtures (long-only):** ≥3 hand-computed scenarios — a winning long, a losing long, a partial fill with slippage — where expected P&L is known on paper; assert to the cent. *No short fixtures: the MVP is **long-only**, SELL = exit a held long (never open a short).* A SELL on a position not held is a no-op (assert it).
- **Closure fixtures (CC#5):** a long closed by **target**, by **stop**, and by **time** — assert `exit_price`/`exit_reason`/realized P&L per the frozen exit rule, including the intrabar **stop-vs-target priority** when a bar's range spans both.
- **Decimal discipline:** assert money math uses Decimal (no float drift); a fixture that would drift under float must still match exactly.
- **Mutation:** flip a sign or drop the slippage term → suite goes red.
- **Reconciliation:** assert Σ(per-trade P&L) = portfolio-level P&L for any run (no leakage between the two views).

## 6. Survivorship / delisting
**Property:** delisted, halted, or renamed tickers are present in the historical universe — not silently dropped (the classic edge-inflator).
**Tests:**
- Assert the historical universe on a past date **includes** names later delisted (use a known delisted ticker as a fixture).
- Assert a position in a ticker that delists mid-hold is resolved (terminal value / removal proceeds), not vanished.
- **Negative:** a universe builder that pulls only *currently-listed* names → test fails.

## 7. Input validation (the TRI-57 class) — verify v0.3.0's vendor guards; own the scanner side
**Property:** garbage data never becomes a confident routed signal.
**Updated (Arbiter, post-TRI-66, 2026-07-02):** the engine is now on `vendor/TradingAgents` **v0.3.0** (TRI-66 Done, zero-mod, Arbiter-signed), which **ships stale-OHLCV rejection + a verified data-access contract + typed `VendorError` + no silent fallback.** The earlier premise ("our wrapper must do it because the vendor lacks the fix") is **superseded** — the engine-side job flips from *build* to **verify**. Two distinct surfaces:

**7a — Engine data (OHLCV/news): now the vendor's job → VERIFY it actually fires.**
- Feed NaN / missing / **stale** (old-timestamp) OHLCV through the engine's data path; assert v0.3.0 **rejects / raises `VendorError`** — no silent fallback, no confident decision on stale data.
- Define **per-field** expectations (missing price vs missing volume vs stale timestamp). **Mutation:** if a garbage bar reaches a confident signal, the suite goes red. Add a *wrapper* guard **only** for a case v0.3.0 misses — documented as a gap, not a default.

**7b — Scanner side (TRI-57/58/59): still OUR code (separate repo, NO vendor, upgrade did not touch it).**
- The scanner (`trifecta-market-scanner`) has no vendor and feeds candidates to the engine. Its garbage-in bugs — NaN/out-of-range detector score → 100-score candidate (**TRI-58**), missing ticker → `UNKNOWN` misroute (**TRI-59**), invalid OHLCV → confident candidate (**TRI-57**) — remain ours.
- Assert the scanner rejects / bounds-checks these; **Mutation:** disable the guard → a garbage candidate is minted with a confident score and the suite goes red.
- **These gate the daily loop** the moment the scanner feeds it — garbage scored confident corrupts the experiment (§B non-negotiable).

## 8. Decision/trade schema integrity
**Property:** the loop's data contract (§B.7) is complete, trade-linked, and append-only.
**Tests:**
- Assert every record carries the full §B.7 field set — including the **provenance** fields (`config_id`, `config_hash`, `universe_version`, `scanner_weights_hash`, `model_version`, `prompt_hash`, `asof_cutoff`, `data_vendor`), the **trade** fields (`trade_id`, `side`, `size`, `entry_timestamp/price`, `exit_timestamp/price/reason`, `stop`, `target`), `raw_model_output`(or hash), and `scanner_only_decision`; reject partial writes.
- Assert `trade_id` links an entry to exactly one exit; assert closed-trade stats (hit rate, realized P&L, the ≥30 count) are computable from the log alone.
- **Append-only by mechanism, not convention:** enforce via **hash-chaining** — each row stores `prev_row_hash`; a verifier recomputes the chain and any mutation/deletion breaks it. (DB-constraint/immutable-storage is an acceptable equivalent; "application logic only" is not.) **Mutation:** edit a past row → chain verification fails.
- Assert `input_snapshot_id` resolves to a retrievable snapshot (so #1a/#1b replay is always possible).

## 9. Run-completeness / heartbeat (honest = complete) — dual-reported
**Property:** silent failures can't create non-random gaps that bias the record.
**Tests:**
- Assert every scheduled run (per config) writes a **succeeded-or-failed** heartbeat; a crashed/timed-out run is recorded as **failed**, not absent.
- **Market-calendar classification:** a **holiday/half-day** (no expected trading) is classified differently from an **engine/data failure**. Assert the report distinguishes them — a holiday is not a "failure," and a failure is not silently a holiday.
- **Dual reporting, never silent exclusion (CX#9):** the report shows **both** (a) scored-days-only performance **and** (b) full calendar-period performance with missed/failed days visible. Assert both are emitted.
- **Mutation:** simulate Ollama-down / data-gap → the day is logged `failed` and surfaced; break the heartbeat writer → the day silently vanishes and the suite goes red.

## 10. Model-selection leakage — LLM training cutoff (CC#2)
**Property:** model selection isn't biased by the model's own memory of past outcomes.
*Selecting a local model on "decision quality over a historical sample" is contaminated: a model trained through 2024/2025 may recall that a name went up on a past date — rewarding memorization, not reasoning. (The forward daily paper run is immune — the future isn't in any training set — but the pre-run bench isn't.)*
**Tests:**
- Assert any historical model-selection/eval sample is dated **after** each candidate model's training cutoff, **or** that selection is validated **forward on paper** (the live parallel run already satisfies this).
- **Nested split (CX#6):** model-selection set / untouched evaluation set / then the frozen live track record — assert no overlap, and that the live record's model was fixed before the live record began.
- **Mutation:** point selection at a pre-cutoff sample → the leakage guard flags it.

## 11. Exit rule / trade closure (CC#5)
**Property:** positions close by a frozen, deterministic rule, so closed-trade stats exist.
**Tests:**
- Assert the frozen exit rule (target / stop / time-cap) is applied; assert the **intrabar stop-vs-target priority** is the frozen one when a bar spans both (conservative default: assume the **adverse** fill — stop — fires first unless proven otherwise).
- Assert a position with no exit signal closes at the **time-cap** (no immortal positions skewing unrealized P&L).
- **Mutation:** flip the stop/target priority → closure fixtures (§5) diverge and the suite goes red.

---

## Definition of done (§B code)
A §B component is "done" only when: its negative test passes (guard rejects bad input) **and** its **mutation** is demonstrated to fail the suite **and** the Arbiter has independently re-verified that the mutation actually red-lights. Until then, no number it produces feeds the §D graduation decision.

## Gating order (CX) — what blocks the run vs what blocks the verdict
**Required BEFORE the first scored run** (a bug here poisons the record from day 1): #1a look-ahead guard, #2 in-sample/out-of-sample, #6 survivorship, #7 input validation, #8 schema + append-only, #9 heartbeat/completeness, #10 selection leakage, #11 exit rule. These are the **Day-0 gate** (plan §B.0).
**Required BEFORE interpreting final results** (needed to trust the number, not to keep an honest record): #1b deterministic replay, #3 fill/slippage sweep, #4 benchmark, #5 P&L fixtures. These must be green before any §D graduation call, but the record stays valid while they're built.

**Staffing reality (CC#9):** 11 components × paired negative + mutation tests is real work — confirm a builder (engine/scanner owner) and a **separate** verifier (Arbiter) before treating §B as schedulable.

*Companion to `TRIFECTA_MVP_PLAN.md`. Implement under the normal gate, but §B is the one part of the MVP that does **not** get a light-touch pass.*
