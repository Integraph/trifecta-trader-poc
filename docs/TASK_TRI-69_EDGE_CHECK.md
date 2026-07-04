# TRI-69 — Lightweight Edge Check (pre-registered) · DEVELOP Work Order · v3

**Issue:** TRI-69 (High) · **Stage:** DEVELOP → QA → UAT → Arbiter · **Depends on:** TRI-66 (Done) · Uses TRI-70
**Pre-registration owner:** Engine App Manager (locks the design + decision rule below). **Execution:** DEVELOP. **Report to:** `docs/TASK_TRI-69_REPORT.md`. **Paper only; no live trading.**

> **This is the project go/no-go:** does the engine's decision predict next-horizon direction better than horizon-matched buy-and-hold, **net of cost**? It is a **KILL-SWITCH, not a green-light** — a small backtest can *falsify* the thesis, but a strong result is only *"promising, run a bigger test,"* never *"we have edge."*
>
> **v2 (after DEVELOP review, verified against the vendor tree):** 🔴 fixed a real **look-ahead leak** — `get_fundamentals()` pulls today's `Ticker.info` (52-week hi/lo, 50/200-day averages = future-relative) regardless of `curr_date`, so "look-ahead-safe by construction" was FALSE; the point-in-time mode must neutralize it. 🔴 fixed the **statistics** (cross-sectional dependence + non-50% base rate). Pinned the judge to **Sonnet + temp=0** (Opus rejects `temperature` → can't be made deterministic; also lower quality/slower). Added **HOLD-dominance power planning**, a stronger **determinism proof**, and **mechanical as-of-date universe selection** (survivorship).
>
> **v3 (after 2nd DEVELOP review):** 🟠 **temp=0 on the judge alone doesn't make the arm deterministic** — the local tool/quick slots sample at Ollama's ~0.7–0.8 default, so the judge can flip even at temp=0 and the determinism proof would fail as designed. Now pins **temp=0 on all three slots** and flags the **new plumbing** required (`create_hybrid_llms` doesn't thread temperature today; `DEFAULT_CONFIG["temperature"]=None`). Also pinned the last free knobs: **`reasoning_quick = qwen3.5:9b`**, a **named reproducible membership source** + a **mechanical subset rule** (top-N by as-of-date mcap or seeded sample — no hand-picking), and a **stated basis for the expected-HOLD estimate** (not a post-hoc knob).

---

## Pre-registration (LOCKED before any eval run — do not change after seeing results)

**Method:** a historical, **out-of-sample, point-in-time backtest** on past dates whose forward returns are realized. Runnable now (~1–2 days). NOT the forward/live-condition stability track (that's later).

**Signal arm — Config A, pinned (TEST-ONLY, never production by default):** `temperature=0` on **all three slots** — local tools `qwen3-coder:30b` + local `reasoning_quick = qwen3.5:9b` + **cloud deep Risk Judge = `claude-sonnet-4-5-20250929`**. Sonnet, not Opus: TRI-70 measured sonnet-deep **8.7** vs opus-4-8 **7.3** (both stability 1.00), sonnet ~17 min vs opus ~30 min, and **opus-4-8 rejects the `temperature` param** (TRI-70 engine-compat note) — so Opus can't be pinned to temp=0 at all.

- **🔴 Why all three slots, not just the judge (else determinism fails by design):** temp=0 on the deep slot alone does **not** make the arm deterministic. The judge's inputs come from the local tool/quick/debater slots, which run at Ollama's **default temperature (~0.7–0.8)** unless pinned; if their narratives shift run-to-run the judge can legitimately flip **even at temp=0**. TRI-70's local **0.60** agreement is direct evidence the upstream stages are decision-relevantly stochastic. So determinism is **NOT "by construction"** — it must be *verified* by the proof (Step 4), and that proof only passes if all three slots are pinned.
- **Plumbing (flag for DEVELOP — this is new code, not just a config value):** the engine does **not** pass temperature today — `DEFAULT_CONFIG["temperature"]=None` and `create_hybrid_llms` (`src/hybrid_llm.py:382/388/399`) threads only `provider`+`model` into `create_llm_client`. Pinning temp=0 means threading `temperature=0` through **all three** `create_llm_client` calls **and verifying each client's `get_llm()` actually applies it** — especially the Ollama/OpenAI-compatible path (confirm it isn't silently dropped). Opus is out of the arm, so a global temp=0 carries **no 400-rejection risk**.

**Point-in-time inputs — look-ahead-safe (must be ENFORCED, not assumed):**
- OHLCV up to the decision date + as-of-date news windows (v0.3.0's contract works here).
- Date-bounded financial **statements** (`get_balance_sheet`/`get_cashflow` use `filter_financials_by_date` ✓).
- **🔴 Neutralize the fundamentals OVERVIEW:** `get_fundamentals()` ignores `curr_date` and returns today's `Ticker.info` (52-week hi/lo, 50/200-day averages, TTM ratios = **future leakage**). In backtest mode, **drop that tool** (or restrict it to the date-filtered statements). Verify **no** `fiftyTwoWeek*/fiftyDayAverage/twoHundredDayAverage/trailing*` field reaches the analyst context.
- **Disable live Reddit/Polymarket** (both a leak and the input-drift confound).

**Out-of-sample universe — selected MECHANICALLY as-of-date (no survivorship):** index membership (e.g. S&P 500) **as of the earliest decision date** per the integrity doc's survivorship rule — **not** hand-picking today's well-known liquid names (that preselects survivors).
- **Membership source (name it — must be reproducible):** point-in-time S&P 500 constituents are **not in yfinance**; DEVELOP must state and cite a reproducible as-of-date source (an archived constituent list, an index-ETF holdings snapshot as-of-date, or whatever `TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md` prescribes). "Mechanical" isn't mechanical if the pool can't be reconstructed.
- **Subset rule (mechanical, not hand-picked):** fixing the ~500-name pool isn't enough — pre-register **how** the ~20–30 eval names are drawn from it: **top-N by as-of-date market cap**, or a **seeded random sample (state the seed)**. No discretionary picking sneaks in through the subset.
- **Exclude AAPL / NVDA / TSLA at ALL dates** (the whole engine was developed staring at AAPL output). ~20–30 names × ~3–5 non-overlapping historical dates; **list the final set in the report before running.**

**Horizon + benchmark:** score at the engine's signal horizon (align to the accuracy tracker's T+5 / T+10). **Primary test = vs the horizon-matched buy-and-hold benchmark, net of cost + slippage** — per `docs/TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md` (fill model, benchmark, survivorship, exit rule). *"Vs chance"* is **secondary**, and its null is the **realized base rate** (equity drift ≠ 50/50), not 0.5.

**Statistics — handle cross-sectional dependence:** ~25 large-caps on one date are largely **one macro bet** (correlated T+N directions). A naive binomial over ~100 decisions treats them as independent and **overstates significance both ways** (a lucky macro week reads as edge; an unlucky one falsely trips the kill-switch). **Effective N ≈ number of dates, not decisions.** Pre-register the dependence handling: **cluster by date** — a date-level sign/permutation test, or score **date-level portfolio excess return** vs the benchmark. State α up front.

**HOLD-dominance power plan:** TRI-70's cloud runs were **modal HOLD** (sonnet/opus HOLD×3). `HOLD` = no-trade, excluded from the directional test. If the engine HOLDs 60–80% of ordinary points, ~100 decisions yield only ~20–40 directional trials → **underpowered by construction.** Pre-register: the **expected HOLD rate** (estimate it from TRI-70's cloud runs + the burned-point determinism runs, and **state that basis** so the number isn't a post-hoc knob), the **minimum directional N** for the stated α/power, and the rule: **INCONCLUSIVE-by-HOLD → expand the universe/dates, not a verdict.** Size the universe to hit the required directional N.

## Decision rule (KILL-SWITCH — pre-registered)
- **No edge / anti-predictive** (≤ benchmark net of cost; and ≤ realized base rate) → **STOP / PIVOT.**
- **Clearly & significantly beats the benchmark** (date-clustered test) → **"PROMISING — run the larger out-of-sample test."** Never "confirmed edge."
- **Ambiguous / underpowered** → **INCONCLUSIVE**; state the directional N needed for power.

## Steps
1. **Pre-register (App Manager locks the rules above; DEVELOP fills the mechanical universe).** Write the fixed universe, dates, horizon, benchmark, dependence handling, α, expected-HOLD, min-directional-N, and the decision rule into `docs/TASK_TRI-69_REPORT.md` and **commit it as its own commit BEFORE any eval run** — the git timestamp proves the criteria preceded the data.
2. **Build Config A** (test-only): `tool_provider: ollama` (`qwen3-coder:30b`) + `reasoning_quick = ollama qwen3.5:9b` + `reasoning_deep = anthropic claude-sonnet-4-5-20250929`, **`temperature=0` on all three slots.** This needs **new plumbing** — `create_hybrid_llms` doesn't thread temperature today (see the Signal-arm note); add it and confirm each client's `get_llm()` applies it. Never selectable as production.
3. **Point-in-time / no-leak mode:** `--date` freezes OHLCV + news; **add: disable live Reddit/Polymarket AND neutralize `get_fundamentals()`'s future-relative fields.** Prove no leaked fields in the context.
4. **Determinism proof (strong — tests the WHOLE pipeline, not just the judge):** N=3 runs on the three **already-burned TRI-70 finalist points** — **`AAPL` / `NVDA` / `TSLA` @ `2026-06-27`** (all ran at N=5 in TRI-70; all three are the pre-registered exclusion set, so none touches the eval universe) — must give **decision-identical** results. This is the check that temp=0 is *actually* plumbed on **all three** slots: if the local slots still sample, it fails here, exactly as TRI-70's 0.60 predicts. **The bar is decision-level identity — a single flip fires the checkpoint; do NOT quietly loosen it to "2 of 3."** (Neither Ollama nor the Anthropic API guarantees bit-identical text even at temp=0; greedy decoding on one box is near-deterministic in practice, so the bar is correctly the *decision*, not the bytes.) **If it fails, STOP and surface** — do not run the backtest with a non-deterministic arm.
5. **Run the backtest** — one point-in-time run per `(ticker,date)` (deterministic → no repeats), run-id'd result files.
6. **Score** per the integrity doc: date-clustered directional test vs the horizon-matched net-of-cost benchmark; report hit-rate, per-date portfolio excess return, HOLD rate, directional N, the test statistic + α, and the small-sample power caveat.
7. **Go/no-go verdict** under the pre-registered kill-switch rule.

## Exit criteria
1. Config A built (`test-only`, **temp=0 on all three slots** — tool/quick/Sonnet-deep — via the new `hybrid_llm.py` plumbing). Point-in-time mode **verified leak-free** (no future-relative fundamentals fields; live social off) and **deterministic** (N=3 on the excluded `AAPL`/`NVDA`/`TSLA` @ `2026-06-27` points → decision-identical — this is what proves all-slot temp=0 took).
2. Pre-registration **committed before results** (own commit), genuinely out-of-sample, mechanically-selected as-of-date universe (no AAPL/NVDA/TSLA any date).
3. Backtest run over the stated universe/dates; results run-id'd.
4. Scored per `docs/TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md`; **date-clustered** test vs horizon-matched net-of-cost benchmark; base rate ≠ 0.5.
5. HOLD rate + directional N reported; if underpowered, the expansion rule applied (not a false verdict).
6. A **go/no-go verdict** (STOP-PIVOT · PROMISING-run-bigger · INCONCLUSIVE) with the honest power caveat and compute/$ spent.

## Budget (compute-time is the constraint)
- Sonnet-deep, deterministic → 1 run per `(ticker,date)`, ~**17 min/run**. ~100 decisions ≈ **~29 h serial**; cost ≈ **$10–30**. If HOLD-dominance forces a larger universe, restate.

## Prereqs & gate
- **Copy `AI/TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md` into `docs/`** (a repo-rooted DEVELOP session won't resolve an `AI/` path) — same treatment as `ECOSYSTEM_CONTEXT.md`. Exit-4's discipline hangs on it.
- Paper only; no live. Config A is test-only, never production by default (edge-with-A-but-not-local → Jeff's call).
- **Checkpoint — surface before committing around it:** if the fundamentals/live-social leak can't be cleanly closed, if determinism fails, if the sample can't be made genuinely out-of-sample, or if HOLD-dominance can't reach the min directional N. Integrity blockers, not workarounds.
- DEVELOP → QA (Codex) → UAT → Arbiter. App Manager does not declare Done.
