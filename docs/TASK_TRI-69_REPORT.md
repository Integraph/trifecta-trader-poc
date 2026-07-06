# TASK TRI-69 — Lightweight Edge Check (pre-registered) · DEVELOP Report

> **⚡ LIVE STATUS** (updated at each completed step)
>
> | Step | Status |
> |---|---|
> | 1. Build Config A (temp=0 all three slots) | ✅ DONE — plumbing + tests + live API-level identity checks |
> | 2. Point-in-time / no-leak mode | ✅ DONE — module + 8 tests + live leak proof (PIT off = leak shown real) |
> | 3. Determinism battery (burned points) | 🔄 RUNNING — N=3 × AAPL/NVDA/TSLA @ 2026-06-27 |
> | 4. Pre-registration commit | ⬜ pending (BEFORE any eval run) |
> | 5. Eval runs | ⬜ pending |
> | 6. Scoring | ⬜ pending |
> | 7. Verdict | ⬜ pending |
>
> Checkpoint triggers fired: **TWO.**
> **Checkpoint 1 (RESOLVED in design window):** temp=0-all-slots wedges the local thinking model; amended to seed-pinned model-default sampling. Bar unchanged.
> **Checkpoint 2 (🔴 OPEN — DETERMINISM FAILURE, eval track STOPPED, 2026-07-06):** after eliminating every fixable nondeterminism channel, the arm remains stochastically unstable at the decision level (AAPL burned point: SELL/HOLD/SELL/SELL across 4 post-fix runs, agreement 0.75). The instability is STACK-INHERENT (evidence below). The pre-registered battery bar (decision-identical N=3 ×3 tickers) is unattainable for this architecture; per the locked rule — "do not run the backtest with a non-deterministic arm; STOP and surface" — the eval is NOT being run. Decision options for the App Manager are written up below. **No eval data exists; the design window is still open.**

**Issue:** TRI-69 · **Branch:** `jeff/tri-69-edge-check` (from TRI-70 tip 769be7d) · **Spec:** `docs/TASK_TRI-69_EDGE_CHECK.md` (v3, LOCKED) · **Integrity doc:** `docs/TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md` · **Paper only; --execute forbidden.**

---

## Pre-registration (LOCKED at the commit that introduces this section's final text — BEFORE any eval run)

**Nothing in this section may change after that commit. Deviations only via the checkpoint triggers, documented prominently.**

### Signal arm — Config A (TEST-ONLY, never production) — AS AMENDED under checkpoint 1

`tri69_config_a`: tool = `ollama/qwen3-coder:30b` · quick = `ollama/qwen3.5:9b` · deep = `anthropic/claude-sonnet-4-5-20250929` · **deep slot temperature = 0.0; LOCAL slots at model-default sampling with pinned `seed=69` and a `max_tokens=16384` runaway guard** · enhancements off · analyst cache off (`--no-cache`) · fresh memory log per run · point-in-time mode on (`TRIFECTA_POINT_IN_TIME=1`). One run per (ticker, date); decision extracted by the existing `extract_decision_detailed` (unchanged from TRI-70).

**Design-window amendment (2026-07-06, BEFORE this pre-registration commit):** the v3 work order pinned temperature=0 on all three slots *as a means to* decision-level determinism. Empirically that means kills the arm: greedy decoding wedges the local thinking model on content-dependent inputs (checkpoint 1 — 4 consecutive 3600 s timeouts across two tickers, one 47-minute HTTP 500 in the Ollama server log), while an otherwise-identical run at model-default sampling completed healthy in 901 s under full point-in-time conditions (`tri69-diag-default-temp`, HOLD). The amendment keeps the *end* (decision-identical repeats, verified — not assumed — by the battery, same bar, no loosening) and swaps the *means*: a pinned sampling seed on the local slots (direct-probe verified byte-identical across repeats on three prompt classes), temp=0 retained on the cloud judge (Anthropic exposes no seed; Sonnet@temp=0 was stability 1.00 in TRI-70). The qwen model card itself warns against greedy decoding for thinking models. Residual risk — seed reproducibility is server/hardware-local (fine: one box) — is accepted and disclosed; the determinism battery remains the gate.

### Universe (mechanical, as-of-date, no survivorship)

- **Membership source (reproducible):** S&P 500 constituents as of the earliest decision date (2026-02-17), from `github.com/fja05680/sp500`, commit `b792557e915703398ef9a67e4b583a37c6ec80d5`, file `S&P 500 Historical Components & Changes (Updated).csv`, row with max(date ≤ 2026-02-17) = **2026-02-09**, 503 tickers. The dataset includes later-delisted names (point-in-time by construction).
- **Subset rule (mechanical, seeded):** exclude {AAPL, NVDA, TSLA} (locked exclusions, all dates) → pool of 500 → sort lexicographically → **primary = `sorted(random.Random(69).sample(pool, 20))`** (CPython 3.13.11):
  `ALB, APA, AZO, CF, ERIE, EXPD, GPC, HBAN, IBKR, LLY, MO, ORLY, SHW, SMCI, SO, SYK, TRMB, TT, VRSK, VRTX`
- **Expansion set (run ONLY if the HOLD-dominance rule fires; drawn now so it can't be cherry-picked later):** `sorted(random.Random(6969).sample(pool_minus_primary, 10))`:
  `APH, APO, BMY, DOW, EXR, FIX, INVH, KMB, SWKS, WBD`
- No prior dev/benchmark run in this repo touched any of these 30 names (checked `results/`).
- Delisting/data-gap handling: a run that fails (no data) is recorded as FAILED and dual-reported, never silently dropped or replaced; a name delisting mid-horizon resolves at its last available close.

### Decision dates (8, non-overlapping T+10 windows, all realized, all post-training-cutoff 2026)

`2026-02-17 · 2026-03-03 · 2026-03-17 · 2026-03-31 · 2026-04-15 · 2026-04-29 · 2026-05-13 · 2026-05-28` — every 10th trading day from the first trading day ≥ 2026-02-17 (mechanical rule; verified against the SPY calendar). T+10 windows are exactly adjacent and disjoint (each entry Open(t+1) falls the day after the previous exit Close(t+10)); last exit 2026-06-11, fully realized before 2026-07-03. Dates postdate the training cutoffs of all three models (integrity doc #10); the excluded burned point 2026-06-27 is not in the set. **8 dates (up from an earlier 5-date draft, changed inside the design window before this commit) because the exact sign-flip test's power floor is 2^-D: with D=5 the minimum p is 1/32 ≈ 0.031 and PROMISING would require near-perfect 5/5 cross-date consistency — nearly unreachable by construction; with D=8 the floor is 1/256 ≈ 0.0039 and a 7-of-8-positive pattern of comparable magnitudes clears α=0.05 (p = 9/256 ≈ 0.035, verified in the fixture tests).**

**Primary sample: 20 tickers × 8 dates = 160 point-in-time decisions.**

### Horizon, fill, benchmark, cost

- **Primary horizon T+10** (accuracy tracker's completion horizon): entry at **Open(t+1)** (first trading day after decision date — integrity doc #3: never the decision bar), exit at **Close(t+10)** (10th trading day after t; frozen time-cap exit, no stop/target — integrity doc #11). T+5 (same entry, Close(t+5)) computed as descriptive secondary only.
- Prices: yfinance **auto-adjusted** (dividends included — integrity doc #4) for both legs.
- **Benchmark: SPY** over the identical window (the engine's own alpha benchmark — vendor `benchmark_map` default). Horizon-matched by construction. **Why SPY rather than same-ticker buy-and-hold:** a same-ticker comparator is vacuous for BUYs (a BUY *is* holding that ticker — excess would be identically −cost) — the signal's information content is name selection versus the investable market alternative; the integrity doc §4 pins horizon/dividends/cost/sizing of the benchmark, not the comparator asset, and SPY satisfies all four.
- **Cost + slippage: 10 bps per side → 20 bps round trip at 1×**, charged to the signal leg (the benchmark's symmetric round trip cancels; charging the leg is conservative). **Sensitivity sweep 1×/2×/3× reported; portfolio-level mean must be monotonically non-increasing** (integrity doc #3). Liquidity guard satisfied by construction (S&P 500 membership on the decision date).

### Statistic and test (cross-sectional dependence handled by DATE CLUSTERING)

- HOLD = no-trade → **excluded** from the directional test (rate reported).
- **Technical-failure rule (NO-DECISION):** a run that fails technically — decode loop, timeout (hard cap 3600 s/run), subprocess/API error, or decision extraction yielding UNKNOWN — is recorded as **NO-DECISION**: excluded from the directional test exactly like HOLD, **counted and dual-reported** (never silently dropped), and it feeds the expansion rule through the min directional-N gate. **No retries for model-behavior failures:** at temperature=0 those reproduce deterministically, so a retry is not an independent draw; the first result stands. **One mechanical exception — infrastructure connection failures:** a run whose stderr matches the frozen classifier `APIConnectionError|Connection error|Connection refused|ConnectError` (the local Ollama socket dropping — not a draw from any model, so retrying cannot bias the signal) gets exactly ONE retry under a `-retry` run-id; a second failure is NO-DECISION. Classifier and retry are code, not judgment (`scripts/run_tri69_edge_check.py`), frozen at this commit. *(Basis: the 2026-07-04 battery hit exactly this — `openai.APIConnectionError` mid-run from the quick slot while the model itself was healthy.)*
- Per directional decision i (BUY sᵢ=+1, SELL sᵢ=−1): **eᵢ = sᵢ·(r_ticker − r_SPY) − c_roundtrip**. SELL is scored as a prediction of SPY-relative underperformance — a measurement of the signal's information. **Long-only consistency (integrity doc §5):** the MVP is long-only and a SELL on an unheld name is a P&L no-op — so SELLs enter the *directional test* but not the *return arm*; the long-only implementable portfolio view (BUY → long net of cost; SELL/HOLD/NO-DECISION → cash) vs equal-weight buy-and-hold of the same names is computed and reported **descriptively only** (`long_only_view` in the scorer, fixture-tested), with no verdict weight.
- **Date-level statistic X_t = mean(eᵢ) over date t.** Effective N = number of dates.
- **Primary test: exact one-sided sign-flip permutation over dates** on mean(X_t) (all 2^D sign assignments; p = fraction of permuted means ≥ observed). **α = 0.05, one-sided.** **Power floor, stated up front:** with D=8 dates the minimum attainable p is 1/256 ≈ 0.0039; significance at α=0.05 requires roughly ≥7 of 8 dates positive at comparable magnitudes. The test still has power only against consistent edges — a real but weak/patchy edge will read INCONCLUSIVE, not PROMISING.
- **Secondary (vs chance):** directional hit rate (share of decisions with sᵢ·(r_ticker − r_SPY) > 0) against the **realized base rate p₀ = max(all-BUY accuracy, all-SELL accuracy)** on the same decisions — NOT 0.5. Reported with the same date-clustered machinery; no independent verdict weight.
- Implementation: `scripts/score_tri69.py`, frozen at the pre-registration commit; 13 golden-fixture tests (hand-computed P&L, cost-application negative test, sweep monotonicity, exact permutation values incl. the D=8 floor, long-only no-op SELL).

### Expected HOLD rate + power plan (basis stated, not a post-hoc knob)

- **Basis:** TRI-70 cloud-deep runs were modal-HOLD (sonnet-deep HOLD×3, opus-deep HOLD×3 on AAPL@2026-06-27); the all-local finalist was 9/15 HOLD (60%) across AAPL/NVDA/TSLA; TRI-69's own determinism battery (9 runs, 3 tickers, same burned date, Config A exactly) observed the HOLD rate recorded in Step 3 below. **Expected HOLD (incl. NO-DECISION) 60–85%; planning value 70%.**
- **Minimum directional N (pre-registered): ≥ 20 directional decisions AND ≥ 6 of 8 dates with ≥ 1 directional decision.** At 70% HOLD/NO-DECISION, 160 decisions → ~48 directional (meets comfortably); at 85% → ~24 (still meets); the gate protects against worse.
- **If the gate fails: INCONCLUSIVE-BY-HOLD → run the pre-registered expansion set (10 more names × same 8 dates = +80 runs) — an expansion, not a verdict.** If it still fails after expansion, report INCONCLUSIVE with the directional N needed.

### Decision rule (KILL-SWITCH — verbatim mapping)

| Outcome | Verdict |
|---|---|
| p ≤ 0.05 AND mean(X_t) > 0 at 1× cost AND mean(X_t) > 0 at 2× cost | **PROMISING — run the larger out-of-sample test** (never "confirmed edge") |
| mean(X_t) ≤ 0 AND hit rate ≤ p₀ AND min-N gate met | **STOP-PIVOT** (no edge / anti-predictive) |
| anything else | **INCONCLUSIVE** (+ directional N needed for power) |

### Guardrails

Paper/analysis-only (`--execute` forbidden; runner cannot emit it). Checkpoint triggers: unclosable leak · determinism failure · non-out-of-sample sample · HOLD-dominance blocking min-N after expansion. On any trigger: stop that track, surface prominently here.

---

## Build log

### Step 1 — Config A (TEST-ONLY): `tri69_config_a`

- **Slots (pinned by pre-reg):** tool = `ollama/qwen3-coder:30b` · quick = `ollama/qwen3.5:9b` · deep = `anthropic/claude-sonnet-4-5-20250929` · **temperature = 0.0 on all three.**
- **New plumbing:** `HybridLLMConfig` gained an optional `temperature` field (absent from YAML for legacy configs → byte-identical round-trip); `create_hybrid_llms` now threads it into **all three** `create_llm_client` calls (`src/hybrid_llm.py`). Both vendor client classes already forward `temperature` via `_PASSTHROUGH_KWARGS` (verified: `anthropic_client.py:9-12`, `openai_client.py:166-169` — the Ollama path is `OpenAIClient`, so nothing is silently dropped).
- **DEVELOP-latitude choice:** enhancement flags OFF (`enhance_local=False`, `enhance_deep=False`) — matches the TRI-70 `bench_deep_*` pattern that shares the same tool+quick slots (measure the raw pipeline; no prompt-prefix wrappers). Made before pre-registration commit.
- **Tests:** `tests/test_tri69_point_in_time.py::TestTemperaturePlumbing` — temp=0 reaches all three slot LLM objects; None leaves provider defaults (negative); YAML round-trip. 3/3 pass.
- Builder: `scripts/build_tri69_config.py` (idempotent).

### Step 2 — Point-in-time / no-leak mode (`src/point_in_time.py`, env-gated `TRIFECTA_POINT_IN_TIME=1`)

Vendor tree stays zero-mod; everything is patched at our layer via the vendor's `VENDOR_METHODS` registry + module namespaces.

Leak surfaces found (date-bounding audit of every analyst tool, 2026-07-04) and closures:

| # | Surface | Verdict | Closure |
|---|---|---|---|
| 1 | `get_fundamentals` (yfinance `Ticker.info` + alpha_vantage OVERVIEW) | 🔴 pre-registered leak — 52wk hi/lo, 50/200-day avg, TTM ratios, all today-relative | **Neutralized** → pointer message to date-bounded statement tools |
| 2 | `get_prediction_markets` (Polymarket) | 🔴 pre-registered — live odds, no point-in-time path | **Disabled** (stub message) |
| 3 | `get_insider_transactions` | 🔴 NEW (audit) — no date param at all; returns filings as of real now | **Disabled** (stub message) |
| 4 | Reddit + StockTwits pre-fetch in sentiment analyst (`sentiment_analyst.py:70-71`) | 🔴 live feeds, no date bound (the "live social" of the pre-reg) | **Disabled** — stubs patched into both dataflow modules AND the analyst's import namespace |
| 5 | Statement tools with `curr_date=None` | 🟠 vendor filter silently no-ops | **curr_date forced to as-of date** when omitted |
| 6 | Any date arg > as-of (hallucinated future window) | 🟠 defensive | **Generic clamp** on every vendor-routed method + `build_verified_market_snapshot` |
| — | `resolve_instrument_identity` (`Ticker.info`) | ✅ SAFE — extracts only name/sector/industry/exchange/quote_type; no price/mcap fields | none needed |
| — | get_stock_data / get_indicators / get_verified_market_snapshot / get_news / get_global_news / get_macro_indicators / statements-with-date | ✅ SAFE (date-bounded in vendor, verified with citations) | leak guard still scans outputs |

**Leak guard (fail-loud):** every vendor-routed output is scanned for raw info-keys (`fiftyTwoWeek*`, `fiftyDayAverage`, `twoHundredDayAverage`, `trailing*`); fundamentals-category outputs additionally for formatted labels (`52 Week High:` …). A hit raises `LookAheadLeakError` — the run dies rather than scoring a contaminated decision. News prose ("hit a 52-week high") does NOT false-positive (tested).

**Defense-in-depth:** eval/battery runs save the four analyst reports (`TRIFECTA_SAVE_REPORTS=1`) and the runner re-scans them post-hoc (`_leak_audit` in `scripts/run_tri69_edge_check.py`).

**Memory isolation:** every run gets a fresh `TRADINGAGENTS_MEMORY_LOG_PATH` — no cross-run reflection/decision contamination, no run-order dependence.

- **Tests:** 8/8 PIT tests pass (neutralization, disables, curr_date forcing, date clamp, leak-guard rejection [negative test], prose non-false-positive, analyst-namespace patch, idempotency).
- Live leak proof (PIT off → fields present; PIT on → absent): recorded below.

### Live leak proof (2026-07-04)

Same live vendor call (`route_to_vendor("get_fundamentals", "MSFT", "2026-03-13")`), PIT off vs on:

- **PIT OFF** — output contains ALL six forbidden formatted labels, with values that are unambiguously today-relative for a 2026-03-13 decision date: `PE Ratio (TTM): 23.257294`, `EPS (TTM): 16.79`, `52 Week High: 555.45`, `52 Week Low: 349.2`, `50 Day Average: 407.5954`, `200 Day Average: 445.4361`. The leak is real; the guard tests are not tautological.
- **PIT ON** — same call returns the neutralization message; zero forbidden info-keys, zero forbidden labels. `get_prediction_markets` and `get_insider_transactions` return their disabled-mode stubs.

### Live temp=0 behavior checks (pre-battery, 2026-07-04)

- **⚠️ Greedy-decoding loop risk found and characterized:** at temp=0 (greedy), `qwen3.5:9b` on a *creative* prompt ("one-sentence metaphor") never left the thinking channel — 1500-token cap hit with **empty content** (`finish_reason=length`); an uncapped invocation ran >15 min before being killed. This is the known qwen thinking-model greedy pathology. On a **realistic analytical prompt** (the engine's regime) it terminates normally (`finish_reason=stop`, 1170 tokens, 34 s). Judged NOT a design-window trigger: the engine's prompts are analytical; the determinism battery is the arbiter, and eval runs get a runaway watch (a looping call would show as an abnormal wall time). Recorded as a TRI-73 caveat: greedy decoding on qwen thinking models is prompt-fragile.
- **Repeat-identity at temp=0 (direct Ollama /v1 API):** `qwen3.5:9b` analytical prompt ×2 → byte-identical; `qwen3-coder:30b` tool-style prompt ×2 → byte-identical.
- Deep slot: `ChatAnthropic(temperature=0)` object confirmed; live Sonnet call OK (credits verified live at task start).

### Step 3 — Determinism battery

**Incident log (2026-07-04 → 07-05, corrected per reviewer session):** first battery launch lost ~29 h. AAPL r1 completed (SELL, 695 s, zero leak hits, $0.048). AAPL r2 failed with `openai.APIConnectionError` from the quick slot mid-run. During r3, the Ollama runner for `qwen3.5:9b` wedged — stuck in "Stopping…" at 100% GPU from ~16:28 Jul 4 — and `run_analysis` blocked indefinitely on the socket (no read timeout at our layer). **The SIGTERM was the reviewer session's deliberate emergency recovery at ~22:13 Jul 5:** the machine had drained to 7% battery with the GPU pegged; they killed the blocked processes and quit Ollama.app. Both Ollama models pass a health check after restart.
**Ops lessons (recorded for eval + future runs):** (1) `pkill ollama` is insufficient — the macOS app respawns the server; **quit Ollama.app** to actually stop it. (2) **Eval runs on AC power only** — a pegged GPU drains even an M3 Max in hours. (3) A wedged runner presents as an indefinitely-blocked client socket; the per-run hard timeout (below) is the guard at our layer.
**Hardening applied before relaunch (pre-commit window):** per-run hard timeout 3600 s; battery mode made resumable (completed run-ids are reused, so r1's SELL is not re-run); the mechanical infra-retry rule (see pre-registration) added to both modes. Determinism-bar semantics unchanged: the bar compares the decisions of the three COMPLETED runs per ticker — an infra failure yields no decision to compare and is completed via the retry, never counted as agreement.

### 🔴 CHECKPOINT 1 (2026-07-06, design window OPEN — no pre-reg commit yet): temp=0 Config A wedges on real inputs

**Evidence (chronological):**
1. Battery AAPL r1 (Jul 4, temp=0, PIT on): completed healthy — SELL, 695 s.
2. AAPL r2 (Jul 5): `openai.APIConnectionError` at the Aggressive Analyst (quick slot) during the Ollama-wedge incident.
3. AAPL r3, NVDA r1, NVDA r2 (Jul 5–6, post-recovery, on AC power): **all three hit the 3600 s hard timeout** — silent stall after the news-analyst stage (last stderr = benign FRED-key warning in every case).
4. Ollama server log: one `/v1/chat/completions` request ran **47m07s → HTTP 500**; surrounding normal calls run 20 s–2.4 m.
5. Direct-API probes: qwen3.5:9b thinking runs away on some prompts at temp=0 (empty content at token cap) — and probe evidence shows very long thinking at temp=0.7 too, so greedy-vs-sampling may not be the discriminating variable.
6. Battery halted after 4 consecutive wedges (config diagnosed non-viable; TSLA never started). No decision flip was ever observed — every completed run pair agreed; the failure mode is non-completion, not instability.

**Confound to resolve:** TRI-69 changed TWO things vs TRI-70's healthy 15-run qwen3.5:9b baseline (~16–26 min/run, zero wedges): (a) the temperature=0 pin, (b) point-in-time mode content (neutralized fundamentals, disabled social, clamped windows). A diagnostic engine run (AAPL @ 2026-06-27, **provider-default temperature — TRI-70 conditions — with PIT ON**, run-id `tri69-diag-default-temp`) is in flight to discriminate:
- Diag completes healthy → the temp=0 pin is the trigger → amend Config A to **seed-pinned sampling on the local slots** (repeat-identity at temp=0.7+seed verified by direct probe ×3 prompt classes) + deep Sonnet temp=0, determinism bar unchanged, verified by a fresh same-day battery.
- Diag also wedges → PIT content (or current news content) is the trigger → investigate content path before any amendment.

**Also fixed regardless:** a generation-length guard (max_tokens) on the local slots so a runaway fails in minutes, not hours; per-run 3600 s timeout already in place.

**RESOLUTION (2026-07-06):** the diagnostic run **completed healthy — 901 s, decision HOLD, quality 7.4, data-grounding 10, zero leak hits** — at provider-default temperature with PIT ON (`tri69-diag-default-temp`). PIT content exonerated; **the temperature=0 pin is the wedge trigger.** Config A amended per the design-window write-up in the pre-registration section (seed-pinned model-default sampling on local slots + temp=0 deep + 16384-token runaway guard). Direct probes confirm default-sampling+seed is byte-identical across repeats and terminates (`finish=stop`) on prompts where pinned-temp generation blew the token cap. Battery re-launching in full (fresh same-day trios, tag `tri69b`); Jul-4 stale runs excluded from the identity comparison (cross-day news drift breaks the identical-inputs premise — recorded as TRI-73 input).

*(first battery attempt — decisions observed: AAPL r1 SELL [Jul 4]; all later runs wedged; NO flip observed.)*

**Checkpoint 1 addendum — tri69b flip root-caused as an INPUT bug, fixed (2026-07-06, still pre-commit):** the amended-config battery (`tri69b`) opened AAPL SELL (887 s) then HOLD (773 s) — a flip. Saved analyst reports localized it precisely: **sentiment and news reports byte-identical** (seeded sampling reproduces perfectly when the context is identical); **market and fundamentals reports diverged from the first generated character** — the signature of a differing prompt token, not sampling. Root cause: the vendor stamps tool outputs with wall-clock retrieval time (`# Data retrieved on: <now>`, six sites in `y_finance.py`); runs minutes apart see different contexts and seeded generation diverges from that token onward. Verified live: two identical PIT calls 2 s apart differed only in that stamp; after adding stamp normalization to the point-in-time wrapper, the same double-call is byte-identical. The flip was therefore **not** model instability and **not** a sampling failure — it was a reproducibility bug in our measurement rig, now closed at the same layer as the leak guards (test added). Battery restarted clean as `tri69c`; `tri69b` runs excluded from identity adjudication (contaminated contexts) but retained as HOLD-basis observations.

**tri69c result (2026-07-06): flip persists with fully deterministic upstream — divergence is DOWNSTREAM of the analysts.** AAPL r1 = SELL (813 s), r2 = HOLD (879 s), and this time **all four analyst reports are byte-identical** (the stamp fix held). The trader plans differ from the first token ("Sell" vs "Hold" reasoning), so divergence enters between the researchers and the trader. Everything before the Research Manager is local+seeded and has now been proven byte-reproducible twice; **the RM is the first cloud (Sonnet temp=0) call in the chain** — Anthropic does not guarantee bit-identical output at temp=0, and a single token of drift cascades through debate → trader → risk → judge into a different decision. Battery stopped (bar already failed for AAPL); two instrumented repeats (`tri69-diag2-r1/r2`, saving bull/bear histories, RM text, risk histories, judge text) were run to localize the first diverging stage with evidence rather than inference.

### 🔴 CHECKPOINT 2 (2026-07-06) — determinism failure is stack-inherent; eval track STOPPED

**Post-fix stability tally, AAPL @ 2026-06-27 (identical config, same day, all channels fixed):** `tri69c-r1` SELL · `tri69c-r2` HOLD · `diag2-r1` SELL · `diag2-r2` SELL → **agreement 0.75, decision-identical bar FAILED.**

**Why this is not fixable at our layer — the elimination chain:**
1. Temperature pathology: eliminated (checkpoint 1 — seed-pinned model-default sampling; healthy runs).
2. Wall-clock context contamination: eliminated (stamp normalization; verified byte-identical double-calls; tri69c analyst reports byte-identical).
3. Input-data drift: eliminated for the diverging pair (diag2 market reports share the same data — 11 common numerals, 1 unique each — different *prose*, not different *facts*).
4. What remains: **the first diverging stage moves between repeats** — in the tri69c pair all four analyst reports were byte-identical and divergence entered in the debate chain; in the diag2 pair the market analyst itself diverged from early tokens while three other analysts were byte-identical. Identical requests are therefore not reliably bit-reproducible on this stack even with a fixed seed. This matches known llama.cpp/Ollama behavior (floating-point non-associativity across batch splits / kv-cache states makes logits state-dependent; seed determinism is only guaranteed for an identical server state), compounded by Anthropic temp=0 being explicitly best-effort. One drifted token anywhere cascades through the 10-stage agent chain into a different final decision ~25% of the time on this point.

**Re-attribution of TRI-70's 0.60 (TRI-73 input, the mandated "freebie" observation):** TRI-70's local decision instability was attributed to unpinned sampling temperature. TRI-69's battery shows that even *seed-pinned* local slots at model-default sampling do not hold still on long multi-agent contexts — a material share of the 0.60 was likely never fixable by temperature pinning. Any future local arm (TRI-73) inherits this: **decision-level determinism is not achievable on the current Ollama stack; stability must be treated statistically (repeats), not assumed.**

**Options (decision owner: App Manager — this changes protocol/budget/bar, not mechanics):**
| # | Protocol | Cost/time | Trade-off |
|---|---|---|---|
| 1 | **Modal-of-3**: pre-register N=3 repeats per (ticker,date); decision = modal; 3-way split → NO-DECISION | 480 runs ≈ ~4.7 days serial, ~$35–55 | Statistically cleaner signal; ~3× budget/time; modal itself still ~15% unstable at per-run agreement 0.75 |
| 2 | **Single-draw stochastic-policy framing**: keep 1 run/point; pre-register that each decision is ONE draw from a stochastic policy — exactly how production would run it (production never takes modal-of-3). The date-clustered directional test stays unbiased; sampling noise costs power, not validity | 160 runs ≈ ~1.5 days, ~$15–25 (unchanged) | Cheapest and most ecologically valid; measures "policy + sampling noise"; a weak edge is harder to detect (noise → INCONCLUSIVE more likely) |
| 3 | **Stop TRI-69** pending an engine/stack change that restores determinism (vLLM deterministic mode, all-cloud arm, etc.) | — | Blocks the go/no-go on infrastructure work of unknown size |

**DEVELOP recommendation: option 2.** The determinism requirement existed to *justify* 1-run-per-point; a pre-registered stochastic-policy framing achieves the same honesty without pretending the arm is something it is not, matches production semantics, and keeps the kill-switch on budget. Option 1 is the fallback if the App Manager wants decision-level stability inside the measurement itself.

**Track state:** eval NOT started; no eval data drawn; pre-registration NOT committed; awaiting App Manager call on options 1/2/3.

---

## Results

*(pending — nothing here until after the pre-registration commit)*

## Compute / $ spent

*(running total; see result JSONs)*

## Verdict

*(pending)*
