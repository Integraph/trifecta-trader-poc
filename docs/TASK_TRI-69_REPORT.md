# TASK TRI-69 — Lightweight Edge Check (pre-registered) · DEVELOP Report

> **⚡ LIVE STATUS** (updated at each completed step)
>
> | Step | Status |
> |---|---|
> | 1. Build Config A | ✅ DONE (as amended: seed-pinned local sampling + temp=0 deep — checkpoints 1+2) |
> | 2. Point-in-time / no-leak mode | ✅ DONE — module + tests + live leak proof + stamp normalization |
> | 3. Determinism battery (burned points) | ✅ CLOSED — bar failed for stack-inherent reasons (checkpoint 2); protocol amended to single-draw stochastic policy per App Manager decision 2026-07-06 |
> | 4. Pre-registration commit | ✅ `ebca159` (2026-07-06, before any eval data) |
> | 5. Eval runs (160 = 20 names × 8 dates, single draw each) | ✅ COMPLETE 2026-07-08 — 160/160, 0 errors, 0 NO-DECISION, 0 leaks, 0 interventions, $8.79 |
> | 6. Scoring (frozen scorer only) | ✅ DONE — `results/tri69/scoring.json` (copies committed in docs/) |
> | 7. Verdict | ✅ **INCONCLUSIVE — no detectable edge at this power** (see Verdict) · ready for Codex QA |
>
> Checkpoint triggers fired: **TWO.**
> **Checkpoint 1 (RESOLVED in design window):** temp=0-all-slots wedges the local thinking model; amended to seed-pinned model-default sampling. Bar unchanged.
> **Checkpoint 2 (RESOLVED by protocol amendment — App Manager decision 2026-07-06, Option 2):** after eliminating every fixable nondeterminism channel, the arm remained stochastically unstable at the decision level (AAPL burned point: SELL/HOLD/SELL/SELL, agreement 0.75) for stack-inherent reasons. Eval was stopped pre-data and the options surfaced; DEVELOP recommendation, reviewer concurrence, and App Manager sign-off aligned on the **single-draw stochastic-policy protocol** (written into the pre-registration — see "Protocol amendment 2"). Not a loosening: the framing, noise level, and directional-trial power caveat are all pre-registered before any eval data.

**Issue:** TRI-69 · **Branch:** `jeff/tri-69-edge-check` (from TRI-70 tip 769be7d) · **Spec:** `docs/TASK_TRI-69_EDGE_CHECK.md` (v3, LOCKED) · **Integrity doc:** `docs/TRIFECTA_MVP_MEASUREMENT_INTEGRITY_TESTS.md` · **Paper only; --execute forbidden.**

---

## Pre-registration (LOCKED at the commit that introduces this section's final text — BEFORE any eval run)

**Nothing in this section may change after that commit. Deviations only via the checkpoint triggers, documented prominently.**

### Signal arm — Config A (TEST-ONLY, never production) — AS AMENDED under checkpoint 1

`tri69_config_a`: tool = `ollama/qwen3-coder:30b` · quick = `ollama/qwen3.5:9b` · deep = `anthropic/claude-sonnet-4-5-20250929` · **deep slot temperature = 0.0; LOCAL slots at model-default sampling with pinned `seed=69` and a `max_tokens=16384` runaway guard** · enhancements off · analyst cache off (`--no-cache`) · fresh memory log per run · point-in-time mode on (`TRIFECTA_POINT_IN_TIME=1`). One run per (ticker, date); decision extracted by the existing `extract_decision_detailed` (unchanged from TRI-70).

**Design-window amendment 1 (2026-07-06, checkpoint 1):** the v3 work order pinned temperature=0 on all three slots *as a means to* decision-level determinism. Empirically that means kills the arm: greedy decoding wedges the local thinking model on content-dependent inputs (checkpoint 1 — 4 consecutive 3600 s timeouts across two tickers, one 47-minute HTTP 500 in the Ollama server log), while an otherwise-identical run at model-default sampling completed healthy in 901 s under full point-in-time conditions (`tri69-diag-default-temp`, HOLD). The amendment swapped the means: pinned sampling seed on the local slots, temp=0 retained on the cloud judge. The qwen model card itself warns against greedy decoding for thinking models.

**Protocol amendment 2 (2026-07-06, checkpoint 2 → App Manager decision, RESOLVED by amendment, not loosening):**
- **Single-draw stochastic-policy protocol.** Each (ticker, date) decision is **ONE draw from the engine run exactly as production would run it — a stochastic policy.** No repeats, no modal voting. The decision-identical battery bar of the v3 draft is **replaced by this framing** per checkpoint 2 (commit `3d17f05`: post-fix instability shown stack-inherent — Ollama seeded generation is not bit-reproducible across server states, Anthropic temp=0 is best-effort, and one drifted token cascades through the 10-stage chain) **and the App Manager decision of 2026-07-06** (Option 2 of the checkpoint write-up; DEVELOP recommendation and reviewer concurrence aligned).
- **Noise level, recorded BEFORE any eval data:** 0.75 modal agreement on the burned AAPL point (SELL/HOLD/SELL/SELL over 4 post-fix same-day runs) ≈ **~25% decision-flip rate** per draw.
- **Policy identity (pinned per slot — the sampling config IS part of what is being tested):**
  | Slot | Model | Sampling |
  |---|---|---|
  | tool | `ollama/qwen3-coder:30b` | model-default sampling params, `seed=69`, `max_tokens=16384` |
  | reasoning_quick | `ollama/qwen3.5:9b` | model-default sampling params, `seed=69`, `max_tokens=16384` |
  | reasoning_deep | `anthropic/claude-sonnet-4-5-20250929` | `temperature=0` (best-effort determinism; no seed API) |
  Enhancements off; analyst cache off; fresh memory log per run; PIT mode on. Exactly the configuration the checkpoint-2 battery/diagnostic runs exercised.
- **Early-restart clause:** the reviewer reads this committed pre-registration during the first eval runs. If a material defect in these rules is flagged within the first hours, the run is stopped, the defect fixed, the pre-registration re-committed, and the eval **restarted from scratch with all prior draws discarded**. Stated here in advance so such a restart is a pre-registered contingency, not rule-shopping.

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
- **Primary test: exact one-sided sign-flip permutation over dates** on mean(X_t) (all 2^D sign assignments; p = fraction of permuted means ≥ observed). **α = 0.05, one-sided.** **Power floor, stated up front:** with D=8 dates the minimum attainable p is 1/256 ≈ 0.0039; significance at α=0.05 requires roughly ≥7 of 8 dates positive at comparable magnitudes (7-of-8 at equal magnitude: p = 9/256 ≈ 0.035).
- **Power caveat computed on DIRECTIONAL trials (checkpoint-2 protocol):** the decision noise concentrates in exactly the trials that enter the test. Expected directional N: 160 draws × (1 − expected HOLD/NO-DECISION rate 60–85%) ≈ **24–64 directional trials** (~48 at the 70% planning value), spread over 8 dates ≈ 3–8 per date. At a ~25% per-draw flip rate, a flipped draw contributes an ~uncorrelated sign, attenuating each date's mean signed excess toward zero by roughly the flip fraction (measured excess ≈ ~0.75× the modal policy's excess, under symmetric flips). Combined with the sign-flip floor, **this design detects only edges that are large and consistent enough to keep ≥7 of 8 date-means positive despite ~25% draw noise and single-digit per-date directional counts.** A real but modest edge will most likely read INCONCLUSIVE, not PROMISING — and the verdict language must say so.
- **Interpretation rule (pre-registered):** a null or negative result reads **"no *detectable* edge at this power"** — never "no edge exists." An underpowered outcome → INCONCLUSIVE plus the pre-registered expansion lever (seed-6969 +10 names × same dates), not a verdict.
- **Secondary (vs chance):** directional hit rate (share of decisions with sᵢ·(r_ticker − r_SPY) > 0) against the **realized base rate p₀ = max(all-BUY accuracy, all-SELL accuracy)** on the same decisions — NOT 0.5. Reported with the same date-clustered machinery; no independent verdict weight.
- Implementation: `scripts/score_tri69.py`, frozen at the pre-registration commit; 13 golden-fixture tests (hand-computed P&L, cost-application negative test, sweep monotonicity, exact permutation values incl. the D=8 floor, long-only no-op SELL).

### Expected HOLD rate + power plan (basis stated, not a post-hoc knob)

- **Basis (all recorded before any eval data):** TRI-70 cloud-deep runs were modal-HOLD (sonnet-deep HOLD×3, opus-deep HOLD×3 on AAPL@2026-06-27); the all-local finalist was 9/15 HOLD (60%) across AAPL/NVDA/TSLA; TRI-69's own burned-point runs under the final Config A policy observed **1 HOLD in 4 completed draws (25%)** on AAPL@2026-06-27 (plus 1 HOLD in 1 draw of the default-temp diagnostic). Config A may therefore HOLD substantially less than the TRI-70 cloud configs — the observed range across related configs is 25–100%. **Planning value stays 70% (conservative for power planning — a lower actual HOLD rate only increases directional N and helps power); range for planning 60–85%.**
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

**Track state:** RESOLVED 2026-07-06 — App Manager selected **Option 2** (single-draw stochastic-policy; DEVELOP recommendation and reviewer concurrence aligned). Written into the pre-registration as "Protocol amendment 2"; pre-reg committed with this resolution and the eval launched immediately after. No eval data existed before that commit.

---

## Eval ops log (mechanical interventions + milestone tallies only — NO scoring before completion)

- **2026-07-06 (launch):** eval started immediately after pre-reg commit `ebca159`; 160 runs, caffeinate, AC power, resumable.
- **25% (39/160):** SELL 30 / HOLD 5 / BUY 4 · 0 errors · 0 NO-DECISION · 0 leak hits · 0 infra retries · $2.06 · mean wall 977 s · ETA ~33 h. Directional N already 34 (min-N 20 exceeded); HOLD ~13%, well below the 70% planning value.
- **50% (79/160):** SELL 57 / BUY 13 / HOLD 9 · 0 errors · 0 NO-DECISION · 0 leaks · 0 infra retries · $4.33 · mean wall 998 s · ETA ~22.5 h. Directional N = 70; all 4 completed dates have ≥15 directional (min-N + date-coverage gates already satisfied). Policy remains SELL-heavy (descriptive tally only).
- **75% (119/160):** SELL 89 / BUY 16 / HOLD 14 · 0 errors · 0 NO-DECISION · 0 leaks · 0 infra retries · $6.50 · ETA ~11.5 h. Directional N = 105.
- **100% (160/160, 2026-07-08):** COMPLETE. SELL 121 / BUY 21 / HOLD 18 · **0 errors · 0 NO-DECISION · 0 leak hits · 0 infra retries · 0 mechanical interventions across the entire run** · $8.79 · all runs run-id'd with saved analyst reports. Scorer (frozen at pre-reg commit `ebca159`) executed once, unmodified.

## Results (frozen scorer, `results/tri69/scoring.json` — all numbers per the pre-registered spec)

**Sample:** 160/160 single-draw decisions (20 seed-69 names × 8 dates). HOLD 18 (11.25%), NO-DECISION 0 → **directional N = 142 across all 8 dates** (min-N gate ≥20 AND ≥6/8 dates: **met**, so the expansion lever does not fire). All 142 directional windows fully realized (0 unrealized).

**Primary test — T+10 signed net excess vs SPY, date-clustered exact sign-flip, α=0.05 one-sided:**
| Metric | Value |
|---|---|
| Mean of date-mean net excess (1× cost, 20 bps RT) | **+0.49%** |
| Exact sign-flip p (one-sided) | **0.348** — not significant (α=0.05) |
| Dates positive | **3 of 8** (+1.98%, +5.82%, +1.10% vs −1.37%, −0.25%, −1.67%, −0.30%, −1.42%) |
| Directional hit rate | 0.5775 |
| Realized base rate p₀ (best constant strategy) | **0.5845** — the hit rate is BELOW it |
| Between-date t (descriptive) | 0.55 |

**Slippage sweep (portfolio-level, monotone non-increasing ✓):** mean date-mean excess +0.49% (1×) → +0.29% (2×) → +0.09% (3×); p rises 0.348 → 0.410 → 0.477. The point estimate barely survives 3× cost and is never near significance.

**Secondary (T+5, descriptive):** mean date-mean −0.21%, p = 0.707, hit 0.514 — no signal at the shorter horizon.

**Descriptive long-only implementable view (no verdict weight):** strategy −0.25% vs equal-weight buy-and-hold −0.40% per date-mean — the mostly-in-cash portfolio lost slightly less than holding in a down-tilted sample window.

**Reading the pattern honestly:** the positive point estimate rests on a single strong date (2026-04-29, +5.82%); 5 of 8 dates are negative; the hit rate does not beat the best constant-sign strategy on the same trials (0.578 < 0.585). The policy was heavily SELL-tilted (121/142 directional), so it largely expresses one repeated view rather than name-by-name discrimination.

## Compute / $ spent

- Eval: 160 runs, ~1.9 days wall (mean ~990 s/run), **$8.79** cloud (Sonnet deep).
- Build/battery/diagnostics (checkpoints 1–2): ~20 runs + probes, ~$1.5.
- **Total ≈ $10.3** — bottom of the $10–30 budget.

## Verdict (pre-registered kill-switch mapping, applied mechanically)

> **Baseline label (applies to every number in this section):** these are the **sealed 160-row baseline** figures — pre-registration `ebca159` → completion `57c1876` → evidence seal `91a6591`; source `docs/TRI-69_scoring.json` (Arbiter-recomputed 2026-07-08; DEVELOP-recomputed independently 2026-07-10, every value matching to the last digit). They do **not** apply to any 180-row output: once the pre-registered extension has run, the 180-row actuals are reported in the extension addendum and this section's figures stand as the sealed historical baseline.

**INCONCLUSIVE.**
- Not PROMISING: p = 89/256 = 0.3477 ≫ α = 0.05 (rule required p ≤ 0.05 AND positive mean at 1× and 2×). *(sealed 160-row baseline)*
- Not STOP-PIVOT: the rule requires mean(X_t) ≤ 0 **AND** hit ≤ p₀ with min-N met; hit ≤ p₀ holds (82/142 = 0.5775 ≤ 83/142 = 0.5845) but the mean is (weakly) positive (+0.49%), so the conjunction fails. *(sealed 160-row baseline)*
- Therefore **INCONCLUSIVE**, with the pre-registered interpretation: **"no *detectable* edge at this power" — never "no edge exists."**

**Point estimate — both means, labeled (sealed 160-row baseline):** the headline +0.49% is the **mean of date means** (equal weight per date, the pre-registered primary statistic): mean(X_t) = **+0.4867%**. The **pooled per-trial mean** (equal weight per directional trial, n = 142) is **+0.4953%**. They differ because realized per-date directional counts vary (15–20); wherever this report says "+0.49%" unqualified, it means the date-means version.

**Date-clustered 95% CI on mean(X_t) (sealed 160-row baseline):** date-mean SD = 2.5060% (D = 8) → CI = +0.4867% ± 2.365 × 2.5060%/√8 = **[−1.61%, +2.58%]** (t₇ = 2.365, two-sided). *Recomputed independently by DEVELOP on 2026-07-10 from the committed per-decision rows in `docs/TRI-69_scoring.json` (rebuilt every eᵢ from raw legs); matches the Arbiter's 2026-07-08 value exactly.* The interval straddles zero, and its upper reach (+2.58%) does not by itself exclude a large edge — the operating table below carries the complementary likelihood statement (a build-worthy ≥200 bp edge would very likely have been detected, and was not).

**Directional N needed (pre-registered requirement for an INCONCLUSIVE verdict — sealed 160-row baseline):** power here is limited by the number of independent DATES, not by within-date names (all 8 dates already carry directional decisions). At the observed effect size (mean +0.49%, between-date σ 2.51%, standardized effect 0.19), detecting significance at α=0.05 one-sided with 80% power requires **~164 independent non-overlapping T+10 dates ≈ 6–7 years of history** — i.e., the observed effect is statistically indistinguishable from zero, and no feasible near-term expansion of this design would resolve it. The expansion lever (more names) does not apply: it adds within-date trials, and the date-clustered test's effective N is dates.

**Operating characteristics at realized parameters (Arbiter-produced 2026-07-08, independently regenerated by DEVELOP 2026-07-10 — sealed 160-row baseline parameters):** realized per-trial net signed-excess dispersion σ = **8.09%** (n = 142) · ~18 directional trials/date · D = 8 dates. Monte Carlo of the exact frozen verdict rule under independent-normal per-trial excess (exact 2⁸ sign-flip per simulation; 2×-cost leg = −20 bp shift on the mean; hit-rate leg thresholded at p₀ ≈ 0.5, which is conservative for STOP since the realized p₀ = 0.5845 ≥ 0.5 makes the real hit leg easier to satisfy). Arbiter table (8 000 sims/row):

| True net edge per trade | PROMISING | STOP | INCONCLUSIVE |
|---|---|---|---|
| 0 bp (coin flip) | 4.6 % | 40.8 % | 54.6 % |
| 30 bp | 10.1 % | 26.7 % | 63.2 % |
| 50 bp | 15.8 % | 17.7 % | 66.6 % |
| 100 bp | 36.8 % | 4.9 % | 58.3 % |
| 200 bp | 83.5 % | 0.0 % | 16.5 % |

*DEVELOP independent regeneration record (2026-07-10):* fresh implementation of the stated model, NumPy `default_rng(seed=20260710)`, **40 000 sims/row**, exact 2⁸ sign-flip per sim. **Comparison tolerance: ±1.25 pp** (≈2σ combined binomial MC noise at 8 000 + 40 000 sims/row). Regenerated cells (PROM/STOP/INC): 0 bp → 4.8/41.7/53.5 · 30 bp → 10.2/26.4/63.4 · 50 bp → 15.8/17.9/66.4 · 100 bp → 36.7/5.0/58.3 · 200 bp → 83.6/0.1/16.3. **Max |difference| vs the Arbiter table = 1.11 pp (0 bp STOP cell) — agreement within MC noise at the stated counts; the table stands.** Reading unchanged: false-PROMISING is pinned at ~α by construction; the test reliably detects only a **large** (≳200 bp/trade net) edge; the observed +49 bp sits where INCONCLUSIVE is the expected outcome even if the effect were real.

**Extension futility bound — corrected two-sided statement (PRE-RUN ESTIMATE, fixed 2026-07-09/10, before any extension run; to be demoted to historical context once extension actuals exist):**
- **PROMISING is unreachable:** with the 8 sealed date means fixed, as the 9th date mean → +∞ the exact sign-flip p converges from above to 89/512 = **0.1738** — no possible 9th-date outcome reaches p ≤ 0.05.
- **STOP-PIVOT is reachable:** the 8 sealed date means sum to +3.89%, so a 9th date mean ≤ −3.89% makes the overall mean non-positive. The mean threshold alone does not guarantee STOP-PIVOT — the recomputed hit-rate leg must also hold, and its current margin is exactly one observation (82 hits vs the 83/142 constant-strategy baseline); it usually holds in the flip scenario, not always. Probability ≈ **3–6 %, model-dependent** (reviewer 500 k-draw simulations: per-trial normal ≈ 2.7 %, centered empirical bootstrap ≈ 2.6 %, between-date-SD normal ≈ 6 %; Arbiter bracket-verified).
- **Net pre-run estimate: ≈94–97 % INCONCLUSIVE / ≈3–6 % STOP-PIVOT / 0 % PROMISING.** Either formal outcome maps to the same STOP-posture, and the mandated verdict language is identical under either label: *"no detectable edge at this power."* A formal STOP-PIVOT would be a stronger operational label, not stronger proof.

**INCONCLUSIVE-ladder state and extension decision (recorded):**
- **Branch fired: INCONCLUSIVE-by-power** (mean > 0 at 1× and 2×, p > 0.05) → obligates ONE pre-registered extension round.
- **Extension scope locked to exactly one date (2026-06-11) × the same 20 names — an approved post-verdict implementation clarification (Jeff/Arbiter, 2026-07-09/10) that alters no scoring or verdict logic.** It is forced by the frozen scorer's hard-coded price window (`score_tri69.py` `fetch_prices(..., end="2026-07-03")`): 2026-06-11 (exit 2026-06-26) is the only extension date executable without a protocol amendment. The date count was not part of the original pre-registration — the ladder pre-registered "one extension round" without pinning it.
- **Jeff's ladder call (recorded): RUN the extension**, after QA validates the scorer and price legs (~$1.10, ~5.7 h serial, single draw per point, original Config A unchanged).
- **Terminal rule (Jeff, frozen 2026-07-06):** after at most one extension the result is FINAL; a final INCONCLUSIVE = **STOP-posture** — *"stop investing — edge not demonstrated at achievable power."* Pivot allowed; chasing this config not.

**Honest caveats carried with the verdict (pre-registered; sealed 160-row baseline):**
1. Single-draw stochastic-policy protocol: each decision includes ~25% draw noise (attenuates a true modal edge by roughly that fraction — a true modal edge of ~+0.65%/date would measure as the observed +0.49%; still far from detectable at D=8).
2. The test detects only large, cross-date-consistent edges; a real but weak or regime-dependent edge reads INCONCLUSIVE by construction.
3. The result neither validates nor falsifies the engine thesis: it says this configuration, on this sample, shows no edge distinguishable from noise, and the hit rate did not beat the base rate.

**What the kill-switch DOES license:** nothing here justifies scaling up compute on Config A as-is (the go/no-go's practical intent). The signal, if any, is too weak/inconsistent to detect at any feasible horizon with this design.
