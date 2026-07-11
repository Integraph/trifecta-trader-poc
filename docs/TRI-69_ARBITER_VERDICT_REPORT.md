# TRI-69 — Arbiter Verdict Report (complete, for adversarial QA review)

**Author:** Ecosystem Arbiter · **Date:** 2026-07-08 · **Status:** verdict verified; gate not yet Done
**Repo:** `trifecta-trader-poc`, branch `jeff/tri-69-edge-check` · **Pre-registration commit:** `ebca159` · **Completion commit:** `57c1876`
**Companion docs (in-repo):** `docs/TASK_TRI-69_EDGE_CHECK.md` (spec v3, LOCKED) · `docs/TASK_TRI-69_REPORT.md` (DEVELOP report) · `docs/TRI-69_scoring.json` / `docs/TRI-69_eval_aggregate.json` (committed data) · `scripts/score_tri69.py` (frozen scorer, 13 golden fixtures) · `AI/ARBITER_TO_ENGINE_MANAGER_TRI-69_VERDICT_REPORTING.md` (reporting directives, INCONCLUSIVE ladder)

---

## 1. Executive summary (plain English)

We asked one question, pre-registered before any data existed: **does the engine's trading decision predict 10-day direction better than holding SPY, net of costs, on stocks and dates it was never tuned on?**

The answer is **INCONCLUSIVE — no detectable edge at this power.** The measured effect was small and positive (+0.49% per trade net of costs), but statistically indistinguishable from zero (p = 0.348 vs the 0.05 bar), carried almost entirely by one good date out of eight, and the directional calls did not beat a naive constant-sign strategy (hit rate 57.75% vs 58.45% base rate).

Under the rules frozen before the number existed, this is **not** "the strategy is dead," and it is emphatically **not** "promising." It means: **the burden of proof was on finding edge; it was not found at achievable power; investment in this configuration stops by default.** A pivot (different signals, different approach, cheaper test) remains open. Spending more on this config requires a deliberate, eyes-open re-investment decision by Jeff — never a reflex.

The measurement itself was flawless: 160/160 runs, zero errors, zero look-ahead leak hits, zero protocol deviations after pre-registration, $8.79 of eval spend (~$10.30 total task).

---

## 2. What was tested (frozen design, abridged)

| Element | Frozen value |
|---|---|
| Signal arm | Config A (TEST-ONLY): tool `ollama/qwen3-coder:30b` + quick `ollama/qwen3.5:9b` (model-default sampling, `seed=69`, 16 384-token guard) + deep `claude-sonnet-4-5` @ temp=0; PIT mode ON, cache OFF, enhancements OFF |
| Protocol | **Single-draw stochastic policy** — ONE engine run per (ticker, date), exactly as production would run; no repeats, no modal voting; ~25 % per-draw decision-flip noise recorded pre-data (checkpoint 2) |
| Universe | 20 S&P 500 names, mechanical seed-69 draw from as-of-2026-02-17 membership (survivorship-safe source, commit-pinned); AAPL/NVDA/TSLA excluded as the tuning set |
| Dates | 8 non-overlapping T+10 windows: 2026-02-17 → 2026-05-28 (every 10th trading day, mechanical) |
| Sample | 160 point-in-time decisions |
| Measurement | Entry Open(t+1), exit Close(t+10), dividend-adjusted; benchmark = SPY over the identical window; cost 10 bps/side (20 bps round trip) charged to the signal leg; sweep 1×/2×/3× |
| Statistic | Per directional decision *i*: eᵢ = sᵢ·(r_ticker − r_SPY) − c. Date-level mean X_t; **primary test = exact one-sided sign-flip permutation over the 8 date means (2⁸ = 256 assignments), α = 0.05.** HOLD/NO-DECISION excluded from the directional test, dual-reported |
| Verdict rule | PROMISING = p ≤ 0.05 AND mean > 0 at 1× AND 2× cost · STOP-PIVOT = mean ≤ 0 AND hit ≤ p₀ AND min-N met · else INCONCLUSIVE |

Two pre-data checkpoints fired and were resolved inside the design window, fully documented in the DEVELOP report: (1) temp=0 wedges local thinking models → seed-pinned default sampling on local slots; (2) decision-level determinism proved stack-inherent → App-Manager-approved single-draw stochastic-policy protocol. No eval data existed before the pre-registration commit.

---

## 3. The result

**Totals:** 160/160 scored · decisions 121 SELL / 21 BUY / 18 HOLD (HOLD rate 11.25 %) · directional n = 142 across all 8 dates · 0 errors · 0 NO-DECISION · 0 leak hits · 0 infra retries.

**Per-date mean net excess X_t (T+10, 1× cost):**

| Date | n | X_t |
|---|---|---|
| 2026-02-17 | 15 | −1.3699 % |
| 2026-03-03 | 20 | −0.2522 % |
| 2026-03-17 | 19 | −1.6738 % |
| 2026-03-31 | 17 | **+1.9842 %** |
| 2026-04-15 | 17 | −0.2954 % |
| 2026-04-29 | 18 | **+5.8188 %** |
| 2026-05-13 | 18 | **+1.1023 %** |
| 2026-05-28 | 18 | −1.4203 % |

**Headline statistics (1× cost):** mean(X_t) = **+0.4867 %** · exact sign-flip **p = 89/256 = 0.3477** · hit rate **0.5775** vs realized base rate p₀ = **0.5845** · 3 of 8 dates positive · date-mean SD = 2.5060 % · **95 % date-clustered CI on mean(X_t): [−1.61 %, +2.58 %]** (t₇ = 2.365).

**Cost sweep (monotonicity requirement satisfied):** 1× mean +0.4867 % (p 0.348) · 2× +0.2867 % (p 0.410) · 3× +0.0867 % (p 0.477).

**Verdict mapping (mechanical):** Not PROMISING (p = 0.348 ≫ 0.05). Not STOP-PIVOT (hit ≤ p₀ held, but mean > 0 breaks the conjunction). → **INCONCLUSIVE.**

---

## 4. Arbiter independent verification (all checks passed)

Performed 2026-07-08 in a sandbox against the committed tree — none of these numbers were taken from the DEVELOP report on trust.

1. **Provenance:** `ebca159` (pre-registration) confirmed ancestor of `57c1876` (completion). `git diff ebca159..57c1876 -- scripts/score_tri69.py scripts/run_tri69_edge_check.py` is **empty** — scorer and runner byte-frozen across the entire eval.
2. **Full recompute from committed per-decision rows** (`docs/TRI-69_scoring.json` → `primary_T10_1x.decisions`, n = 142): re-derived every eᵢ from raw legs (sign × (r_ticker − r_SPY) − cost), rebuilt the 8 date means, re-ran the exact 256-assignment permutation, recomputed hit rate. **Every headline number matches the scorer to the last digit** (mean +0.004867; p = 89/256 = 0.347656; hit 0.5775).
3. **Row-level consistency:** max |recomputed − stored| = 1.0e-06 on both net and gross columns (= 6-decimal JSON rounding); 0 of 142 hit-flag mismatches; decision counts sum to 160; directional n = 160 − 18 HOLD = 142 ✓.
4. **Verdict mapping** re-applied mechanically from the frozen rule → INCONCLUSIVE, reproduced.
5. **Operational claims** cross-checked against the eval tracker and result files during the run (progress, cost, cadence, zero leak hits) on 2026-07-06 and 2026-07-07 (standup agent + Arbiter spot checks).

**Verification boundary, stated honestly:** the sandbox has no market-data network access, so raw entry/exit **prices** were not independently re-fetched. The price legs rest on the frozen scorer's yfinance pulls (dividend-adjusted, fixture-tested). This is the one link Codex can strengthen: independently re-fetch a sample of (ticker, date) windows and confirm entry Open(t+1)/exit Close(t+10) values — see §7.

---

## 5. Was the test capable of a different answer? (operating characteristics at REALIZED parameters)

Mandated by the Arbiter reporting directive; computed from the committed data. Realized per-trial dispersion σ = **8.09 %** (n = 142 net signed excess values); realized per-date n ≈ 18; D = 8 dates. Monte Carlo of the exact frozen verdict rule (8 000 sims/row):

| True net edge per trade | PROMISING | STOP | INCONCLUSIVE |
|---|---|---|---|
| 0 bp (coin flip) | 4.6 % | 40.8 % | 54.6 % |
| 30 bp | 10.1 % | 26.7 % | 63.2 % |
| 50 bp | 15.8 % | 17.7 % | 66.6 % |
| 100 bp | 36.8 % | 4.9 % | 58.3 % |
| 200 bp | 83.5 % | 0.0 % | 16.5 % |

Reading: **false-PROMISING is controlled at ~α by construction** (the permutation test's Type-I guarantee holds under draw noise — noise only costs power). The test reliably detects only a **large** edge (≥ ~200 bp/trade net). The observed +49 bp/trade point estimate sits in the zone where INCONCLUSIVE is the *expected* outcome even if the effect were real. Conversely: a 200 bp edge — the kind worth building a business on — would have shown PROMISING with ~83 % probability. **It did not.**

*Assumptions footnote (added 2026-07-09 per review):* the table is conditional on its Monte-Carlo model — independent normal per-trial excess at realized σ = 8.09 %, ~18 trials/date, 8 dates; it is not a distribution-free guarantee. Note also that +200 bp lies inside the 95 % CI's upper reach (+258 bp): the CI does not *exclude* a large edge; the power statement says a large edge would very likely have *announced itself*, and it didn't. Both statements are true simultaneously — the likelihood evidence runs against a large edge; certainty is not claimed. Detection of the observed effect size by this test family would need ~164 independent dates (~6–7 years of T+10 windows); more names per date cannot substitute (the test's effective N is dates).

---

## 6. The INCONCLUSIVE ladder — state and the futility proof

The ladder was **frozen pre-verdict** (2026-07-06, Jeff's terminal ruling) precisely so this moment involves no improvisation.

- **Branch fired: INCONCLUSIVE-by-power** (mean > 0 at 1× and 2×, p > 0.05) → obligates ONE pre-registered extension round (add dates by the mechanical rule, realized windows only).
- **Futility finding — CORRECTED 2026-07-09 after independent review.** The original claim ("the extension cannot change the verdict") was **overstated**; what the proof establishes is one-sided:
  - **PROMISING is unreachable** (this half stands): with the observed 8 date means fixed, as the 9th date value → +∞ the exact sign-flip p converges from above to 89/512 = **0.1738** — no possible 9th-date outcome reaches p ≤ 0.05. *(Proof: every sign-assignment negating the new date falls strictly below the observed mean; the survivors reduce to the original 89-of-256 count.)*
  - **STOP-PIVOT is reachable** (reviewer's catch, Arbiter-verified against `score_tri69.py`'s verdict rule): the 8 date means sum to +3.89 %, so a 9th date mean ≤ −3.89 % makes the overall mean non-positive. **The mean threshold alone does not guarantee STOP-PIVOT** — the recomputed hit-rate leg must also hold, and its current margin is exactly one observation (82 hits vs an 83/142 constant-strategy baseline); it usually holds in the flip scenario, not always. Probability ≈ **3–6 %, model-dependent** (reviewer's 500 k-draw simulations: per-trial normal ≈ 2.7 %, centered empirical bootstrap ≈ 2.6 %; between-date-SD normal ≈ 6 %; Arbiter bracket-verified).
  - **A formal STOP-PIVOT would be a stronger operational label, not stronger proof.** The STOP rule carries no negative-significance test — one extreme date can trigger it. The mandated verdict language remains *"no detectable edge at this power"* under **either** formal outcome; "actively no edge" is forbidden phrasing.
  - **Net:** the extension is ~94–96 % INCONCLUSIVE / ~4–6 % STOP-PIVOT / **0 % PROMISING**. Both possible outcomes map to the same STOP-posture, so the *business* consequence is invariant — but the extension is not "zero information": it has a small chance of formally strengthening the kill.
  - **Extension spec — one date, locked as a documented post-verdict implementation clarification (Jeff/Arbiter-approved), not an originally-frozen count.** The ladder said "one extension round" without pinning the date count; the scorer's hard-coded price window through 2026-07-03 (`score_tri69.py` `fetch_prices(..., end="2026-07-03")`) forces the executable spec to **exactly 2026-06-11** (exit 2026-06-26, inside the window) × the same 20 names ≈ $1.10, **~5.7 h serial**. Any additional date requires changing frozen code = a protocol amendment = forbidden; the 3-date PROMISING floor (0.0435) is thereby unreachable in practice as well as ruled out in policy. The clarification is committed as a pre-registered **extension addendum BEFORE the runs start**.
- **Terminal rule (Jeff, frozen 2026-07-06):** after at most one extension the result is FINAL; final INCONCLUSIVE = **STOP-posture** — *"stop investing — edge not demonstrated at achievable power."* Pivot allowed; chasing this config not; noise-reduction (TRI-73 → a NEW pre-registered battery) only as a deliberate re-investment decision, never the default.
- **Open decision (Jeff) — Arbiter recommendation REVISED 2026-07-09:** run the pre-registered extension (exactly 2026-06-11 × 20 names), **after** QA validates the scorer and price legs. It is no longer a ritual: ~5 % chance of a formally stronger kill (STOP-PIVOT), zero chance of revival, trivial cost, and it removes any need for a waiver. If Jeff still elects to skip, the record must use the accurate waiver language: *"a one-date extension cannot produce PROMISING but could produce STOP-PIVOT; the pre-registered extension is explicitly waived because either formal outcome maintains STOP-posture."* — never "provably neutral."

---

## 7. Reproduction & retest protocol (for Codex QA)

**What a valid retest IS:**
1. **Recompute the verdict** from `docs/TRI-69_scoring.json`: rebuild eᵢ from raw legs, date means, the exact 2⁸ permutation, hit rate, mapping. (Arbiter's 20-line check: group `primary_T10_1x.decisions` by date; eᵢ = ±(r_ticker − r_SPY) − 0.002 with sign from decision; p = #{assignments with mean ≥ observed}/256.)
2. **Re-run the frozen scorer** at `57c1876` against the committed result JSONs; confirm 13 golden fixtures green and identical output.
3. **Independently re-fetch prices** for a sample of (ticker, date) rows (yfinance, auto-adjusted) and confirm entry Open(t+1) / exit Close(t+10) legs — this closes the one link the Arbiter could not verify offline (§4). **Then SEAL the price legs (added 2026-07-09, provenance-honest per review v2):** the scorer re-downloads mutable yfinance history on every run and no raw OHLC snapshot is committed — "byte-reproducible" holds today but is not durable if adjusted history changes. QA fetches once and commits: the exact OHLC arrays for every leg (incl. SPY), **provider settings, fetch timestamp, and content hashes — labeled honestly as a QA re-fetch snapshot (not the original Jul-8 scorer input)** — plus a **verifier script (new file; the frozen scorer stays untouched)** that reproduces every return in `scoring.json` from the snapshot. Durable reproducibility then attaches to the snapshot + verifier; QA's sample re-fetch comparison is what ties the snapshot back to the original run's legs.
4. **Audit the point-in-time guarantee:** leak-guard code + saved analyst-report snapshots per run-id; confirm no as-of-date violations in a sampled run.
5. **Verify the futility bound:** compute p for the 8 observed date means plus a synthetic 9th ∈ {+2 %, +6 %, +10 %, +50 %}; confirm p never ≤ 0.05.
6. **Recompute the §5 table** from realized σ if desired (seeded MC; Arbiter used 8 000 sims/row).

**What a valid retest is NOT:** re-running engine analyses on the same (ticker, date) points and comparing decisions. The pre-registered protocol is a **single-draw stochastic policy** — the stack is non-deterministic by documented finding (checkpoint 2), so repeat draws WILL differ ~25 % of the time. That is measured noise, not a scoring error, and it is already carried in the verdict's power caveats. Decision-level re-execution tests the pipeline's operability, not the verdict's correctness.

**Also open for this QA round (required changes already filed):** fold the §3 CI and the §5 realized-σ table into `docs/TASK_TRI-69_REPORT.md` §Verdict (reporting-only; scorer and its fixtures must remain untouched).

---

## 8. What this means — and what it does not (anti-spin, both directions)

- It does **not** mean the thesis is disproven. A modest real edge (≤ 50 bp/trade net) is consistent with this data — the test cannot see effects that small, by pre-registered design.
- It **does** mean no *detectable* edge at the power we could afford — and, decisively for the business question: **a large, build-worthy edge (≥ ~200 bp) would very likely have been detected, and was not.**
- It does **not** license "run it again until it says yes." The anti-spin clause forbids reporting this as promising; the ladder forbids reflexive re-runs. The one strong date (2026-04-29, +5.82 %) is exactly the pattern that seduces — one date of eight is what luck looks like.
- What survives regardless: the **scanner** (rule-based, zero LLMs, its own gate-verified test suite), the **measurement machine** (point-in-time store, leak guards, frozen-scorer discipline, the gate itself), and the **stack findings** (local quality solved at 8.13; determinism stack-inherent; TRI-73 input recorded). Any pivot inherits all of it, and the next thesis gets tested this cheaply again.
- Total cost of the answer: **~$10.30 and 9 days**, against the months of polish it pre-empted.

---

## 9. Path to Done (gate state)

1. **Codex QA** of scoring + verdict per §7, including the two reporting insertions (CI, realized-σ table). ← current step
2. **Behavioral UAT** (headless carve-out): confirm engine-consumable artifacts exist as committed (result JSONs, scoring, aggregate; spot-render one decision chain).
3. **Arbiter final sign-off** per the ecosystem protocol (`ECOSYSTEM_CONTEXT.md` §10 — external reference, not a section of this document): ancestor check ✓ (done), byte-identical merged main once merged (pending — branch not yet merged), suite green on merged main.
4. **Jeff:** FINAL-vs-extension call (§6) · then the pivot conversation (crypto's parked state, TRI-28 EU-AI-Act scheduling, and what the next cheap pre-registered test should be).

*Prepared by the Ecosystem Arbiter. Every number in this document was recomputed independently from the committed tree; where a claim could not be independently verified offline (raw price legs), that boundary is stated in §4 and assigned to QA in §7.*
