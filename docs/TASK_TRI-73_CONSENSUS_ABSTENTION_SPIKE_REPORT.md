# TRI-73 SPIKE — Consensus-With-Abstention

**Date:** 2026-07-08  
**Status:** Spike record only. TRI-73 is not authorized for implementation or more engine runs.  
**Scope:** Post-TRI-70/TRI-69 stability follow-up: local model alternatives, hybrid Haiku-tools variants, cloud final judge, and consensus-with-abstention.

---

## Executive Summary

This spike found one useful mechanism, but it does not change the TRI-69 conclusion.

**Verdict:** technically useful, procedurally out of order, and not sufficient to move STOP-posture.

- TRI-69 remains the controlling result: **no detectable edge at that power**.
- Consensus can stabilize a noisy signal, but it cannot create edge from a signal that already failed the edge check.
- The cloud final judge was not universally stable: AAPL passed 3/3, but NVDA flipped.
- The consensus rule is correctly conservative: `0.80` agreement with `N=3` requires unanimity, ties abstain, and `UNKNOWN`/error rows penalize agreement.
- The rule also abstains often. With an observed rough flip rate near 25%, a simple unanimity-of-3 screen passes about `0.75^3 = 42.2%` before quality gating, leaving about 57.8% abstained away.

This should be shelved as a spike. If TRI-90 later finds the scanner has real signal and Jeff chooses to re-invest in the engine arm, this consensus protocol should be pre-registered into that future test.

---

## Process Caveat

These runs happened while TRI-73 was parked under STOP-posture. They reused benchmark-style points, mostly `2026-06-27` on AAPL/NVDA/TSLA, and are valid only as operational stability screens.

They are **not** a new pre-registered edge test. They must not be reported as evidence that the strategy has predictive edge.

No additional engine runs should happen on this lane without an explicit work order.

---

## NO_TRADE Contract

`NO_TRADE` must not be treated as a fancy `HOLD`.

The implemented contract is:

- `HOLD`: a confident portfolio decision to hold.
- `NO_TRADE`: the repeat-run evidence failed the consensus gate.
- `NO_TRADE` authorizes no new entry.
- `NO_TRADE` authorizes no size increase or decrease from this signal alone.
- Existing-position handling must be supplied by the caller's risk policy, such as keep-with-existing-stops, manual review, or flatten.

If downstream execution treats `NO_TRADE` as "do nothing" with the same behavior as `HOLD`, the mechanism is cosmetic and should not be promoted.

---

## Artifacts

| File | Purpose |
|---|---|
| `src/consensus.py` | Reusable consensus-with-abstention rule and explicit `NO_TRADE` semantics |
| `scripts/consensus_with_abstention.py` | Applies the rule to saved result JSONs |
| `tests/test_consensus.py` | Unit coverage for pass, abstain, weak majority, tie, quality, and `NO_TRADE` semantics |
| `scripts/run_tri70_benchmark.py` | Benchmark aggregates include consensus output |

Generated result artifacts are under `results/`, which is gitignored. The Arbiter correctly flagged that this is dangerous for TRI-69 evidence because decisive run artifacts currently live as single-copy local files unless force-added and pushed.

---

## Configs Screened

| Config | N | Purpose |
|---|---:|---|
| `bench_deep_gemma4_31b` | 1 | Local deep-slot Gemma 31B screen |
| `bench_deep_gptoss_20b` | 1 | Local deep-slot gpt-oss 20B screen |
| `bench_deep_llama33_70b` | 1 | Local deep-slot Llama 3.3 70B screen |
| `bench_deep_r1_8b_seeded` | 2 | R1 8B with Ollama seed/max-token pins |
| `hybrid_haiku_r1_8b` | 0 completed | Haiku tools + qwen3.5 quick + R1 deep, enhanced |
| `hybrid_haiku_qwen25_r1_8b` | 0 completed | Haiku tools + qwen2.5 quick + R1 deep, enhanced |
| `hybrid_haiku_r1_8b_raw` | 2 | Haiku tools + qwen3.5 quick + R1 deep, no enhancement |
| `hybrid_haiku_qwen25_r1_8b_raw` | 1 | Haiku tools + qwen2.5 quick + R1 deep, no enhancement |
| `tri69_config_a` cloud final judge | 6 | AAPL x3 and NVDA x3 |

The N=1 model screens are not stability evidence. They are only quality/runtime eliminators.

---

## Results

### Local Model Candidate Screens

All three N=1 local candidates failed the quality bar on AAPL.

| Config | Ticker/date | Decision | Method | Quality | Wall time | Finding |
|---|---|---|---|---:|---:|---|
| `bench_deep_gemma4_31b` | AAPL / 2026-06-27 | HOLD | structured render | 5.1 | 1572.3s | Fail |
| `bench_deep_gptoss_20b` | AAPL / 2026-06-27 | HOLD | structured render | 5.6 | 807.6s | Fail |
| `bench_deep_llama33_70b` | AAPL / 2026-06-27 | HOLD | structured render | 4.2 | 1029.2s | Fail |

Conclusion: stop model fishing in this direction. These were not near misses.

### Seeded R1 Stability Probe

`bench_deep_r1_8b_seeded` used `local_seed = 73` and `local_max_tokens = 16384`.

| Repeat | Decision | Method | Quality | Wall time |
|---:|---|---|---:|---:|
| 1 | HOLD | regex | 9.1 | 1052.2s |
| 2 | SELL | regex | 7.5 | 1044.3s |

Conclusion: seed pinning did not solve decision instability.

### Haiku Tools + R1 Deep Variants

The enhanced `hybrid_haiku_r1_8b` path produced no first result after more than 30 minutes and was interrupted.

| Config | N | Decisions | Quality | Finding |
|---|---:|---|---|---|
| `hybrid_haiku_r1_8b_raw` | 2 | BUY, HOLD | 9.1, 8.5 | Good quality, unstable |
| `hybrid_haiku_qwen25_r1_8b_raw` | 1 | HOLD | 5.4 | Fail |

Conclusion: Haiku tools improved one quality path, but did not solve stability.

### Cloud Final Judge

`tri69_config_a`:

- tool slot: `ollama/qwen3-coder:30b`
- quick slot: `ollama/qwen3.5:9b`
- deep/final judge: `anthropic/claude-sonnet-4-5-20250929`
- `deep_temperature = 0.0`
- `local_seed = 69`
- `local_max_tokens = 16384`

| Ticker | N | Decisions | Agreement | Quality mean | Consensus |
|---|---:|---|---:|---:|---|
| AAPL | 3 | SELL, SELL, SELL | 1.000 | 8.300 | SELL |
| NVDA | 3 | BUY, BUY, SELL | 0.667 | 8.233 | NO_TRADE |

Conclusion: cloud final judge helps extraction and quality, but the NVDA flip proves it is not a universal stability fix.

### Existing Local R1 Finalist Evidence

From `results/tri70_finalist_agg.json`:

| Group | N | Decisions | Agreement | Quality mean | Consensus |
|---|---:|---|---:|---:|---|
| `benchmark_local_b@AAPL` | 5 | BUY, BUY, HOLD, HOLD, HOLD | 0.600 | 8.440 | NO_TRADE |
| `benchmark_local_b@NVDA` | 5 | SELL, HOLD, HOLD, HOLD, SELL | 0.600 | 7.660 | NO_TRADE |
| `benchmark_local_b@TSLA` | 5 | HOLD, BUY, SELL, HOLD, HOLD | 0.600 | 8.280 | NO_TRADE |

Conclusion: a 3/5 modal vote is disagreement, not confidence.

---

## Consensus Rule

```text
min_runs = 3
min_agreement = 0.80
min_quality = 8.0
```

Output:

- `BUY`, `HOLD`, or `SELL` only if all gates pass.
- `NO_TRADE` if runs are insufficient, agreement is weak, quality is low, or votes tie.
- `UNKNOWN` and error rows penalize agreement rather than being silently ignored.

Generated review artifacts:

| Artifact | Summary |
|---|---|
| `results/tri70_local_r1_consensus.json` | All local R1 groups abstain |
| `results/tri70_cloudjudge_consensus.json` | AAPL passes SELL; NVDA abstains |

---

## Required Quarantine And Evidence Seal

Per Arbiter, these git operations should be done by Jeff, not an agent:

1. Quarantine the spike to `jeff/tri-73-consensus-spike` off `main`.
2. Commit the spike files there, including the 4 new files and 3 modified files.
3. Optionally force-add the spike's result JSONs so the report references resolve.
4. Restore the TRI-69 eval branch tree to pristine `57c1876`:

```bash
git restore config/hybrid_llm.yaml scripts/build_tri70_configs.py scripts/run_tri70_benchmark.py
```

5. Seal TRI-69 evidence on the eval branch by force-adding the ignored run artifacts and analyst snapshots, then push a single "TRI-69 evidence seal" commit.
6. Let QA proceed from the clean code/verdict target at `57c1876`, citing the evidence-seal commit only for artifacts.

---

## Verification

Commands run:

```bash
PYTHONPATH=".:vendor/TradingAgents" pytest tests/test_consensus.py tests/test_hybrid_llm.py -q
PYTHONPATH=".:vendor/TradingAgents" python -m py_compile src/consensus.py scripts/consensus_with_abstention.py scripts/run_tri70_benchmark.py
```

Result:

```text
24 passed
```

---

## Recommended Decision

Do not promote this into TRI-73 now.

Shelve it as a spike with useful engineering notes. The next authorized work should stay on TRI-90 / scanner validation. If scanner signal exists and the engine arm becomes worth re-opening, pre-register consensus-with-abstention as part of that future test before running anything.
