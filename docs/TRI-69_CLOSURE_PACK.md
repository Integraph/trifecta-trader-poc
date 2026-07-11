# TRI-69 Closure Pack — FINAL (Arbiter-authored)

**Author:** Ecosystem Arbiter, 2026-07-10 — authorship escalated after two authoring rounds per the §10 two-round cap. Supersedes `TRI-69_CLOSURE_PROMPTS*.md` v1–v3. **Rev 2 (2026-07-10):** all four blockers + operational corrections from the reviewer's first pass applied — full-artifact drift gate (deep equality + byte `cmp`), harness constrained to monkeypatch-only + untouched `main()`, corrected `../../AI/` source paths, assembler metadata-recompute contract, `caffeinate`/append-log run wrapper, enumerated evidence manifest, noncanonical labeling for all audit re-fetches, MC provenance recording.
**Rev 3 (2026-07-10, hash-pinned re-review):** evidence enumeration reconciled with the permitted-failure protocol (per-row mapping: result JSON **or** error record + console/memory evidence; no retries to satisfy artifact counts); all eight assembler metadata fields pinned with exact-value fixtures (mode/config equality-required, universe as structured source-pair with hashes); Step-0 pack-hash verification + post-copy `cmp`; manifest excludes itself; Step-9 executes and records manifest verification; preflight results appended to the console log.
**Review path:** this pack → independent reviewer → on APPROVED, terminal executes. **Nothing runs before that APPROVED.**
**Jeff's ladder call (recorded):** RUN the extension after QA.
**Citation rule (applies to this document):** every concrete claim below carries the source it was verified against. Anything discovered to contradict a citation at execution time = STOP, escalate to Arbiter.

**Pinned SHAs:** pre-registration `ebca159` · completion `57c1876` · evidence seal `91a6591` (immutable — never amended, only referenced). Future SHAs are variables recorded at creation: `$STEP1_SHA`, `$STEP3_SHA`, `$STEP5_SHA`, `$STEP8_SHA`.

---

## G. Global invariants (every step)

- **Branch:** all work on `jeff/tri-69-edge-check`, each handoff commit pushed immediately (mini QA clones from origin).
- **Freeze list — byte-identical from `ebca159` through Done** *(existence verified on branch 2026-07-10)*: `scripts/score_tri69.py` · `tests/test_tri69_scoring.py` · `scripts/run_tri69_edge_check.py` · `config/hybrid_llm.yaml` · `src/hybrid_llm.py` · `src/point_in_time.py` · `tests/test_tri69_point_in_time.py`. All new tooling is **additive files only**.
- **Forbidden phrasing:** "provably neutral," "zero information," "actively no edge." Verdict language identical under either label: *"no detectable edge at this power."*
- **Forbidden substitutions (explicit, for the terminal):** the TRI-73 consensus implementation (`src/consensus.py` etc., quarantined on `jeff/tri-73-consensus-spike`), modal voting, repeated draws, any Haiku/DeepSeek/R1/alternative-model configs. The extension is **original Config A, single draw per point, exactly as the 160 ran.**
- **Single-writer:** the execution terminal makes and pushes every commit, including committing QA's review artifacts verbatim. QA authors text; the terminal commits it; nobody approves their own code (Step 3 authored by DEVELOP, attacked in Step 4 by QA).
- **Statistics labels:** the 160-row numbers are the **sealed 160-row baseline** (they do not apply to 180-row outputs): mean of date means **+0.4867%** · pooled per-trial mean **+0.4953%** · date-mean SD **2.506%** · 95% CI **[−1.61%, +2.58%]** (t₇=2.365) · p = **89/256 = 0.3477** · hit **82/142 = 0.5775** vs p₀ **83/142 = 0.5845** · sweep **+0.4867/+0.2867/+0.0867%** · realized σ **8.09%** *(source: `docs/TRI-69_scoring.json`, Arbiter-recomputed 2026-07-08)*. The **"≈94–97% INCONCLUSIVE / ≈3–6% STOP-PIVOT / 0% PROMISING"** figures are the **pre-run model-conditional estimate** — after the extension, report actuals and demote the estimate to historical context.
- **One-date lock language:** an **approved post-verdict implementation clarification** (Jeff/Arbiter, 2026-07-09/10) that alters no scoring or verdict logic — forced by the scorer's frozen price window `end="2026-07-03"` *(source: `score_tri69.py` `fetch_prices` call, verified)*. Never describe it as originally pre-registered.

---

## Step 0 — Worktree prep + sealing external docs into the repo

1. In `trifecta-trader-poc`: `git status` must be clean (only `.claude/` untracked permitted); `git fetch origin`; `git checkout jeff/tri-69-edge-check`; assert `git rev-parse HEAD` = `91a6591` *(current branch tip, verified 2026-07-10)*. Note: the worktree currently sits on `main` — this checkout is required, not optional.
2. Copy into the repo so the from-origin mini can cite them — **sources are outside the repo** (no repo-local `AI/` exists; `../../AI/` from the repo root, verified): `../../AI/TRI-69_ARBITER_VERDICT_REPORT.md` → `docs/TRI-69_ARBITER_VERDICT_REPORT.md`; `../../AI/TRI-69_CLOSURE_PACK_FINAL.md` → `docs/TRI-69_CLOSURE_PACK.md`. (Absolute fallback: `/Users/jeffbezenyan/Projects/Trifecta_Trader/AI/…`.) **Before copying: verify the pack's SHA-256 matches the reviewer-APPROVED hash recorded with the authorization. After copying: `cmp` each repo copy against its source.** (The stale-copy incident of 2026-07-10 is why.)
3. Commit ("TRI-69 closure: seal Arbiter report + closure pack") and push. This commit may ride into Step 1's commit instead — either way both files land before Step 2.

## Step 1 — DEVELOP reporting commit → `$STEP1_SHA`

Edit **`docs/TASK_TRI-69_REPORT.md` §Verdict** only (prose/tables). Fold in: (a) the date-clustered 95% CI (recomputed independently; must match the sealed baseline above); (b) the realized-σ operating table — **regenerated independently** (record the MC seed, simulation count, and comparison tolerance in the report; agreement within MC noise at the stated count), labeled "Arbiter-produced 2026-07-08, independently regenerated by DEVELOP"; (c) the corrected two-sided futility statement labeled **pre-run estimate**; (d) both means labeled (date-means vs pooled); (e) ladder state + Jeff's RUN decision; (f) the sealed-160-baseline label on all existing numbers. Commit, push, **record `$STEP1_SHA`**.

## Step 2 — Baseline Codex QA (mini, from origin) → verdict artifact

- **Checkout `$STEP1_SHA`** (it contains the insertions QA must inspect). Freeze assertion, exactly:
  ```
  git diff --exit-code 91a6591 $STEP1_SHA -- scripts/score_tri69.py tests/test_tri69_scoring.py scripts/run_tri69_edge_check.py config/hybrid_llm.yaml src/hybrid_llm.py src/point_in_time.py tests/test_tri69_point_in_time.py
  git diff --name-only 91a6591 $STEP1_SHA
  ```
  First command must exit 0 (empty); second must list **only** the expected `docs/` files. Separately assert `git diff --exit-code 57c1876 91a6591 -- <same freeze list>` is empty *(known true: seal touched only `results/` — verified 2026-07-08)*.
- QA protocol (from `docs/TRI-69_ARBITER_VERDICT_REPORT.md` §7): recompute the verdict from `docs/TRI-69_scoring.json`; re-run the 13 fixtures (`PYTHONPATH=. pytest tests/test_tri69_scoring.py -q` → all pass); verify Step-1 insertions match recomputation; verify the futility bound **both halves**; PIT/leak audit on one sampled run's saved reports. **If QA re-executes the live scorer:** that is another yfinance fetch — label it *noncanonical audit-only* with its actual retrieval timestamp, write to a **temporary distinct output** (never any pinned path), and on mismatch with the sealed baseline in any verdict-relevant number (date means, p, hit, sweep, verdict string): **record the exact diff and escalate to the Arbiter immediately** — the canonical drift determination still happens at Steps 3–4, but a verdict-relevant mismatch is early warning that must not wait. Non-verdict-relevant differences: annotate in the review artifact. **No engine re-runs.**
- **Durable artifact:** QA writes `docs/TASK_TRI-69_CODEX_REVIEW.md` — reviewed SHA (`$STEP1_SHA`), each check's result, findings, verdict **APPROVED / CHANGES-REQUESTED**. Terminal commits + pushes it. CHANGES-REQUESTED → fix → re-review (§10 two-round cap applies) — do not proceed.

## Step 3 — DEVELOP tooling commit (additive only) → `$STEP3_SHA`

New files, exact names and contracts:

| File | Contract |
|---|---|
| `scripts/build_tri69_price_snapshot.py` | Fetches ONCE via the same call shape as the frozen scorer *(source: `score_tri69.py` `fetch_prices(tickers, start, end)` with `yf.download(..., auto_adjust=True)`)*: 21 symbols (20 universe tickers + SPY), `start="2026-02-17"`, `end="2026-07-03"`. Writes `docs/TRI-69_price_snapshot.json` (per symbol: trading-day index, open[], close[]; auto-adjusted) + manifest `docs/TRI-69_price_snapshot.sha256` (SHA-256 of the canonical serialization: sorted keys, compact separators). Records retrieval timestamp + yfinance version INSIDE the snapshot metadata. |
| `scripts/score_tri69_from_snapshot.py` | **Contract (strict, to prevent a second scorer):** loads the snapshot and reconstructs the exact DataFrame shape `yf.download` returns; **monkeypatches ONLY `score_tri69.fetch_prices`; then invokes the untouched `score_tri69.main()`** (same CLI plumbing, same serializer). Reimplementing any orchestration or verdict logic in the harness is a defect even if outputs agree. CLI: `python scripts/score_tri69_from_snapshot.py --eval-file <F> --snapshot docs/TRI-69_price_snapshot.json --output <O>`. |
| `scripts/assemble_tri69_final.py` | CLI: `python scripts/assemble_tri69_final.py --base results/tri69/eval_tri69-eval.json --extension results/tri69/eval_tri69-ext-eval.json --output results/tri69/eval_tri69-final-180.json`. Preserves row schema; **rejects duplicates on BOTH (ticker,date) AND run_id; asserts exactly 180 RUN ROWS** (rows, not decisions — errors/NO-DECISION remain rows). **Recomputes ALL EIGHT aggregate metadata fields — never concatenates them** *(keys verified in the sealed aggregate: `mode, config, universe, progress, total_usd, decision_counts, errors, leak_hits_total`)*: `mode` — require both inputs equal `"eval"`, output `"eval"` (hard fail otherwise) · `config` — require both inputs equal `"tri69_config_a"`, output that value (hard fail otherwise) · `universe` — output a structured pair: both source universe paths with their SHA-256 hashes (`{"base": {path, sha256}, "extension": {path, sha256}}`) · `progress: "180/180"` · summed `total_usd` · merged `decision_counts` · summed `errors` · summed `leak_hits_total` — plus provenance: both source **aggregate** paths and their SHA-256 hashes. |
| `tests/test_tri69_assembler.py` | Fixtures: happy-path 160+20→180 **asserting EXACT VALUES for all eight metadata fields** (mode `"eval"`, config `"tri69_config_a"`, the structured universe pair, progress `"180/180"`, cost sum, merged decision counts, error sum, leak total — not merely presence) plus provenance hashes; mismatched mode or config between inputs → hard fail; duplicate (ticker,date) → hard fail; duplicate run_id → hard fail; wrong count → hard fail. |
| `tests/test_tri69_snapshot_harness.py` | **Golden reproduction = FULL ARTIFACT EQUIVALENCE, not headline numbers:** harness over the snapshot + sealed `results/tri69/eval_tri69-eval.json` must produce output where `generated == json.load(open("results/tri69/scoring.json"))` (deep equality, every key: T+5 secondary, long-only view, base rate, sweep, decision rows, error metadata) **AND** `cmp generated_scoring.json results/tri69/scoring.json` passes byte-for-byte (same serializer via untouched `main()`). *(The bar is real: `results/tri69/scoring.json` and `docs/TRI-69_scoring.json` are byte-identical on the branch — verified 2026-07-10.)* This test IS the drift gate. **Harness tests never touch the network — they consume only the committed snapshot.** |

Run: `PYTHONPATH=. pytest tests/test_tri69_assembler.py tests/test_tri69_snapshot_harness.py -q` → all pass locally. Commit ("TRI-69 closure tooling: snapshot builder + harness + assembler, additive") + snapshot + manifest, push, **record `$STEP3_SHA`**. **If the golden reproduction FAILS: yfinance history has drifted since Jul 8 — STOP, commit the failing evidence, escalate to the Arbiter. No silent reconciliation.**

## Step 4 — Tooling QA (mini) → second verdict artifact

Checkout `$STEP3_SHA`. QA attacks the new code (it did not author it): re-run both fixture files (commands above, expected all-pass); independently recompute the snapshot SHA-256 against the manifest; verify the full-artifact golden reproduction (deep equality + `cmp`); verify the harness monkeypatches only `fetch_prices` and calls the untouched `main()` (read the code, not just the outputs); **price verification happens here** — spot-check sampled snapshot legs against a fresh retrieval, **honestly labeled: same provider (yfinance), so this is a consistency check, not provider-independent confirmation**; record exact differences with the retrieval timestamp; **the canonical snapshot is never replaced regardless of findings** — discrepancies are documented and escalated. Durable artifact: `docs/TASK_TRI-69_TOOLING_REVIEW.md` (reviewed SHA, fixture outputs, snapshot hash, drift result, findings, verdict). Terminal commits + pushes. Proceed only on APPROVED.

## Step 5 — Extension implementation addendum → `$STEP5_SHA`

Commit **`docs/TRI-69_extension_universe.json`**:
```json
{"dates": ["2026-06-11"],
 "tickers": ["ALB","APA","AZO","CF","ERIE","EXPD","GPC","HBAN","IBKR","LLY","MO","ORLY","SHW","SMCI","SO","SYK","TRMB","TT","VRSK","VRTX"]}
```
*(tickers verified identical to `docs/TRI-69_universe.json`, 2026-07-10)* — plus an addendum section in `docs/TASK_TRI-69_REPORT.md` documenting the one-date lock (language per §G), the exact Step-6 invocation, and the expected run-ids. The addendum's commit timestamp **must precede any extension run.** Push, **record `$STEP5_SHA`**.

## Step 6 — Extension run (M3 Max, overnight)

**Preflight (all must pass; append each check's result to `results/tri69/ext_run_console.log` BEFORE the runner starts — the `tee` below only captures the run itself):** `ollama list` shows `qwen3-coder:30b` AND `qwen3.5:9b` · Ollama healthy (`curl -s localhost:11434/api/tags` returns) · `ollama ps` shows no competing inference · `test -n "$ANTHROPIC_API_KEY"` (never print it) · `git status` clean, HEAD = `$STEP5_SHA` on `jeff/tri-69-edge-check` · ≥20 GB free disk · on AC power (`pmset -g ps | grep -q 'AC Power'`).

**Exact invocation** *(semantics verified: `main()` dispatches `run_eval(args.universe, f"{args.tag}-eval")` at `run_tri69_edge_check.py:314`; `run_eval` builds `run_id = f"{tag}-{ticker}-{date}"` and `out_file = OUT_DIR / f"eval_{tag}.json"`; built-in AAPL/NVDA/TSLA guard; resumable — existing result files are skipped)*:
```
set -o pipefail
caffeinate -is python scripts/run_tri69_edge_check.py --eval \
  --universe docs/TRI-69_extension_universe.json --tag tri69-ext \
  2>&1 | tee -a results/tri69/ext_run_console.log
```
→ run-ids `tri69-ext-eval-{TICKER}-2026-06-11`, aggregate `results/tri69/eval_tri69-ext-eval.json`. ~$1.10, ~5.7 h serial. `caffeinate -is` prevents sleep for the run's lifetime; the console log is **named and append-only (`tee -a`)** — on interruption, resume the identical command; it appends to the log and skips existing result files. **Allow uninterrupted execution except for safety or integrity failures; never delete partials.** Pre-run leak posture: PIT env forced (`TRIFECTA_POINT_IN_TIME=1` — the frozen runner sets it); the canonical snapshot is scoring-only and must never appear in any analyst context (structurally true: the frozen runner predates the snapshot file and never reads it).

## Step 7 — Post-run leak gate + assembly

1. **Zero-leak gate:** `leak_hits_total == 0` in `results/tri69/eval_tri69-ext-eval.json` AND a sampled saved-report audit of one extension run. **Any leak hit → STOP the track, escalate.** Technical failures are NO-DECISION rows under the frozen rule — they proceed.
2. Assemble: exact Step-3 command → `results/tri69/eval_tri69-final-180.json` (asserts 180 run rows, dup-rejection both keys).

## Step 8 — Final score + extension-evidence commit → `$STEP8_SHA`

1. `python scripts/score_tri69_from_snapshot.py --eval-file results/tri69/eval_tri69-final-180.json --snapshot docs/TRI-69_price_snapshot.json --output results/tri69/scoring_180.json`
2. **Verdict re-issued mechanically** from the frozen rule. Report separately: attempted rows (180) · scored runs · directional N · HOLDs · NO-DECISIONs. Update §Verdict with actuals; demote the 3–6% figures to "pre-run estimate"; keep the sealed-160 baseline in its historical subsection.
3. Mirrors: `docs/TRI-69_eval_aggregate_180.json`, `docs/TRI-69_scoring_180.json` — each proven byte-identical to its `results/` source via `cmp` (recorded in the run log).
4. **New extension-evidence commit** (force-add past the `results/` gitignore), **enumerated contents — nothing summarized as "artifacts":** **every extension result JSON that exists** (`results/*/analysis_2026-06-11_tri69_config_a_tri69-ext-eval-*.json`), including permitted `-retry` files — **for each of the 20 aggregate rows, the manifest must identify either its result JSON or its aggregate error record plus console/memory evidence; missing result files caused by recorded technical failures are valid NO-DECISION evidence and must NEVER be retried merely to satisfy an artifact count** · `results/tri69/eval_tri69-ext-eval.json` · extension memory logs (`results/tri69/memory/memory_tri69-ext-eval-*.md`) · `results/tri69/ext_run_console.log` · `results/tri69/eval_tri69-final-180.json` · `results/tri69/scoring_180.json` · both `docs/` mirrors · the updated report · **and `docs/TRI-69_extension_evidence.sha256` — a SHA-256 manifest of every file above (the manifest excludes itself from its own hash list), executed and recorded at UAT (Step 9) and re-verified at Arbiter sign-off.** References `$STEP3_SHA` for the snapshot — **does not re-commit it; `91a6591` remains immutable.** Push, **record `$STEP8_SHA`**.

## Step 9 — Behavioral UAT → `docs/TASK_TRI-69_UAT_RESULT.md`

Corrected chain, rendered for the sample: **PIT evidence → analyst reports → PM decision → scoring row ← sealed forward-price legs** (the snapshot holds *outcome* prices for scoring; it is never upstream of analysis). Sample: **ALB's extension run**; if ALB is HOLD or NO-DECISION, render its exclusion path (why it exits the directional test) **and** fall forward alphabetically (APA, AZO, …) to the first directional extension run for the complete chain. Verify `scoring_180.json` / `eval_tri69-final-180.json` / §Verdict mutual consistency; verify the snapshot harness reproduces `scoring_180.json` byte-identically (the claim attaches to the **snapshot harness**, never to the live-yfinance scorer); **execute the evidence-manifest verification (`docs/TRI-69_extension_evidence.sha256` — every listed file present, every hash matching) and record the result in the UAT document** — including the per-row evidence mapping (result JSON or error record + console/memory evidence for each of the 20 rows). Terminal commits + pushes.

## Step 10 — Arbiter sign-off → merge → Done

Package to the Arbiter: `$STEP1_SHA…$STEP8_SHA`, both QA verdicts, UAT result. Arbiter independently re-verifies (recompute from `scoring_180.json`, freeze-list diff, leak gate, addendum-precedes-run timestamps). Jeff merges to `main`; Arbiter runs `ECOSYSTEM_CONTEXT.md` §10 post-merge re-verification; ticket flips **Done** with terminal **STOP-posture declared regardless of the formal label**, anti-spin language intact; engine ECOSYSTEM_CONTEXT mirror syncs; standup babysit permanently retired.

---
*Escalated authorship, Arbiter 2026-07-10. Reviewer: attack this document with the same standard applied to v1–v3 — especially Steps 3/6/8, where I am most exposed to the same under-verification failure I sanctioned.*
