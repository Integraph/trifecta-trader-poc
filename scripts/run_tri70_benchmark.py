"""TRI-70 Step 6 — staged repeat-run benchmark harness.

Runs each config through N repeats on a ticker (subprocess per run, so state
never leaks between runs), with cache OFF and a unique run-id per repeat, then
aggregates the TRI-70 signals:

  * decision-stability  — modal decision + agreement fraction across repeats
  * extractability       — fraction yielding a parseable decision (UNKNOWN is
                           surfaced, NEVER counted as HOLD) + decision_extraction_method breakdown
  * quality              — composite mean / stdev (+ sub-scores)
  * wall-time/ticker     — mean seconds

Staging (work order): screen deep-slot variants at 1 ticker x N=3, then run the
finalists at the small watchlist x N=5. Pass the config list + ticker(s) + N.

SAFETY: analysis-only (no --execute, no --dry-run) so no order path is touched.
Configs whose ollama models aren't installed are SKIPPED (logged), so it's safe
to run before every pull finishes.

Usage:
  PYTHONPATH=".:vendor/TradingAgents" python scripts/run_tri70_benchmark.py \
      --configs bench_deep_qwen36_35b bench_deep_qwen36_27b \
      --tickers AAPL --n 3 --date 2026-06-27
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
REPO = Path(__file__).resolve().parent.parent


def _installed_models() -> set:
    try:
        resp = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        data = json.loads(resp.read())
        return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


def _config_local_models(cfg) -> list:
    """Return the ollama model tags a config depends on (empty for cloud slots)."""
    tags = []
    if cfg.tool_provider == "ollama":
        tags.append(cfg.tool_model)
    if cfg.reasoning_quick_provider == "ollama":
        tags.append(cfg.reasoning_quick_model)
    if cfg.reasoning_deep_provider == "ollama":
        tags.append(cfg.reasoning_deep_model)
    return tags


def _run_once(ticker: str, date: str, config_name: str, run_id: str) -> dict:
    """Run one analysis subprocess; return the parsed result JSON (or an error dict)."""
    env = dict(os.environ)
    env["TRIFECTA_RUN_ID_FILES"] = "1"
    env["PYTHONPATH"] = ".:vendor/TradingAgents"
    cmd = [
        sys.executable, "-m", "src.run_analysis",
        "--ticker", ticker, "--date", date,
        "--hybrid", config_name,
        "--no-cache", "--no-debug", "--no-cost-breakdown",
        "--run-id", run_id,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    result_file = REPO / "results" / ticker / f"analysis_{date}_{config_name}_{run_id}.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        data["_wall_s_subprocess"] = round(elapsed, 1)
        data["_returncode"] = proc.returncode
        return data
    # Failure: capture tail of stderr for diagnosis
    return {
        "_error": True,
        "_returncode": proc.returncode,
        "_wall_s_subprocess": round(elapsed, 1),
        "_stderr_tail": (proc.stderr or "")[-800:],
        "_stdout_tail": (proc.stdout or "")[-400:],
    }


def _aggregate(config_name: str, runs: list) -> dict:
    ok = [r for r in runs if not r.get("_error")]
    decisions = [r.get("decision", "UNKNOWN") for r in ok]
    methods = [r.get("decision_extraction_method", "missing") for r in ok]
    ratings = [r.get("pm_rating_5") for r in ok]
    quals = [r.get("quality_score", {}).get("composite") for r in ok
             if isinstance(r.get("quality_score", {}).get("composite"), (int, float))]
    walls = [r.get("elapsed_seconds") for r in ok
             if isinstance(r.get("elapsed_seconds"), (int, float))]

    n = len(runs)
    n_ok = len(ok)
    parseable = [d for d in decisions if d != "UNKNOWN"]
    modal = Counter(decisions).most_common(1)[0] if decisions else ("N/A", 0)

    return {
        "config": config_name,
        "runs": n,
        "runs_ok": n_ok,
        "errors": n - n_ok,
        "decisions": decisions,
        "modal_decision": modal[0],
        "decision_agreement": round(modal[1] / n_ok, 3) if n_ok else 0.0,
        "extractability": round(len(parseable) / n_ok, 3) if n_ok else 0.0,
        "unknowns": sum(1 for d in decisions if d == "UNKNOWN"),
        "method_breakdown": dict(Counter(methods)),
        "ratings_5": ratings,
        "quality_mean": round(statistics.mean(quals), 2) if quals else None,
        "quality_stdev": round(statistics.stdev(quals), 2) if len(quals) > 1 else 0.0,
        "quality_values": quals,
        "walltime_mean_s": round(statistics.mean(walls), 1) if walls else None,
        "walltime_values": walls,
    }


def main():
    parser = argparse.ArgumentParser(description="TRI-70 staged benchmark runner")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--tickers", nargs="+", default=["AAPL"])
    parser.add_argument("--n", type=int, default=3, help="repeats per (config,ticker)")
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/tri70_benchmark_agg.json")
    parser.add_argument("--tag", type=str, default="screen",
                        help="run-id tag prefix (e.g. screen / finalists)")
    args = parser.parse_args()

    from src.hybrid_llm import CONFIGS

    installed = _installed_models()
    summary = []
    for config_name in args.configs:
        if config_name not in CONFIGS:
            print(f"[SKIP] unknown config: {config_name}")
            continue
        cfg = CONFIGS[config_name]
        needed = _config_local_models(cfg)
        missing = [m for m in needed if m not in installed]
        if missing:
            print(f"[SKIP] {config_name}: models not installed yet: {missing}")
            summary.append({"config": config_name, "skipped": True, "missing": missing})
            continue

        for ticker in args.tickers:
            runs = []
            for i in range(args.n):
                run_id = f"{args.tag}-{config_name}-{ticker}-r{i+1}"
                print(f"\n>>> {config_name} | {ticker} | repeat {i+1}/{args.n} | run_id={run_id}")
                r = _run_once(ticker, args.date, config_name, run_id)
                if r.get("_error"):
                    print(f"    ERROR rc={r['_returncode']} ({r['_wall_s_subprocess']}s)")
                    print(f"    stderr: {r['_stderr_tail'][-300:]}")
                else:
                    print(f"    decision={r.get('decision')} rating={r.get('pm_rating_5')} "
                          f"method={r.get('decision_extraction_method')} "
                          f"quality={r.get('quality_score',{}).get('composite')} "
                          f"wall={r.get('elapsed_seconds')}s")
                runs.append(r)
                # persist incrementally so progress survives interruption
                agg = _aggregate(f"{config_name}@{ticker}", runs)
                _dump(args.output, summary + [agg])
            summary.append(_aggregate(f"{config_name}@{ticker}", runs))
            _dump(args.output, summary)

    _dump(args.output, summary)
    print(f"\nAggregate written to {args.output}")
    _print_summary(summary)


def _dump(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _print_summary(summary):
    print(f"\n{'='*90}\nTRI-70 BENCHMARK SUMMARY\n{'='*90}")
    hdr = f"{'Config@Ticker':<32}{'Modal':>7}{'Agree':>7}{'Extract':>8}{'Q mean':>8}{'Q σ':>6}{'Wall s':>8}"
    print(hdr)
    print("-" * 90)
    for s in summary:
        if s.get("skipped"):
            print(f"{s['config']:<32}  SKIPPED (missing {s.get('missing')})")
            continue
        print(f"{s['config']:<32}{str(s['modal_decision']):>7}{s['decision_agreement']:>7}"
              f"{s['extractability']:>8}{str(s['quality_mean']):>8}{str(s['quality_stdev']):>6}"
              f"{str(s['walltime_mean_s']):>8}")


if __name__ == "__main__":
    main()
