"""TRI-69 — edge-check runner (determinism battery + point-in-time eval).

Two modes, both subprocess-per-run (state never leaks between runs), both
forcing the TRI-69 no-leak environment on every child:

  TRIFECTA_POINT_IN_TIME=1   point-in-time mode, as-of the run's --date
  TRIFECTA_SAVE_REPORTS=1    persist the four analyst reports (leak audit)
  TRIFECTA_RUN_ID_FILES=1    run-id-suffixed result files (never clobber)
  TRADINGAGENTS_MEMORY_LOG_PATH=<fresh per-run file>
                             no cross-run memory contamination: every run
                             starts from an empty memory log, so run order
                             cannot leak decisions/reflections across points
                             (and the sandbox-permissions failure from the
                             TRI-70 QA review can't recur)

Battery mode (Step 4 of the pre-reg — burned points, excluded from eval):
  python scripts/run_tri69_edge_check.py --battery --n 3 \
      --tickers AAPL NVDA TSLA --date 2026-06-27
  PASS = decision-identical across all N repeats for EVERY ticker.
  A single flip = FAIL (checkpoint; do not run the eval).

Eval mode (Step 5 — one deterministic run per (ticker,date)):
  python scripts/run_tri69_edge_check.py --eval --universe docs/TRI-69_universe.json
  The universe file is the PRE-REGISTERED {"dates": [...], "tickers": [...]}
  committed before any eval run. Resumable: existing result files are kept.

SAFETY: analysis-only (never --execute / --dry-run); Config A is TEST-ONLY.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_NAME = "tri69_config_a"
OUT_DIR = REPO / "results" / "tri69"

# Per-run hard timeout. Observed healthy Config A runs take ~12 min; a run
# past this is wedged (decode loop or dead Ollama socket) — kill and record.
# The 2026-07-04 battery lost ~29h to exactly this failure mode.
RUN_TIMEOUT_S = 3600

# Mechanical infra-failure classifier (pre-registered): ONLY a connection
# error to the local Ollama server earns one retry — it is not a draw from
# the model, so retrying cannot bias the signal. Everything else (decode
# loop, timeout, extraction UNKNOWN, other exceptions) is NO-DECISION with
# NO retry: at temp=0 those reproduce deterministically.
INFRA_RE = re.compile(r"APIConnectionError|Connection error|Connection refused|ConnectError")

# Leak audit on saved analyst reports (defense-in-depth on top of the
# in-process guard in src/point_in_time.py).
INFO_KEYS_RE = re.compile(
    r"fiftyTwoWeek\w*|fiftyDayAverage|twoHundredDayAverage"
    r"|trailingPE\b|trailingEps\b|trailingPegRatio\b|trailingAnnualDividend\w*"
)
FUND_LABELS_RE = re.compile(
    r"52 Week High:|52 Week Low:|50 Day Average:|200 Day Average:"
    r"|PE Ratio \(TTM\):|EPS \(TTM\):"
)


def _leak_audit(data: dict) -> list:
    """Scan a result JSON's analyst reports for forbidden fields."""
    hits = []
    reports = data.get("analyst_reports") or {}
    for name, text in reports.items():
        if not isinstance(text, str):
            continue
        m = INFO_KEYS_RE.search(text)
        if m:
            hits.append(f"{name}: info-key {m.group(0)!r}")
        if name == "fundamentals_report":
            m2 = FUND_LABELS_RE.search(text)
            if m2:
                hits.append(f"{name}: formatted label {m2.group(0)!r}")
    return hits


def _run_once(ticker: str, date: str, run_id: str, memdir: Path) -> dict:
    env = dict(os.environ)
    env["TRIFECTA_POINT_IN_TIME"] = "1"
    env["TRIFECTA_SAVE_REPORTS"] = "1"
    env["TRIFECTA_RUN_ID_FILES"] = "1"
    env["PYTHONPATH"] = ".:vendor/TradingAgents"
    memdir.mkdir(parents=True, exist_ok=True)
    env["TRADINGAGENTS_MEMORY_LOG_PATH"] = str(memdir / f"memory_{run_id}.md")

    cmd = [
        sys.executable, "-m", "src.run_analysis",
        "--ticker", ticker, "--date", date,
        "--hybrid", CONFIG_NAME,
        "--no-cache", "--no-debug", "--no-cost-breakdown",
        "--run-id", run_id,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired as te:
        def _tail(stream, n):
            if stream is None:
                return ""
            if isinstance(stream, bytes):
                return stream[-n:].decode(errors="replace")
            return stream[-n:]
        # Keep a generous stdout tail: with --debug on, the last graph-node
        # line localizes WHERE a wedged run stalled.
        return {
            "_error": True, "_timeout": True, "ticker": ticker,
            "trade_date": date, "run_id": run_id, "_returncode": None,
            "_wall_s_subprocess": round(time.time() - t0, 1),
            "_stderr_tail": _tail(te.stderr, 1200),
            "_stdout_tail": _tail(te.stdout, 4000),
        }
    elapsed = time.time() - t0

    result_file = REPO / "results" / ticker / f"analysis_{date}_{CONFIG_NAME}_{run_id}.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        data["_wall_s_subprocess"] = round(elapsed, 1)
        data["_returncode"] = proc.returncode
        data["_leak_audit_hits"] = _leak_audit(data)
        return data
    return {
        "_error": True, "ticker": ticker, "trade_date": date, "run_id": run_id,
        "_returncode": proc.returncode,
        "_wall_s_subprocess": round(elapsed, 1),
        "_stderr_tail": (proc.stderr or "")[-1200:],
        "_stdout_tail": (proc.stdout or "")[-400:],
    }


def _existing_result(ticker: str, date: str, run_id: str):
    """Load a prior result for run_id (or its -retry) if one exists (resume)."""
    for rid in (run_id, f"{run_id}-retry"):
        f = REPO / "results" / ticker / f"analysis_{date}_{CONFIG_NAME}_{rid}.json"
        if f.exists():
            with open(f) as fh:
                data = json.load(fh)
            data["_resumed"] = True
            data["_leak_audit_hits"] = _leak_audit(data)
            return data
    return None


def _run_with_infra_retry(ticker: str, date: str, run_id: str, memdir: Path) -> dict:
    """One run; a single retry ONLY on the mechanical infra signature."""
    r = _run_once(ticker, date, run_id, memdir)
    if (r.get("_error") and not r.get("_timeout")
            and INFRA_RE.search(r.get("_stderr_tail") or "")):
        print(f"    INFRA failure (connection) — one pre-registered retry", flush=True)
        r2 = _run_once(ticker, date, f"{run_id}-retry", memdir)
        r2["_infra_retry_of"] = run_id
        return r2
    return r


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _cost_of(r: dict) -> float:
    return (r.get("cost_breakdown") or {}).get("total_usd") or 0.0


def _battery_ticker_agg(runs: list) -> dict:
    decisions = [x.get("decision", "ERROR") if not x.get("_error") else "ERROR"
                 for x in runs]
    return {
        "decisions": decisions,
        "identical": len(set(decisions)) == 1 and "ERROR" not in decisions,
        "leak_hits": sum(len(x.get("_leak_audit_hits") or []) for x in runs),
        "run_ids": [x.get("run_id") for x in runs],
        "walls_s": [x.get("elapsed_seconds") for x in runs],
        "infra_retries": sum(1 for x in runs if x.get("_infra_retry_of")),
    }


def run_battery(tickers, date, n, tag):
    out_file = OUT_DIR / f"battery_{tag}.json"
    memdir = OUT_DIR / "memory"
    summary = {"mode": "battery", "config": CONFIG_NAME, "date": date,
               "n": n, "tickers": {}, "pass": None, "total_usd": 0.0}
    all_identical = True
    for ticker in tickers:
        runs = []
        for i in range(n):
            run_id = f"{tag}-{ticker}-r{i+1}"
            prior = _existing_result(ticker, date, run_id)
            if prior is not None:
                print(f"\n>>> BATTERY {ticker} repeat {i+1}/{n} RESUMED "
                      f"decision={prior.get('decision')}", flush=True)
                runs.append(prior)
                summary["tickers"][ticker] = _battery_ticker_agg(runs)
                _dump(out_file, summary)
                continue
            print(f"\n>>> BATTERY {ticker} repeat {i+1}/{n} run_id={run_id}", flush=True)
            r = _run_with_infra_retry(ticker, date, run_id, memdir)
            if r.get("_error"):
                print(f"    ERROR rc={r['_returncode']}: {r['_stderr_tail'][-300:]}", flush=True)
            else:
                print(f"    decision={r.get('decision')} wall={r.get('elapsed_seconds')}s "
                      f"leaks={r['_leak_audit_hits'] or 'none'}", flush=True)
            runs.append(r)
            summary["total_usd"] = round(summary["total_usd"] + _cost_of(r), 4)
            summary["tickers"][ticker] = _battery_ticker_agg(runs)
            _dump(out_file, summary)
        if not summary["tickers"][ticker]["identical"]:
            all_identical = False
    summary["pass"] = all_identical and all(
        t["leak_hits"] == 0 for t in summary["tickers"].values())
    _dump(out_file, summary)
    print(f"\nBATTERY {'PASS' if summary['pass'] else 'FAIL'} — {out_file}", flush=True)
    for tk, t in summary["tickers"].items():
        print(f"  {tk}: {t['decisions']} identical={t['identical']} leaks={t['leak_hits']}")
    return 0 if summary["pass"] else 1


def run_eval(universe_file, tag):
    with open(universe_file) as f:
        universe = json.load(f)
    dates = universe["dates"]
    tickers = universe["tickers"]
    excluded = {"AAPL", "NVDA", "TSLA"}
    banned = excluded & set(tickers)
    if banned:
        raise SystemExit(f"pre-registered exclusions present in universe: {banned}")

    out_file = OUT_DIR / f"eval_{tag}.json"
    memdir = OUT_DIR / "memory"
    rows = []
    total = len(dates) * len(tickers)
    done = 0
    total_usd = 0.0
    for date in dates:
        for ticker in tickers:
            done += 1
            run_id = f"{tag}-{ticker}-{date}"
            r = _existing_result(ticker, date, run_id)
            if r is not None:
                print(f"[{done}/{total}] SKIP (exists) {ticker}@{date}", flush=True)
            else:
                print(f"\n[{done}/{total}] >>> {ticker}@{date} run_id={run_id}", flush=True)
                r = _run_with_infra_retry(ticker, date, run_id, memdir)
                if r.get("_error"):
                    print(f"    ERROR rc={r['_returncode']}: "
                          f"{r['_stderr_tail'][-300:]}", flush=True)
                else:
                    print(f"    decision={r.get('decision')} "
                          f"wall={r.get('elapsed_seconds')}s "
                          f"leaks={r['_leak_audit_hits'] or 'none'}", flush=True)
            total_usd += _cost_of(r)
            rows.append({
                "ticker": ticker, "date": date, "run_id": run_id,
                "decision": r.get("decision", "ERROR"),
                "pm_rating_5": r.get("pm_rating_5"),
                "error": bool(r.get("_error")),
                "leak_hits": r.get("_leak_audit_hits") or [],
                "elapsed_seconds": r.get("elapsed_seconds"),
                "quality": (r.get("quality_score") or {}).get("composite"),
                "cost_usd": _cost_of(r),
            })
            _dump(out_file, {
                "mode": "eval", "config": CONFIG_NAME, "universe": str(universe_file),
                "progress": f"{done}/{total}", "total_usd": round(total_usd, 4),
                "decision_counts": dict(Counter(x["decision"] for x in rows)),
                "errors": sum(1 for x in rows if x["error"]),
                "leak_hits_total": sum(len(x["leak_hits"]) for x in rows),
                "rows": rows,
            })
    print(f"\nEVAL COMPLETE — {out_file}", flush=True)
    print(f"decisions: {dict(Counter(x['decision'] for x in rows))} "
          f"errors={sum(1 for x in rows if x['error'])} "
          f"cost=${total_usd:.2f}", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description="TRI-69 edge-check runner")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--battery", action="store_true")
    mode.add_argument("--eval", action="store_true")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "TSLA"])
    p.add_argument("--date", type=str, default="2026-06-27")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--universe", type=str, default="docs/TRI-69_universe.json")
    p.add_argument("--tag", type=str, default="tri69")
    args = p.parse_args()

    from src.hybrid_llm import CONFIGS
    if CONFIG_NAME not in CONFIGS:
        raise SystemExit(f"{CONFIG_NAME} missing — run scripts/build_tri69_config.py first")
    cfg = CONFIGS[CONFIG_NAME]
    if cfg.temperature != 0.0:
        raise SystemExit(f"{CONFIG_NAME}.temperature={cfg.temperature!r} — must be 0.0")

    if args.battery:
        sys.exit(run_battery(args.tickers, args.date, args.n, f"{args.tag}-battery"))
    sys.exit(run_eval(args.universe, f"{args.tag}-eval"))


if __name__ == "__main__":
    main()
