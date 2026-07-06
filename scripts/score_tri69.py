"""TRI-69 — scoring per the pre-registered spec in docs/TASK_TRI-69_REPORT.md.

Implements EXACTLY the pre-registered rules (no alternatives, no A/B):

  Horizon      : primary T+10 — entry Open(t+1), exit Close(t+10), trading
                 days after the decision date t; auto-adjusted prices
                 (dividends in). T+5 computed as descriptive secondary.
  Cost         : c = 10 bps/side -> 20 bps round trip at 1x, charged to the
                 signal leg. Sensitivity sweep at 1x/2x/3x reported.
  Per decision : directional only (BUY s=+1, SELL s=-1; HOLD excluded):
                 e_i = s_i * (r_ticker - r_SPY) - c_roundtrip
  Date cluster : X_t = mean(e_i) over date t's directional decisions.
  Primary test : exact one-sided sign-flip permutation over dates on
                 mean(X_t) (all 2^D sign assignments; p = fraction of
                 permuted means >= observed). alpha = 0.05. With D=8
                 pre-registered dates the power floor is 1/256 ~ 0.0039.
  NO-DECISION  : runs that fail technically (decode loop / timeout / API
                 error / extraction UNKNOWN) are recorded as NO-DECISION,
                 excluded from the directional test exactly like HOLD,
                 counted + dual-reported, and feed the expansion rule via
                 the min directional-N gate. No retries: at temp=0 a
                 failure reproduces deterministically.
  Secondary    : directional hit rate (e_i^gross = s_i*(r_i - r_SPY) > 0)
                 vs realized base rate p0 = max(all-BUY, all-SELL accuracy
                 on the same decisions), date-clustered by the same
                 sign-flip machinery on per-date (hit_t - p0).
  Verdict rule : PROMISING  iff p <= 0.05 and mean(X_t) > 0 at 1x and
                              mean(X_t) > 0 at 2x cost.
                 STOP-PIVOT iff mean(X_t) <= 0 and hit_rate <= p0 and the
                              min directional-N gate was met.
                 else INCONCLUSIVE (report directional N needed).
  Min-N gate   : >= 20 directional decisions AND >= 6 of 8 dates with >= 1
                 directional decision; otherwise INCONCLUSIVE-by-HOLD and
                 the pre-registered expansion set must be run instead of a
                 verdict.

Usage:
  PYTHONPATH=".:vendor/TradingAgents" python scripts/score_tri69.py \
      --eval-file results/tri69/eval_tri69-eval.json \
      --output results/tri69/scoring.json
"""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

COST_PER_SIDE_BPS = 10.0          # 1x slippage+cost, per side
ALPHA = 0.05
MIN_DIRECTIONAL_N = 20
MIN_DATES_WITH_DIRECTIONAL = 6
HORIZONS = {"T+10": 10, "T+5": 5}
PRIMARY_HORIZON = "T+10"


def fetch_prices(tickers, start, end):
    data = yf.download(sorted(set(tickers)), start=start, end=end,
                       progress=False, auto_adjust=True)
    return data


def window_return(prices, ticker, decision_date, horizon_days):
    """Return over [Open(t+1), Close(t+horizon)] in trading days after t.

    Returns (ret, entry_date, exit_date) or (None, why, None) when the
    window is unrealized/unavailable (dual-reported, never silently dropped).
    """
    opens = prices["Open"][ticker].dropna()
    closes = prices["Close"][ticker].dropna()
    days = [d.strftime("%Y-%m-%d") for d in closes.index]
    after = [d for d in days if d > decision_date]
    if len(after) < horizon_days:
        return None, f"unrealized (only {len(after)} trading days after t)", None
    entry_date, exit_date = after[0], after[horizon_days - 1]
    entry = float(opens.loc[entry_date])
    exitp = float(closes.loc[exit_date])
    return (exitp - entry) / entry, entry_date, exit_date


def sign_flip_pvalue(date_stats):
    """Exact one-sided sign-flip permutation p-value for mean(X_t) > 0."""
    D = len(date_stats)
    observed = sum(date_stats) / D
    count = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=D):
        m = sum(s * abs(x) for s, x in zip(signs, date_stats)) / D
        if m >= observed - 1e-15:
            count += 1
        total += 1
    return count / total


def score(rows, prices, cost_mult=1.0, horizon_key=PRIMARY_HORIZON):
    """Score directional decisions; returns dict with per-date and aggregate."""
    c_rt = 2 * COST_PER_SIDE_BPS * cost_mult / 10_000.0
    h = HORIZONS[horizon_key]
    per_date = {}
    decisions_out = []
    unrealized = []
    for r in rows:
        d = r["decision"]
        if d not in ("BUY", "SELL"):
            continue
        s = 1 if d == "BUY" else -1
        r_tkr, entry_date, exit_date = window_return(prices, r["ticker"], r["date"], h)
        r_spy, _, _ = window_return(prices, "SPY", r["date"], h)
        if r_tkr is None or r_spy is None:
            unrealized.append({**r, "why": entry_date or "SPY window unavailable"})
            continue
        gross = s * (r_tkr - r_spy)
        e = gross - c_rt
        per_date.setdefault(r["date"], []).append(e)
        decisions_out.append({
            "ticker": r["ticker"], "date": r["date"], "decision": d,
            "entry": entry_date, "exit": exit_date,
            "r_ticker": round(r_tkr, 6), "r_spy": round(r_spy, 6),
            "gross_signed_excess": round(gross, 6), "net": round(e, 6),
            "hit": gross > 0,
        })
    date_stats = {t: sum(v) / len(v) for t, v in sorted(per_date.items())}
    X = list(date_stats.values())
    out = {
        "horizon": horizon_key, "cost_mult": cost_mult,
        "cost_roundtrip_bps": round(c_rt * 10_000, 1),
        "directional_n": len(decisions_out),
        "dates_with_directional": len(X),
        "per_date_mean_net_excess": {t: round(x, 6) for t, x in date_stats.items()},
        "mean_of_date_means": round(sum(X) / len(X), 6) if X else None,
        "p_sign_flip_one_sided": round(sign_flip_pvalue(X), 6) if X else None,
        "hit_rate": (round(sum(d["hit"] for d in decisions_out) / len(decisions_out), 4)
                     if decisions_out else None),
        "unrealized": unrealized,
        "decisions": decisions_out,
    }
    return out


def long_only_view(rows, prices, cost_mult=1.0, horizon_key=PRIMARY_HORIZON):
    """DESCRIPTIVE ONLY (no verdict weight): the long-only implementable
    portfolio per integrity doc §5 — BUY -> long (net of round-trip cost);
    SELL on an unheld name is a P&L NO-OP (cash, 0), as are HOLD and
    NO-DECISION. Compared per date against equal-weight buy-and-hold of ALL
    scored names that date (same window, same cost)."""
    c_rt = 2 * COST_PER_SIDE_BPS * cost_mult / 10_000.0
    h = HORIZONS[horizon_key]
    strat, bench = {}, {}
    for r in rows:
        r_tkr, _, _ = window_return(prices, r["ticker"], r["date"], h)
        if r_tkr is None:
            continue
        strat.setdefault(r["date"], []).append(
            (r_tkr - c_rt) if r["decision"] == "BUY" else 0.0)
        bench.setdefault(r["date"], []).append(r_tkr - c_rt)
    per_date = {
        t: {"strategy": round(sum(strat[t]) / len(strat[t]), 6),
            "buy_and_hold": round(sum(bench[t]) / len(bench[t]), 6)}
        for t in sorted(strat)
    }
    n = len(per_date)
    return {
        "per_date": per_date,
        "mean_strategy": round(sum(v["strategy"] for v in per_date.values()) / n, 6) if n else None,
        "mean_buy_and_hold": round(sum(v["buy_and_hold"] for v in per_date.values()) / n, 6) if n else None,
    }


def base_rate(rows, prices, horizon_key=PRIMARY_HORIZON):
    """Realized base rate p0 on the SAME directional (ticker,date) set:
    accuracy of the best constant-sign strategy (all-BUY vs all-SELL)."""
    h = HORIZONS[horizon_key]
    gross_signs = []
    for r in rows:
        if r["decision"] not in ("BUY", "SELL"):
            continue
        r_tkr, _, _ = window_return(prices, r["ticker"], r["date"], h)
        r_spy, _, _ = window_return(prices, "SPY", r["date"], h)
        if r_tkr is None or r_spy is None:
            continue
        gross_signs.append(1 if (r_tkr - r_spy) > 0 else 0)
    if not gross_signs:
        return None
    frac_up = sum(gross_signs) / len(gross_signs)
    return max(frac_up, 1 - frac_up)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-file", required=True)
    p.add_argument("--output", default="results/tri69/scoring.json")
    args = p.parse_args()

    with open(args.eval_file) as f:
        ev = json.load(f)
    rows = [r for r in ev["rows"] if not r["error"]]
    errors = [r for r in ev["rows"] if r["error"]]
    tickers = sorted({r["ticker"] for r in rows} | {"SPY"})
    dates = sorted({r["date"] for r in rows})

    prices = fetch_prices(tickers, start=min(dates), end="2026-07-03")

    from collections import Counter
    decision_counts = dict(Counter(r["decision"] for r in rows))
    n_scored = len(rows)
    hold_rate = decision_counts.get("HOLD", 0) / n_scored if n_scored else None
    # NO-DECISION (pre-registered technical-failure rule): subprocess errors
    # plus scored runs whose decision is not BUY/SELL/HOLD (e.g. UNKNOWN).
    # Excluded from the directional test exactly like HOLD; dual-reported.
    unknown_rows = [r for r in rows if r["decision"] not in ("BUY", "SELL", "HOLD")]
    no_decision_count = len(errors) + len(unknown_rows)

    primary = score(rows, prices, 1.0, PRIMARY_HORIZON)
    sweep = {f"{m}x": score(rows, prices, m, PRIMARY_HORIZON)
             for m in (1.0, 2.0, 3.0)}
    t5 = score(rows, prices, 1.0, "T+5")
    p0 = base_rate(rows, prices, PRIMARY_HORIZON)

    min_n_met = (primary["directional_n"] >= MIN_DIRECTIONAL_N
                 and primary["dates_with_directional"] >= MIN_DATES_WITH_DIRECTIONAL)

    mean_x = primary["mean_of_date_means"]
    pval = primary["p_sign_flip_one_sided"]
    mean_x_2x = sweep["2.0x"]["mean_of_date_means"]
    hit = primary["hit_rate"]

    if not min_n_met:
        verdict = "INCONCLUSIVE-BY-HOLD (run pre-registered expansion set)"
    elif (pval is not None and pval <= ALPHA and mean_x is not None
          and mean_x > 0 and mean_x_2x is not None and mean_x_2x > 0):
        verdict = "PROMISING — run the larger out-of-sample test"
    elif (mean_x is not None and mean_x <= 0
          and hit is not None and p0 is not None and hit <= p0):
        verdict = "STOP-PIVOT (no edge / anti-predictive)"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "pre_registered": {
            "alpha": ALPHA, "primary_horizon": PRIMARY_HORIZON,
            "cost_per_side_bps": COST_PER_SIDE_BPS,
            "min_directional_n": MIN_DIRECTIONAL_N,
            "min_dates_with_directional": MIN_DATES_WITH_DIRECTIONAL,
        },
        "n_scored_runs": n_scored, "n_error_runs": len(errors),
        "error_runs": [{"ticker": r["ticker"], "date": r["date"]} for r in errors],
        "no_decision_count": no_decision_count,
        "no_decision_runs": ([{"ticker": r["ticker"], "date": r["date"],
                               "decision": r["decision"]} for r in unknown_rows]
                             + [{"ticker": r["ticker"], "date": r["date"],
                                 "decision": "ERROR"} for r in errors]),
        "decision_counts": decision_counts,
        "hold_rate": round(hold_rate, 4) if hold_rate is not None else None,
        "base_rate_p0": round(p0, 4) if p0 is not None else None,
        "primary_T10_1x": primary,
        "slippage_sweep_T10": {k: {kk: v[kk] for kk in (
            "cost_roundtrip_bps", "mean_of_date_means",
            "p_sign_flip_one_sided", "hit_rate")} for k, v in sweep.items()},
        "secondary_T5_1x": {k: t5[k] for k in (
            "directional_n", "mean_of_date_means",
            "p_sign_flip_one_sided", "hit_rate")},
        "descriptive_long_only_T10_1x": long_only_view(rows, prices, 1.0,
                                                       PRIMARY_HORIZON),
        "min_n_gate_met": min_n_met,
        "verdict": verdict,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in (
        "decision_counts", "hold_rate", "base_rate_p0", "min_n_gate_met",
        "verdict")}, indent=2))
    print("primary:", json.dumps({k: primary[k] for k in (
        "directional_n", "dates_with_directional", "mean_of_date_means",
        "p_sign_flip_one_sided", "hit_rate",
        "per_date_mean_net_excess")}, indent=2))
    print(f"full scoring -> {args.output}")


if __name__ == "__main__":
    main()
