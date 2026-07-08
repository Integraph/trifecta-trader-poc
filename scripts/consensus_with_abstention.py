"""Apply the consensus-with-abstention rule to saved result artifacts.

Examples:
  python scripts/consensus_with_abstention.py results/AAPL/analysis_*.json

  python scripts/consensus_with_abstention.py results/tri70_finalist_agg.json \
      --output results/tri70_consensus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.consensus import consensus_with_abstention


def _rows_from_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "decision" in data:
        row = dict(data)
        row["_consensus_group"] = _group_for_result(row)
        return [row]
    if isinstance(data, list):
        rows: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and "decisions" in item:
                qualities = item.get("quality_values") or []
                for idx, decision in enumerate(item.get("decisions") or []):
                    quality = qualities[idx] if idx < len(qualities) else None
                    rows.append({
                        "decision": decision,
                        "quality_score": {"composite": quality},
                        "source_config": item.get("config"),
                        "_consensus_group": item.get("config") or "aggregate",
                    })
            elif isinstance(item, dict) and "decision" in item:
                rows.append(item)
        return rows
    raise ValueError(f"{path} is not a supported result JSON shape")


def _group_for_result(row: dict[str, Any]) -> str:
    config = row.get("hybrid_config") or row.get("provider") or "unknown_config"
    ticker = row.get("ticker") or "unknown_ticker"
    date = row.get("trade_date") or "unknown_date"
    return f"{config}@{ticker}@{date}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Consensus decision with NO_TRADE abstention")
    parser.add_argument("results", nargs="+", help="Result JSON files or aggregate JSON files")
    parser.add_argument("--min-agreement", type=float, default=0.8)
    parser.add_argument("--min-quality", type=float, default=8.0)
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for raw_path in args.results:
        rows.extend(_rows_from_file(Path(raw_path)))

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("_consensus_group") or "all", []).append(row)

    grouped_out = []
    for group, group_rows in sorted(groups.items()):
        verdict = consensus_with_abstention(
            group_rows,
            min_agreement=args.min_agreement,
            min_quality=args.min_quality,
            min_runs=args.min_runs,
        )
        verdict["group"] = group
        grouped_out.append(verdict)

    out = grouped_out[0] if len(grouped_out) == 1 else grouped_out

    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
