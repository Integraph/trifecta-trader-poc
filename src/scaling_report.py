"""
Generate a scaling report comparing local model sizes.

Usage:
    python -m src.scaling_report results/model_scaling_comparison.txt
"""

import sys
import re
from pathlib import Path


def parse_comparison_output(filepath: str) -> dict:
    """Parse the test output to extract metrics per model."""
    content = Path(filepath).read_text()

    data = {
        "raw": content,
        "models": {},
        "timing": {},
    }

    timing_pattern = re.compile(
        r"(\S+)\s+/\s+(bull|bear|trader):\s+([\d.]+)s,\s+(\d+)\s+words"
    )
    for match in timing_pattern.finditer(content):
        model, role, elapsed, words = match.groups()
        if model not in data["timing"]:
            data["timing"][model] = {}
        data["timing"][model][role] = {
            "elapsed": float(elapsed),
            "words": int(words),
        }

    table_pattern = re.compile(
        r"^(\S+)\s+(bull|bear|trader)\s+(\d+)\s+(\d+)\s+"
        r"(\w*)\s+(Yes|No)\s+(Yes|No)\s+([\d.]+)",
        re.MULTILINE,
    )
    for match in table_pattern.finditer(content):
        model, role, words, numbers, decision, stop_loss, target, time_s = match.groups()
        if model not in data["models"]:
            data["models"][model] = {}
        data["models"][model][role] = {
            "words": int(words),
            "numbers": int(numbers),
            "decision": decision,
            "stop_loss": stop_loss == "Yes",
            "target": target == "Yes",
            "time": float(time_s),
        }

    return data


def generate_report(data: dict) -> str:
    """Generate a formatted scaling report."""
    lines = []
    lines.append("=" * 80)
    lines.append("LOCAL MODEL SCALING REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This report summarizes quality and performance across local model sizes,")
    lines.append("compared against Claude Sonnet 4.5 as the cloud baseline.")
    lines.append("")

    if data["timing"]:
        lines.append("TIMING SUMMARY (seconds per prompt):")
        lines.append(f"{'Model':<22} {'Bull':>8} {'Bear':>8} {'Trader':>8} {'Total':>8}")
        lines.append("-" * 60)
        for model, timings in data["timing"].items():
            bull = timings.get("bull", {}).get("elapsed", 0)
            bear = timings.get("bear", {}).get("elapsed", 0)
            trader = timings.get("trader", {}).get("elapsed", 0)
            total = bull + bear + trader
            lines.append(f"{model:<22} {bull:>8.1f} {bear:>8.1f} {trader:>8.1f} {total:>8.1f}")
        lines.append("")

    if data["models"]:
        lines.append("QUALITY SUMMARY (trader prompt):")
        lines.append(f"{'Model':<22} {'Words':>7} {'Numbers':>8} {'Decision':>10} {'StopLoss':>9} {'Target':>8}")
        lines.append("-" * 70)
        for model, roles in data["models"].items():
            trader = roles.get("trader", {})
            if trader:
                lines.append(
                    f"{model:<22} {trader.get('words', 0):>7} "
                    f"{trader.get('numbers', 0):>8} "
                    f"{trader.get('decision', ''):>10} "
                    f"{'Yes' if trader.get('stop_loss') else 'No':>9} "
                    f"{'Yes' if trader.get('target') else 'No':>8}"
                )
        lines.append("")

    lines.append("See results/model_scaling_comparison.txt for full test output.")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.scaling_report <comparison_output.txt>")
        sys.exit(1)

    data = parse_comparison_output(sys.argv[1])
    print(generate_report(data))


if __name__ == "__main__":
    main()
