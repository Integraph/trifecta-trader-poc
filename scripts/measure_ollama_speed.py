"""Measure tokens/second for each Ollama model on a financial reasoning prompt.

Usage:
    python scripts/measure_ollama_speed.py                 # TRI-70 candidate set
    python scripts/measure_ollama_speed.py --all           # every installed Ollama model
    python scripts/measure_ollama_speed.py --models qwen3.6:35b deepseek-r1:14b

Candidate-driven (TRI-70 Step 5): the model list is no longer hardcoded to the
qwen2.5/3.5 sweep. It defaults to the TRI-70 benchmark candidates (deep / tool /
quick slots), accepts an explicit --models list, or --all to auto-discover every
installed model from Ollama's registry. Not-installed candidates are skipped
cleanly, so it's safe to run mid-pull.

Supersedes the speed numbers in Tasks 005 / 010 / 011 (those measured the
qwen2.5/3.5 generation on the pre-v0.3.0 engine).
"""

import argparse
import time
import json
import sys
import requests

OLLAMA_URL = "http://localhost:11434"

# TRI-70 candidate set, by slot (verify every tag at pull time). Not-installed
# entries are skipped automatically, so the whole set can be listed here even
# while Wave-1 pulls are still in flight.
CANDIDATE_MODELS = [
    # Baseline / current production (for a like-for-like delta)
    ("qwen2.5:14b",     "Baseline (prior production)"),
    # Deep / Risk-Judge slot (quality-critical)
    ("qwen3.6:35b",     "Deep — Qwen 3.6 35B MoE (~3B active)"),
    ("qwen3.6:27b",     "Deep — Qwen 3.6 27B dense"),
    ("gpt-oss:120b",    "Deep — gpt-oss 120B"),
    ("deepseek-r1:8b",  "Deep — DeepSeek-R1 8B (R1-0528 refresh)"),
    ("deepseek-r1:14b", "Deep — DeepSeek-R1 14B distill (qwen2.5 base)"),
    ("deepseek-r1:32b", "Deep — DeepSeek-R1 32B distill (qwen2.5 base)"),
    ("deepseek-r1:70b", "Deep — DeepSeek-R1 70B distill (llama3.3 base)"),
    # Tool slot (function-calling; gated separately in Step 2)
    ("qwen3-coder:30b", "Tool — Qwen3-Coder 30B"),
    ("gpt-oss:20b",     "Tool/Quick — gpt-oss 20B"),
    ("llama3.3:70b",    "Tool — Llama 3.3 70B"),
    ("gemma-4:27b",     "Tool — Gemma-4 27B (tag TBD)"),
    # Quick slot (already present)
    ("qwen3.5:9b",      "Quick — Qwen 3.5 9B dense"),
]

PROMPT = """Analyze the following financial data and provide a detailed risk assessment:

- Revenue: $4.2B (up 12% YoY)
- Net Income: $380M (down 5% YoY)
- Debt/Equity: 1.8 (industry avg: 1.2)
- Free Cash Flow: $520M
- P/E Ratio: 28.5 (industry avg: 22)
- Short Interest: 8.2%
- RSI: 64 (moderate momentum)
- MACD: Bearish crossover, -0.8 below signal line
- 50-day SMA: $48.20, current price $47.50 (below SMA)

Provide a detailed risk assessment covering:
1. Financial health analysis
2. Valuation risk (premium vs peers)
3. Technical momentum signals
4. Market sentiment from short interest
5. Overall risk rating from 1-10 with rationale

Be specific and cite the data points above."""


def check_ollama_running():
    """Return True if Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def model_is_available(model_name: str) -> bool:
    """Check if a model is listed in Ollama's model registry."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return model_name in models
    except Exception:
        return False


def measure_model(model_name: str, label: str) -> dict:
    """Run a single generation and return speed metrics."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({label})")
    print(f"{'='*60}")

    if not model_is_available(model_name):
        print(f"  ⚠ SKIP — model not installed in Ollama")
        return {
            "model": model_name,
            "label": label,
            "status": "not_installed",
            "tokens_generated": 0,
            "elapsed_s": 0,
            "tokens_per_sec": 0,
            "response_preview": "",
        }

    print(f"  Sending prompt ({len(PROMPT)} chars)...")
    start = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "prompt": PROMPT, "stream": False},
            timeout=300,
        )
        elapsed = time.time() - start

        if response.status_code != 200:
            print(f"  ✗ HTTP {response.status_code}: {response.text[:200]}")
            return {
                "model": model_name,
                "label": label,
                "status": f"http_error_{response.status_code}",
                "tokens_generated": 0,
                "elapsed_s": elapsed,
                "tokens_per_sec": 0,
                "response_preview": "",
            }

        data = response.json()
        total_tokens = data.get("eval_count", 0)
        prompt_tokens = data.get("prompt_eval_count", 0)
        speed = total_tokens / elapsed if elapsed > 0 else 0
        response_text = data.get("response", "")

        print(f"  Prompt tokens:    {prompt_tokens}")
        print(f"  Generated tokens: {total_tokens}")
        print(f"  Time:             {elapsed:.1f}s")
        print(f"  Speed:            {speed:.1f} t/s")
        print(f"  Response preview: {response_text[:200]!r}")

        return {
            "model": model_name,
            "label": label,
            "status": "ok",
            "prompt_tokens": prompt_tokens,
            "tokens_generated": total_tokens,
            "elapsed_s": round(elapsed, 1),
            "tokens_per_sec": round(speed, 1),
            "response_preview": response_text[:300],
        }

    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"  ✗ TIMEOUT after {elapsed:.0f}s")
        return {
            "model": model_name,
            "label": label,
            "status": "timeout",
            "tokens_generated": 0,
            "elapsed_s": elapsed,
            "tokens_per_sec": 0,
            "response_preview": "",
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ ERROR: {e}")
        return {
            "model": model_name,
            "label": label,
            "status": f"error: {e}",
            "tokens_generated": 0,
            "elapsed_s": elapsed,
            "tokens_per_sec": 0,
            "response_preview": "",
        }


def _installed_models() -> list:
    """Return every model name installed in Ollama (registry order)."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return []
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _resolve_models(args) -> list:
    """Resolve the (model, label) list from CLI args.

    Precedence: --models <list> > --all (every installed model) > TRI-70 default.
    """
    if args.models:
        return [(m, "explicit") for m in args.models]
    if args.all:
        return [(m, "installed") for m in _installed_models()]
    return list(CANDIDATE_MODELS)


def main():
    parser = argparse.ArgumentParser(
        description="Ollama tokens/sec benchmark (candidate-driven; TRI-70 Step 5)"
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explicit model tags to test (overrides the default candidate set)")
    parser.add_argument("--all", action="store_true",
                        help="Test every model currently installed in Ollama")
    parser.add_argument("--output", type=str, default="results/tri70_speed_benchmark.json",
                        help="Where to write the JSON results")
    args = parser.parse_args()

    if not check_ollama_running():
        print("ERROR: Ollama server is not running at", OLLAMA_URL)
        print("Start it with: ollama serve")
        sys.exit(1)

    models = _resolve_models(args)
    print(f"Ollama Speed Benchmark — {len(models)} model(s) "
          f"[{'--models' if args.models else '--all' if args.all else 'TRI-70 candidates'}]")
    print("=" * 60)

    results = []
    for model_name, label in models:
        result = measure_model(model_name, label)
        results.append(result)

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Speed (t/s)':>12} {'Time (s)':>10} {'Status':>12}")
    print("-" * 60)
    for r in results:
        status = r["status"]
        tps = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] > 0 else "N/A"
        elapsed = f"{r['elapsed_s']:.1f}" if r["elapsed_s"] > 0 else "N/A"
        print(f"{r['model']:<25} {tps:>12} {elapsed:>10} {status:>12}")

    # Save results to JSON (default path supersedes task_011_speed_benchmark.json)
    output_path = args.output
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
