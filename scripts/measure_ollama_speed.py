"""Measure tokens/second for each Ollama model on a financial reasoning prompt.

Usage:
    python scripts/measure_ollama_speed.py

Tests all four models (baseline Qwen 2.5:14b + three Qwen 3.5 candidates)
on a representative financial analysis prompt that matches the type of reasoning
our pipeline agents perform.
"""

import time
import json
import sys
import requests

OLLAMA_URL = "http://localhost:11434"

MODELS = [
    ("qwen2.5:14b",    "Baseline (current production)"),
    ("qwen3.5:9b",     "Qwen 3.5  9B dense"),
    ("qwen3.5:27b",    "Qwen 3.5 27B dense"),
    ("qwen3.5:35b-a3b","Qwen 3.5 35B MoE (3B active)"),
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


def main():
    if not check_ollama_running():
        print("ERROR: Ollama server is not running at", OLLAMA_URL)
        print("Start it with: ollama serve")
        sys.exit(1)

    print("Ollama Speed Benchmark — Qwen 2.5 vs Qwen 3.5")
    print("=" * 60)

    results = []
    for model_name, label in MODELS:
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

    # Save results to JSON
    output_path = "results/task_011_speed_benchmark.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
