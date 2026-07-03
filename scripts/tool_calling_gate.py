"""TRI-70 Step 2 — tool-calling gate for tool-slot candidates.

A tool-slot model must reliably emit well-formed function calls, or it's
disqualified from the tool slot (the analysts bind_tools() and depend on it).
This gate reuses the methodology of tests/test_local_tool_calling.py:
ChatOpenAI against Ollama's OpenAI-compatible endpoint, bind_tools(), inspect
.tool_calls. It runs each candidate through:

  1. basic single-tool call  (N trials — measures well-formed-call rate)
  2. multi-tool selection     (must pick the right tool of two)

PASS = basic-call rate >= --threshold (default 1.0) AND multi-tool correct.
Not-installed candidates are skipped (safe to run mid-pull). Correctness only —
timing is measured separately in Step 5/6, so this is safe to co-run with pulls.

Usage:
    python scripts/tool_calling_gate.py                    # TRI-70 tool candidates
    python scripts/tool_calling_gate.py --models qwen3-coder:30b llama3.3:70b
    python scripts/tool_calling_gate.py --trials 5 --threshold 0.8
"""

import argparse
import json
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434"

# Tool-slot candidates (TRI-70 work order). Quant floor Q4_K_M. Not-installed
# tags are skipped. mistral-small:22b is a known FAIL (kept as a control).
TOOL_CANDIDATES = [
    "qwen3-coder:30b",   # primary, tool-built
    "gpt-oss:20b",       # tool use + structured output
    "gpt-oss:120b",
    "llama3.3:70b",      # carries the Ollama `tools` tag (~97% well-formed)
    "gemma-4:27b",       # include ONLY if it passes (tag TBD at pull)
    "mistral-small:22b",  # control — expected FAIL
    "qwen2.5:14b",        # control — prior tool model
]


def _installed_models() -> list:
    try:
        resp = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _make_llm(model_name: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        base_url=f"{OLLAMA_URL}/v1",
        api_key="ollama",
        temperature=0,
        timeout=180,
    )


def _basic_tool():
    from langchain_core.tools import tool

    @tool
    def get_stock_price(ticker: str, date: str) -> str:
        """Get the stock price for a given ticker on a given date.

        Args:
            ticker: The stock ticker symbol (e.g., 'AAPL')
            date: The date in YYYY-MM-DD format
        """
        return json.dumps({"ticker": ticker, "date": date, "close": 185.50})

    return get_stock_price


def _multi_tools():
    from langchain_core.tools import tool

    @tool
    def get_stock_price(ticker: str) -> str:
        """Get the current stock price for a ticker symbol."""
        return json.dumps({"ticker": ticker, "price": 185.50})

    @tool
    def get_company_news(company: str) -> str:
        """Get recent news articles about a company."""
        return json.dumps({"company": company, "articles": []})

    return get_stock_price, get_company_news


def _wellformed_basic(result) -> bool:
    """True iff result carries a well-formed get_stock_price call w/ ticker arg."""
    tcs = getattr(result, "tool_calls", None) or []
    if not tcs:
        return False
    tc = tcs[0]
    return tc.get("name") == "get_stock_price" and "ticker" in (tc.get("args") or {})


def gate_model(model_name: str, trials: int, threshold: float) -> dict:
    print(f"\n{'='*64}\nGating: {model_name}\n{'='*64}")
    if model_name not in _installed_models():
        print("  SKIP — not installed")
        return {"model": model_name, "status": "not_installed", "gate": "SKIP"}

    llm = _make_llm(model_name)

    # 1) basic single-tool call, N trials
    basic_tool = _basic_tool()
    llm_basic = llm.bind_tools([basic_tool])
    ok = 0
    errors = []
    t0 = time.time()
    for i in range(trials):
        try:
            r = llm_basic.invoke("What is the stock price of AAPL on 2026-02-27?")
            if _wellformed_basic(r):
                ok += 1
            else:
                errors.append((r.content or "")[:120])
        except Exception as e:  # noqa: BLE001
            errors.append(f"EXC: {e}")
    basic_rate = ok / trials if trials else 0.0

    # 2) multi-tool selection (1 trial)
    gsp, gcn = _multi_tools()
    llm_multi = llm.bind_tools([gsp, gcn])
    try:
        rm = llm_multi.invoke("What is AAPL trading at?")
        mtcs = getattr(rm, "tool_calls", None) or []
        multi_ok = bool(mtcs) and mtcs[0].get("name") == "get_stock_price"
    except Exception as e:  # noqa: BLE001
        multi_ok = False
        errors.append(f"MULTI-EXC: {e}")
    elapsed = round(time.time() - t0, 1)

    passed = basic_rate >= threshold and multi_ok
    gate = "PASS" if passed else "FAIL"
    print(f"  basic-call rate: {ok}/{trials} = {basic_rate:.0%}")
    print(f"  multi-tool selection correct: {multi_ok}")
    print(f"  GATE: {gate}  ({elapsed}s)")
    if errors:
        print(f"  sample failures: {errors[:2]}")

    return {
        "model": model_name,
        "status": "ok",
        "basic_rate": round(basic_rate, 3),
        "basic_ok": ok,
        "trials": trials,
        "multi_tool_ok": multi_ok,
        "threshold": threshold,
        "gate": gate,
        "elapsed_s": elapsed,
        "sample_failures": errors[:3],
    }


def main():
    parser = argparse.ArgumentParser(description="TRI-70 tool-calling gate")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explicit model tags (overrides the default candidate set)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Basic-call trials per model (default 3)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Min basic-call rate to PASS (default 1.0)")
    parser.add_argument("--output", type=str, default="results/tri70_tool_gate.json")
    args = parser.parse_args()

    if not _installed_models() and not _ollama_up():
        print(f"ERROR: Ollama not reachable at {OLLAMA_URL}")
        sys.exit(1)

    models = args.models or TOOL_CANDIDATES
    results = [gate_model(m, args.trials, args.threshold) for m in models]

    print(f"\n\n{'='*64}\nTOOL-GATE SUMMARY (threshold basic-rate >= {args.threshold:.0%})\n{'='*64}")
    print(f"{'Model':<22}{'Basic':>10}{'Multi':>8}{'Gate':>8}")
    print("-" * 48)
    for r in results:
        if r["status"] == "not_installed":
            print(f"{r['model']:<22}{'—':>10}{'—':>8}{'SKIP':>8}")
        else:
            print(f"{r['model']:<22}{str(r['basic_ok'])+'/'+str(r['trials']):>10}"
                  f"{('yes' if r['multi_tool_ok'] else 'no'):>8}{r['gate']:>8}")

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
