"""TRI-69 — build Config A into config/hybrid_llm.yaml (idempotent).

Config A (TEST-ONLY, pinned by the TRI-69 pre-registration, never production):
  * tool slot        = ollama / qwen3-coder:30b   (TRI-70 tool-gate winner)
  * reasoning_quick  = ollama / qwen3.5:9b
  * reasoning_deep   = anthropic / claude-sonnet-4-5-20250929
  * temperature      = 0 on ALL THREE slots (new plumbing in create_hybrid_llms)

Enhancement flags are OFF — this matches the TRI-70 bench_deep_* pattern that
shares the same tool+quick slots (measure the raw pipeline, no prompt-prefix
wrappers). DEVELOP-latitude choice, recorded in docs/TASK_TRI-69_REPORT.md.

Sonnet (not Opus) per the locked pre-reg: sonnet-deep scored 8.7 vs opus 7.3
in TRI-70, is ~2x faster, and opus-4-8 rejects the temperature param.

Run:  PYTHONPATH=".:vendor/TradingAgents" python scripts/build_tri69_config.py
"""

from src.hybrid_llm import CONFIGS, HybridLLMConfig, save_config

CONFIG_NAME = "tri69_config_a"


def build():
    save_config(CONFIG_NAME, HybridLLMConfig(
        tool_provider="ollama", tool_model="qwen3-coder:30b",
        reasoning_quick_provider="ollama", reasoning_quick_model="qwen3.5:9b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
        enhance_local=False, enhance_deep=False,
        temperature=0.0,
    ))
    print(f"Built {CONFIG_NAME} (TEST-ONLY): {CONFIGS[CONFIG_NAME].to_dict()}")


if __name__ == "__main__":
    build()
