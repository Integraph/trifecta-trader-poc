"""TRI-69 — build Config A into config/hybrid_llm.yaml (idempotent).

Config A (TEST-ONLY, pinned by the TRI-69 pre-registration, never production),
AS AMENDED under checkpoint 1 (2026-07-06, design window — see
docs/TASK_TRI-69_REPORT.md):
  * tool slot        = ollama / qwen3-coder:30b   (TRI-70 tool-gate winner)
  * reasoning_quick  = ollama / qwen3.5:9b
  * reasoning_deep   = anthropic / claude-sonnet-4-5-20250929, temperature=0
  * LOCAL slots      = model-DEFAULT sampling + pinned seed=69 +
                       max_tokens=16384 runaway guard

Why not temp=0 everywhere (the v3 draft): greedy decoding wedges the local
thinking model on content-dependent inputs — 4 consecutive 3600s timeouts and
a 47-minute HTTP 500, vs a healthy 901s run at model-default sampling under
identical point-in-time conditions. A pinned seed at default sampling is
byte-identical across repeats (verified by direct probe); decision-level
determinism is verified by the battery either way, never assumed.

Enhancement flags OFF (matches TRI-70 bench_deep_* raw-pipeline pattern).
Sonnet (not Opus) deep: 8.7 vs 7.3 in TRI-70, ~2x faster, and opus-4-8
rejects the temperature param.

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
        deep_temperature=0.0,
        local_seed=69,
        local_max_tokens=16384,
    ))
    print(f"Built {CONFIG_NAME} (TEST-ONLY): {CONFIGS[CONFIG_NAME].to_flat_dict()}")


if __name__ == "__main__":
    build()
