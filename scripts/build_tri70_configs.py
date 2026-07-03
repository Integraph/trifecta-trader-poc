"""TRI-70 — build the benchmark configs into config/hybrid_llm.yaml.

Creates (idempotent — safe to re-run):
  * benchmark_local_b       — the prize: genuinely all-local (no cloud in any slot)
  * benchmark_opus_a        — cloud REFERENCE only (test-only): Haiku tools, local quick,
                              claude-opus-4-8 deep. NEVER shipped.
  * bench_deep_<candidate>  — deep-slot head-to-head variants; tool+quick fixed
                              (tool=qwen3-coder:30b [passed the gate], quick=qwen3.5:9b),
                              deep = each local reasoner candidate.

Tool slot is the gate-passing local qwen3-coder:30b so the deep-slot comparison
is isolated and the whole config stays local. Enhancements are OFF on the local
head-to-head (measure the raw model); benchmark_opus_a mirrors hybrid_haiku_tools'
enhancement settings so it's comparable to the existing cloud reference.

Run:  PYTHONPATH=".:vendor/TradingAgents" python scripts/build_tri70_configs.py
"""

from src.hybrid_llm import HybridLLMConfig, save_config, CONFIGS

LOCAL_TOOL = "qwen3-coder:30b"   # passed the TRI-70 tool gate (3/3 basic, multi ok)
LOCAL_QUICK = "qwen3.5:9b"        # fast local quick slot (already present)

# Deep-slot candidates (local reasoners) for the head-to-head screen.
DEEP_CANDIDATES = {
    "bench_deep_qwen36_35b": "qwen3.6:35b",     # MoE ~3B active — primary hope
    "bench_deep_qwen36_27b": "qwen3.6:27b",     # dense current-gen
    "bench_deep_r1_8b":      "deepseek-r1:8b",  # R1-0528 refresh dark horse
    "bench_deep_r1_14b":     "deepseek-r1:14b",  # 2025 distill (qwen2.5 base)
    "bench_deep_r1_32b":     "deepseek-r1:32b",  # 2025 distill (qwen2.5 base)
    "bench_deep_r1_70b":     "deepseek-r1:70b",  # 2025 distill (llama3.3 base)
    "bench_deep_gptoss_120b": "gpt-oss:120b",   # strong reasoning/structured
}


def all_local(deep_model: str) -> HybridLLMConfig:
    return HybridLLMConfig(
        tool_provider="ollama", tool_model=LOCAL_TOOL,
        reasoning_quick_provider="ollama", reasoning_quick_model=LOCAL_QUICK,
        reasoning_deep_provider="ollama", reasoning_deep_model=deep_model,
        enhance_local=False, enhance_deep=False,
    )


def build():
    made = []

    # benchmark_local_b — default deep = qwen3.6:27b (current-gen local; the
    # FINAL deep model is the head-to-head winner, re-pinned in TRI-73).
    save_config("benchmark_local_b", all_local("qwen3.6:27b"))
    made.append("benchmark_local_b (all-local: tool=%s quick=%s deep=qwen3.6:27b)"
                % (LOCAL_TOOL, LOCAL_QUICK))

    # benchmark_opus_a — cloud reference ONLY (test-only). Mirrors
    # hybrid_haiku_tools enhancement settings, deep swapped to claude-opus-4-8.
    save_config("benchmark_opus_a", HybridLLMConfig(
        tool_provider="anthropic", tool_model="claude-haiku-4-5-20251001",
        reasoning_quick_provider="ollama", reasoning_quick_model=LOCAL_QUICK,
        reasoning_deep_provider="anthropic", reasoning_deep_model="claude-opus-4-8",
        enhance_local=True, enhance_style="financial_analysis",
        enhance_deep=True, enhance_deep_style="execution_params_only",
    ))
    made.append("benchmark_opus_a (TEST-ONLY cloud ref: Haiku tools / local quick / opus-4-8 deep)")

    # Per-candidate deep-slot variants
    for name, deep in DEEP_CANDIDATES.items():
        save_config(name, all_local(deep))
        made.append(f"{name} (deep={deep})")

    print("Built %d TRI-70 configs:" % len(made))
    for m in made:
        print("  -", m)
    print("\nTotal configs now in CONFIGS: %d" % len(CONFIGS))


if __name__ == "__main__":
    build()
