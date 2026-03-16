# Task 018 Pre-Implementation Questions

---

## Q1: Config Count Discrepancy (Python → YAML migration)

The spec says "all 12 existing configs migrated." The current `src/hybrid_llm.py`
actually contains **13 configs**:

1. `all_cloud`
2. `hybrid_qwen`
3. `hybrid_mistral`
4. `hybrid_aggressive_qwen`
5. `hybrid_aggressive_mistral`
6. `hybrid_qwen32`
7. `hybrid_aggressive_qwen32`
8. `hybrid_qwen_enhanced`
9. `hybrid_haiku_tools`
10. `hybrid_haiku_qwen35_27b`
11. `hybrid_haiku_qwen35_35b`
12. `hybrid_haiku_qwen35_9b`
13. `hybrid_haiku_aggressive`

Should I migrate **all 13** to `config/hybrid_llm.yaml`, or is there a specific
set of 12 to include (i.e., exclude one)?

My assumption: migrate all 13. Please confirm or specify which to exclude.

---

## Q2: Providers List Source

The spec says `GET /config/hybrid-configs` should return a `providers` array
"from the factory." There is no provider registry/factory in the codebase —
providers are referenced as plain strings in `HybridLLMConfig` instances.

The spec example shows: `["anthropic", "ollama", "openai", "google", "xai", "openrouter"]`

Two options:

**Option A — Hardcode the canonical list** (matches spec example exactly):
```python
KNOWN_PROVIDERS = ["anthropic", "ollama", "openai", "google", "xai", "openrouter"]
```
Reliable; doesn't drift if a config uses a new provider string.

**Option B — Derive dynamically from `CONFIGS` dict** (only shows providers
actually in use):
```python
providers = sorted({p for cfg in CONFIGS.values()
                    for p in [cfg.tool_provider, cfg.reasoning_quick_provider, cfg.reasoning_deep_provider]})
```
May miss providers that are valid but not yet in any config.

**My recommendation:** Option A (hardcoded canonical list) since validation logic
also needs a fixed set, and Option B would silently accept invalid providers for
new configs if they weren't yet in any existing config.

---

## Summary

| # | Question | My Assumption |
|---|----------|---------------|
| Q1 | Migrate all 13 or specific 12 configs? | All 13 |
| Q2 | Providers from factory or hardcoded list? | Option A — hardcoded canonical list |

No other blocking ambiguities found. The spec is comprehensive — enhance_styles are
confirmed as `["financial_analysis", "structured", "few_shot", "execution_params_only"]`
from `src/enhanced_llm.py`'s `ENHANCEMENT_STYLES` dict (verified). Route ordering
for `/ab` vs `/{task_id}` in `test_run.py` is an implementation detail (register
`/ab/*` before `/{task_id}`).
