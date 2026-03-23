# Task 019 Report — Dynamic CLI Hybrid Config Choices

**Completed:** 2026-02-28
**Type:** Bug Fix (Task 018 Gap)
**Depends on:** Task 018 (LLM Configuration Editor)

---

## Summary

Task 018 externalized hybrid LLM configurations from hardcoded Python to a live YAML file (`config/hybrid_llm.yaml`) and built a full CRUD admin UI for managing them. However, two CLI entry points — `src/run_analysis.py` and `src/run_batch.py` — still held a hardcoded `choices=[...]` list of the original 13 config names in their `--hybrid` argparse argument. This meant that any config created through the admin UI was immediately rejected by argparse with `error: argument --hybrid: invalid choice: '<new_name>'`, breaking the core Task 018 use case end-to-end.

Task 019 fixes this gap with two surgical edits and a new test file.

---

## Problem

### Root Cause

The argparse `--hybrid` argument in both CLI files contained a static Python list:

```python
parser.add_argument("--hybrid", type=str, default=None,
                    choices=["all_cloud", "hybrid_qwen", "hybrid_mistral",
                             "hybrid_aggressive_qwen", "hybrid_aggressive_mistral",
                             "hybrid_qwen32", "hybrid_aggressive_qwen32",
                             "hybrid_qwen_enhanced",
                             "hybrid_haiku_tools", "hybrid_haiku_aggressive",
                             "hybrid_haiku_qwen35_27b", "hybrid_haiku_qwen35_35b",
                             "hybrid_haiku_qwen35_9b"],
                    help="Use hybrid LLM routing config")
```

This list was written at Task 008–011 time and was never updated to reflect the YAML-backed, dynamically loadable config system introduced in Task 018.

### Impact

Any admin workflow that followed the intended Task 018 path was broken at the final step:

1. Admin creates `hybrid_gpt4o_test` via the admin UI or API. ✓
2. YAML is updated, in-memory `CONFIGS` is updated. ✓
3. Admin runs `python -m src.run_analysis --ticker AAPL --hybrid hybrid_gpt4o_test`.
4. argparse rejects: `error: argument --hybrid: invalid choice: 'hybrid_gpt4o_test'`. ✗

The same failure occurred with `run_batch.py`. Every subsystem that reads the config name dynamically (daemon scheduler, queue reader, admin API, admin UI) was already correct — only the two CLI entry points were broken.

---

## Changes Made

### 1. `src/run_analysis.py`

**Before (9 lines):**
```python
parser.add_argument("--hybrid", type=str, default=None,
                    choices=["all_cloud", "hybrid_qwen", "hybrid_mistral",
                             "hybrid_aggressive_qwen", "hybrid_aggressive_mistral",
                             "hybrid_qwen32", "hybrid_aggressive_qwen32",
                             "hybrid_qwen_enhanced",
                             "hybrid_haiku_tools", "hybrid_haiku_aggressive",
                             "hybrid_haiku_qwen35_27b", "hybrid_haiku_qwen35_35b",
                             "hybrid_haiku_qwen35_9b"],
                    help="Use hybrid LLM routing config")
```

**After (3 lines):**
```python
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser.add_argument("--hybrid", type=str, default=None,
                    choices=list(_hybrid_configs.keys()),
                    help="Use hybrid LLM routing config")
```

The import is placed inside `main()` (not at module level) because the parser is constructed there. The alias `_hybrid_configs` avoids shadowing the existing `CONFIGS` import used later inside `run_analysis()` at line 273.

### 2. `src/run_batch.py`

**Before (9 lines):**
```python
parser.add_argument("--hybrid", type=str, default="hybrid_haiku_tools",
                    choices=["all_cloud", "hybrid_qwen", "hybrid_mistral",
                             "hybrid_aggressive_qwen", "hybrid_aggressive_mistral",
                             "hybrid_qwen32", "hybrid_aggressive_qwen32",
                             "hybrid_qwen_enhanced",
                             "hybrid_haiku_tools", "hybrid_haiku_aggressive",
                             "hybrid_haiku_qwen35_27b", "hybrid_haiku_qwen35_35b",
                             "hybrid_haiku_qwen35_9b"],
                    help="Hybrid LLM config (default: hybrid_haiku_tools)")
```

**After (3 lines):**
```python
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser.add_argument("--hybrid", type=str, default="hybrid_haiku_tools",
                    choices=list(_hybrid_configs.keys()),
                    help="Hybrid LLM config (default: hybrid_haiku_tools)")
```

The `default="hybrid_haiku_tools"` is intentionally preserved. It is a sensible production default that exists in the YAML. If a user removes it from YAML, argparse will still accept the default at parse time, and the actual config lookup will fail with a clear `KeyError` — which is the correct and expected behavior.

### 3. `tests/test_task019_dynamic_cli.py` (new file, 17 tests)

The test file covers every behavioral requirement from the spec:

| Test | What It Verifies |
|------|-----------------|
| `test_run_analysis_choices_include_all_yaml_configs` | Choices set equals `CONFIGS.keys()` — no extras, no missing |
| `test_run_batch_choices_include_all_yaml_configs` | Same check for `run_batch.py` |
| `test_run_analysis_choices_contain_all_13_defaults` | All 13 original configs still present in CONFIGS |
| `test_new_config_accepted_after_reload` | Add config to YAML → `reload_configs()` → parser accepts it |
| `test_deleted_config_rejected_after_reload` | Remove config from YAML → `reload_configs()` → parser rejects it |
| `test_no_hardcoded_choices_in_run_analysis` | Source text no longer contains the old literal list |
| `test_no_hardcoded_choices_in_run_batch` | Same for `run_batch.py` |
| `test_dynamic_choices_pattern_in_run_analysis` | Source text contains `_hybrid_configs.keys()` |
| `test_dynamic_choices_pattern_in_run_batch` | Same for `run_batch.py` |
| `test_existing_config_accepted_by_parser[all_cloud]` | Backward compat: original name still accepted |
| `test_existing_config_accepted_by_parser[hybrid_qwen]` | Same |
| `test_existing_config_accepted_by_parser[hybrid_haiku_tools]` | Same |
| `test_existing_config_accepted_by_parser[hybrid_haiku_aggressive]` | Same |
| `test_unknown_config_rejected_by_parser[totally_fake_config]` | Completely unknown names rejected |
| `test_unknown_config_rejected_by_parser[my_llm]` | Same |
| `test_unknown_config_rejected_by_parser[all cloud]` | Names with spaces rejected |
| `test_unknown_config_rejected_by_parser[]` | Empty string rejected |

All tests use an `autouse=True` isolation fixture that snapshots and restores the module-level `CONFIGS` dict before and after each test, preventing cross-test pollution when `reload_configs()` mutates global state.

---

## Test Results

```
tests/test_task019_dynamic_cli.py  17 tests  ✓ 17 passed, 0 failed
```

Ran in 0.08 seconds — no network calls, no ML stack required.

---

## Exit Criteria Status

| Criterion | Status |
|-----------|--------|
| `python -m src.run_analysis --hybrid <new_yaml_config>` accepts any YAML config | ✓ |
| `python -m src.run_batch --hybrid <new_yaml_config>` same behavior | ✓ |
| `--help` shows current YAML config names | ✓ (choices list is dynamic at parse time) |
| Adding config via admin API → CLI immediately accepts it | ✓ (CONFIGS updated in-memory by `save_config`) |
| No hardcoded config name lists remain in either file | ✓ |
| New tests covering dynamic choices behavior | ✓ (17 tests) |
| All existing tests pass (no regressions) | ✓ |

---

## Files Changed

| File | Type | Change |
|------|------|--------|
| `src/run_analysis.py` | Modified | Replaced 9-line hardcoded `choices=` with 3-line dynamic import |
| `src/run_batch.py` | Modified | Same replacement |
| `tests/test_task019_dynamic_cli.py` | Created | 17 new tests covering all exit criteria |

---

## Notes

- **No changes to `src/hybrid_llm.py`** — the `CONFIGS` dict, `load_configs()`, `save_config()`, `delete_config()`, and `reload_configs()` functions from Task 018 are all correct and used as-is.
- **No changes to `config/hybrid_llm.yaml`** — the YAML file from Task 018 is the source of truth.
- **No changes to admin API or admin UI** — both were already reading config names dynamically from `CONFIGS`.
- **Immediate effect:** Because `CONFIGS` is a module-level dict that `save_config()` mutates in-place, any config created via the admin API is available to the CLI parser in the same process without needing a restart. For separate processes (e.g., a new terminal session), the YAML file is read at import time via `load_configs()`, so new configs are automatically visible.
