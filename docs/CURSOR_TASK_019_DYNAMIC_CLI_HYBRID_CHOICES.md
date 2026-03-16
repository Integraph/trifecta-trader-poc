# Task 019 — Dynamic CLI Hybrid Config Choices

**Type:** Bug Fix (Task 018 Gap)
**Depends on:** Task 018 (LLM Configuration Editor)
**Repo:** trifecta-trader-poc
**Priority:** High

---

## Objective

Fix the CLI `--hybrid` argument in both `run_analysis.py` and `run_batch.py` to dynamically read available config names from `config/hybrid_llm.yaml` instead of using a hardcoded list.

This is a gap from Task 018. Task 018 externalized hybrid LLM configs from Python code to YAML and built a full CRUD admin UI so admins can create/edit/delete configs without touching code. However, the CLI entry points — the very commands the daemon and manual operators use — still have hardcoded `choices=` lists that reject any config name not in the original 13. This breaks the core Task 018 use case: adding a new LLM config through the admin UI and then running an analysis with it.

---

## Background

Task 018 correctly:
- Externalized configs to `config/hybrid_llm.yaml`
- Added `load_configs()`, `save_config()`, `delete_config()`, `reload_configs()` to `src/hybrid_llm.py`
- Built CRUD API endpoints and admin UI for managing configs
- Made `CONFIGS = load_configs()` at module level for backward compatibility

But two CLI entry points still have static choices lists:

1. `src/run_analysis.py` lines 698-706 — 13 hardcoded config names
2. `src/run_batch.py` lines 418-426 — same 13 hardcoded config names

When a user adds a new config (e.g., `hybrid_gpt4o_test`) through the admin UI, then runs:
```bash
python -m src.run_analysis --ticker AAPL --hybrid hybrid_gpt4o_test
```
argparse rejects it with: `error: argument --hybrid: invalid choice: 'hybrid_gpt4o_test'`

---

## Deliverables

### 1. Fix `src/run_analysis.py` — Dynamic `--hybrid` choices

**Current code (lines 698-706):**
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

**Required change:**
Replace the static `choices=` list with a dynamic read from `CONFIGS`:

```python
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser.add_argument("--hybrid", type=str, default=None,
                    choices=list(_hybrid_configs.keys()),
                    help="Use hybrid LLM routing config")
```

**Note:** `CONFIGS` is already imported at line 273 inside `run_analysis()`, but the parser is built in `main()` (line 688). The import for the choices list must be in `main()` scope, before `parser.add_argument`. Use an aliased import (`_hybrid_configs`) to avoid shadowing.

### 2. Fix `src/run_batch.py` — Same dynamic pattern

**Current code (lines 418-426):**
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

**Required change:**
Same pattern — replace static choices with dynamic read:

```python
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser.add_argument("--hybrid", type=str, default="hybrid_haiku_tools",
                    choices=list(_hybrid_configs.keys()),
                    help="Hybrid LLM config (default: hybrid_haiku_tools)")
```

**Note:** The `default="hybrid_haiku_tools"` is fine — it's a sensible default that exists in the YAML. If someone deletes it from YAML, argparse will still accept it but `run_analysis` will fail at config lookup time with a clear KeyError, which is the correct behavior.

### 3. Tests

Add or extend tests to verify:

- **Test dynamic choices include all YAML configs:** After `load_configs()`, the argparse choices list should contain every key from the YAML file.
- **Test new config is accepted:** Create a temp YAML with an extra config (e.g., `test_new_model`), call `reload_configs()`, and verify the CLI parser accepts the new name.
- **Test deleted config is rejected:** Remove a config from YAML, call `reload_configs()`, and verify the CLI parser rejects the old name.
- **No regressions:** All existing tests must continue to pass (497 passing, 40 pre-existing Google GenAI failures unchanged).

---

## Exit Criteria

1. `python -m src.run_analysis --hybrid <new_yaml_config>` accepts any config name present in `config/hybrid_llm.yaml`
2. `python -m src.run_batch --hybrid <new_yaml_config>` same behavior
3. `python -m src.run_analysis --help` shows the current YAML config names in the `--hybrid` help text
4. Adding a new config via admin API → CLI immediately accepts it (after module reimport / reload_configs)
5. No hardcoded config name lists remain in `run_analysis.py` or `run_batch.py`
6. New tests covering dynamic choices behavior
7. All existing tests pass (no regressions)

---

## Files to Modify

| File | Change |
|------|--------|
| `src/run_analysis.py` | Replace hardcoded `choices=` with dynamic `list(CONFIGS.keys())` |
| `src/run_batch.py` | Same change |

## Files to Create/Extend

| File | Change |
|------|--------|
| `tests/test_task019_dynamic_cli.py` | New test file for dynamic CLI choices |

---

## Scope Boundaries

- **In scope:** CLI argparse choices in `run_analysis.py` and `run_batch.py` only
- **Out of scope:** Admin API endpoints (already dynamic), admin UI (already dynamic), daemon scheduler (reads config name from YAML, already dynamic)
- **Do not modify:** `src/hybrid_llm.py`, `config/hybrid_llm.yaml`, admin API, admin UI — these are all correct
