# Live Trading Readiness — Claude Code Development Prompt

> ⛔ **SUPERSEDED (2026-06-30) — DO NOT EXECUTE.** This live-trading readiness sprint is out of scope under the engine-first, **paper-only MVP**: `--execute` is forbidden and `paper=True` stays hardcoded (`src/execution/executor.py:49`). Live trading is **deferred (TRI-31)** behind the **TRI-69** "prove-it-before-real-money" edge gate. Current runnable order: **TRI-66 → TRI-70 (re-benchmark) → TRI-69**. Kept for historical reference only — do not hand this to a development agent.

**Date:** June 27, 2026
**Prepared by:** Project Manager (Cowork/Claude)
**For:** Claude Code (development agent)
**Sprint:** Live Trading Readiness (5 features)
**Repo:** `trifecta-trader-poc`
**Branch:** Create `live-trading-readiness` from `main`

---

## YOUR ROLE

You are the **implementation agent** for the Trifecta Stock Trader Engine. You implement features end-to-end: code, tests, and documentation. You do NOT need approval between steps — execute all 6 steps sequentially, then report.

---

## CONTEXT

The Stock Trader Engine is an AI-driven stock trading system built on a multi-agent LLM analysis pipeline with a vendor submodule (`vendor/TradingAgents`). The system uses a HybridTradingGraph for 3-tier LLM routing (tool/quick/deep), executes via Alpaca, and publishes to Supabase. It has a FastAPI admin API and React admin-ui.

As of v1.0.2, 20 tasks are complete (52 source files, 43 test files). The system works end-to-end in **paper trading mode only** — `paper=True` is hardcoded in two places. The largest remaining gap is live trading capability with appropriate safety infrastructure.

Before starting, read these two files in the repo root:
- **`PROJECT_BRIEF.md`** — full project context, development history, architecture
- **`ECOSYSTEM_CONTEXT.md`** — how this engine fits into the 5-component Trifecta ecosystem

---

## HARD CONSTRAINTS

1. **Zero vendor modifications.** `vendor/TradingAgents` is a git submodule. NEVER modify any file inside it. Verify with `cd vendor/TradingAgents && git status`.
2. **Paper mode is the default.** `paper=True` must remain the default. Live trading requires BOTH changing a config value AND passing `--confirm-live` on the CLI. No single-point-of-failure path to live.
3. **All tests use mocked Alpaca.** No real broker calls in unit tests. Ever. Existing tests use `patch("src.execution.executor.TradingClient")`.
4. **Audit-first logging.** Every order intent is logged BEFORE the broker call. This pattern already exists in `executor.py` — preserve it.
5. **Atomic write pattern.** All persistent state uses write-to-`.tmp` → validate → `os.replace()`. This is the project-wide convention.
6. **Backward compatibility.** All existing tests must continue to pass. All existing CLI flags must work unchanged. Existing `config/automation.yaml` without new fields must work with defaults.
7. **Existing position safety gates stay.** `PositionManager` (lines 15–19 of `src/execution/position_manager.py`) has `MAX_POSITION_PCT=15.0`, `MIN_QUALITY_SCORE=8.0`, `MAX_PORTFOLIO_RISK_PCT=2.0`, `MIN_RISK_REWARD=1.5`. These remain enforced in both paper and live mode.
8. **Config via YAML.** All new configurable values go in `config/automation.yaml` (daemon/queue settings) or a new `config/execution.yaml` (execution layer settings). Secrets stay in `.env`.
9. **Admin API extensibility.** New features that have runtime state should expose it via the existing FastAPI admin API (`src/admin/app.py`, port 8420) so the React admin-ui and the Stock Trader UI (tt-curser) can eventually surface it.
10. **Follow existing code patterns.** This codebase uses dataclasses (not Pydantic for execution layer), YAML config, `logging.getLogger(__name__)`, and pytest with `tmp_path` fixtures.

---

## STEP 0: Housekeeping

### 0.1 Commit ecosystem context document

The file `ECOSYSTEM_CONTEXT.md` was placed in the repo root but may not be committed yet. If it's untracked or modified:

```bash
git add ECOSYSTEM_CONTEXT.md PROJECT_BRIEF.md
git commit -m "docs: add ecosystem context and update project brief"
```

### 0.2 Push existing commits

There are approximately 13 commits ahead of origin. Push them:

```bash
git push origin main
```

If push fails due to auth or remote issues, log the error and continue — do not block on this.

### 0.3 Create the working branch

```bash
git checkout -b live-trading-readiness
```

### 0.4 Update Linear

Update issue TRI-32 (Push unpushed commits) to Done if the push succeeded.

**Throughout the sprint**, update these Linear issues as you complete each step:
- **TRI-31** (Live trading capability) → In Progress when you start Step 1, Done after Step 2 completes
- **TRI-33** (Queue dir config) → Done after Step 3 completes
- **TRI-34** (Fix Ollama tests) → Done after Step 4 completes

---

## STEP 1: Configurable Paper/Live Trading Mode

**Problem:** `paper=True` is hardcoded in two places, making live trading impossible:
- `src/execution/executor.py` line 45–50: `self._client = TradingClient(api_key, secret_key, paper=True)`
- `src/run_analysis.py` lines 70–75: `_make_trading_client()` also hardcodes `paper=True`

**Goal:** Make the trading mode configurable with a two-key safety lock (config + CLI flag).

### 1.1 Create execution config file (`config/execution.yaml`)

```yaml
# Trading mode: "paper" (default) or "live"
# CAUTION: Switching to "live" also requires passing --confirm-live on the CLI.
# Both must agree — config alone is NOT sufficient.
mode: paper

# Daily loss limit (USD). Orders are rejected once cumulative daily realized
# losses reach this threshold. Resets at midnight UTC.
daily_loss_limit_usd: 500.0

# Path to persist daily loss state across restarts
daily_loss_persist_path: results/daily_loss_state.json

# Kill switch file. If this file exists, ALL new orders are blocked.
kill_switch_path: HALT_TRADING

# Order recovery on startup (live mode only)
recover_orders_on_startup: true
```

### 1.2 Update `TradeExecutor.__init__` (`src/execution/executor.py`)

Currently (lines 27–53):
```python
class TradeExecutor:
    def __init__(self, api_key: str, secret_key: str, audit_dir: str = "audit"):
        ...
        self._client = TradingClient(api_key, secret_key, paper=True)  # ← HARDCODED
```

Change to:
```python
class TradeExecutor:
    def __init__(self, api_key: str, secret_key: str, audit_dir: str = "audit", paper: bool = True):
        ...
        self._client = TradingClient(api_key, secret_key, paper=paper)
```

**Default remains `paper=True`.** The parameter is only set to `False` when both conditions are met (see Step 1.4).

Update the module docstring (lines 1–6) to remove the "ONLY connects to Alpaca paper trading" language. Replace with:
```python
"""Trade execution via Alpaca API.

SAFETY: Defaults to paper trading. Live trading requires BOTH:
  1. config/execution.yaml: mode: live
  2. CLI flag: --confirm-live
"""
```

### 1.3 Update `_make_trading_client` (`src/run_analysis.py`)

Currently (lines 70–75):
```python
def _make_trading_client():
    ...
    return TradingClient(api_key, secret_key, paper=True)
```

This function is used for `PositionManager` (read-only account queries). It should ALWAYS use paper mode for position queries — do NOT change this function. The live/paper distinction only applies to order submission in `TradeExecutor`.

### 1.4 Add CLI flag and mode resolution (`src/run_analysis.py`)

Add to argparse (near line 704–706):
```python
parser.add_argument("--confirm-live", action="store_true",
                    help="Required alongside config mode:live to enable live trading. "
                         "Paper mode is used if this flag is absent.")
```

Add a mode resolution function:
```python
def _resolve_trading_mode(args, execution_config: dict) -> bool:
    """Determine if paper mode should be used.
    
    Returns True for paper mode, False for live mode.
    Live requires BOTH config mode=live AND --confirm-live flag.
    """
    config_mode = execution_config.get("mode", "paper")
    if config_mode == "live" and args.confirm_live:
        logger.warning("🔴 LIVE TRADING MODE ACTIVE — real money at risk")
        return False  # paper=False means live
    if config_mode == "live" and not args.confirm_live:
        logger.info("Config says 'live' but --confirm-live not passed. Using paper mode.")
    return True  # paper=True (default, safe)
```

### 1.5 Load execution config

Add a config loader (either in `src/run_analysis.py` or a new `src/config/execution_config.py`):
```python
def load_execution_config(path: str = "config/execution.yaml") -> dict:
    """Load execution config with defaults for missing fields."""
    defaults = {
        "mode": "paper",
        "daily_loss_limit_usd": 500.0,
        "daily_loss_persist_path": "results/daily_loss_state.json",
        "kill_switch_path": "HALT_TRADING",
        "recover_orders_on_startup": True,
    }
    if os.path.exists(path):
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    return defaults
```

### 1.6 Wire into execution flow (`src/run_analysis.py`)

In `_run_execution_flow()` (around line 510–596), where `TradeExecutor` is instantiated:

**Important:** The current code instantiates `TradeExecutor` without explicit credentials — it reads them from environment variables internally (lines 552 and 569 call `TradeExecutor(audit_dir=audit_dir)`). You need to either:
- Add `paper` as a parameter to `_run_execution_flow()` and pass it through, OR
- Load the execution config inside the function

Recommended approach — add `paper` param to the function signature:
```python
def _run_execution_flow(result, config, args, tracker, analysis_id, paper=True):
    ...
    executor = TradeExecutor(audit_dir=audit_dir, paper=paper)
```

Then in `main()` (around line 754), resolve the mode before calling the execution flow:
```python
execution_config = load_execution_config()
paper = _resolve_trading_mode(args, execution_config)
# Pass paper to the execution flow
_run_execution_flow(result, config, args, tracker, analysis_id, paper=paper)
```

### 1.6b Wire into the daemon path (separate execution path)

**Critical:** The daemon's queue processing does NOT go through `_run_execution_flow()`. It uses a separate path:
- `src/automation/daemon.py` calls `create_analyze_fn()` (imported from `src/automation/queue_reader.py` or a factory module)
- This factory creates a function that runs the analysis pipeline independently

You must trace this path and ensure the `paper` mode is resolved and passed through the daemon's analysis function as well. The daemon should load `config/execution.yaml` during startup (in `PipelineDaemon.__init__` or `start()`) and pass the resolved `paper` value to whatever function creates the `TradeExecutor` in the queue processing path.

**The daemon should NEVER use `--confirm-live` from CLI.** For daemon mode, the `config/execution.yaml` setting `mode: live` is sufficient (the CLI safety lock is designed for interactive use only). Add a separate daemon-level confirmation: a log WARNING at startup if the daemon is running in live mode.

### 1.7 Update audit entry

In `executor.py`, `_create_audit_entry()` (lines 164–182), add a field:
```python
"trading_mode": "paper" if self._paper else "live",
```

Store `self._paper = paper` in `__init__` for this purpose.

### 1.8 Tests

Add to `tests/test_executor.py` (at least 4 new tests):
- `test_paper_mode_default` — no `paper` arg → `TradingClient` called with `paper=True`
- `test_live_mode_explicit` — `paper=False` → `TradingClient` called with `paper=False`
- `test_audit_entry_includes_trading_mode` — verify `"trading_mode"` field in audit JSON
- `test_existing_paper_only_test_still_passes` — existing `test_paper_only_hardcoded` should still pass (update it to expect `paper=True` default)

Add new file `tests/test_trading_mode.py` (at least 4 tests):
- `test_resolve_paper_default` — no config, no flag → paper
- `test_resolve_config_live_no_flag` — config=live, no --confirm-live → paper (safe fallback)
- `test_resolve_config_live_with_flag` — config=live + --confirm-live → live
- `test_resolve_config_paper_with_flag` — config=paper + --confirm-live → paper (config wins)

---

## STEP 2: Safety Infrastructure

**Problem:** The Stock Trader has position-level safety (PositionManager gates) but no session-level or system-level safety controls. The Crypto Trader has an 8-layer safety gate stack including a kill switch and daily loss limit. The Stock Trader needs equivalent protection before live trading is safe.

### 2.1 Kill Switch (`src/execution/kill_switch.py` — new file)

```python
class KillSwitch:
    """File-based kill switch. If the kill switch file exists, all new orders are blocked.
    
    Usage:
        ks = KillSwitch(path="HALT_TRADING")
        if ks.is_engaged():
            # reject order
        
        # To engage: touch HALT_TRADING
        # To disengage: rm HALT_TRADING
    """
    def __init__(self, path: str = "HALT_TRADING"):
        self._path = Path(path)
    
    def is_engaged(self) -> bool:
        """Check if the kill switch file exists."""
        return self._path.exists()
    
    def engage(self, reason: str = "") -> None:
        """Create the kill switch file with an optional reason."""
        self._path.write_text(reason or f"Engaged at {datetime.now(UTC).isoformat()}")
    
    def disengage(self) -> None:
        """Remove the kill switch file."""
        self._path.unlink(missing_ok=True)
    
    def get_status(self) -> dict:
        """Return status for admin API / UI consumption."""
        engaged = self.is_engaged()
        return {
            "engaged": engaged,
            "reason": self._path.read_text().strip() if engaged else None,
            "path": str(self._path),
        }
```

### 2.2 Daily Loss Tracker (`src/execution/daily_loss_tracker.py` — new file)

```python
class DailyLossTracker:
    """Persists daily realized loss accumulation to a JSON file.
    
    File format:
    {
        "date": "2026-06-27",
        "accumulated_loss_usd": 123.45,
        "updated_at": "2026-06-27T14:30:00Z"
    }
    """
    def __init__(self, path: str = "results/daily_loss_state.json",
                 daily_limit_usd: float = 500.0):
        self._path = Path(path)
        self._daily_limit_usd = daily_limit_usd
    
    def get_daily_loss(self) -> float:
        """Load from file. If date doesn't match today (UTC), return 0.0."""
        ...
    
    def record_loss(self, loss_usd: float) -> float:
        """Add loss amount, persist atomically, return new total.
        Only positive values accepted."""
        ...
    
    def is_limit_exceeded(self) -> bool:
        """Check if daily loss has reached the configured limit."""
        return self.get_daily_loss() >= self._daily_limit_usd
    
    def reset_if_new_day(self) -> bool:
        """Check date (UTC), reset if new day. Return True if reset occurred."""
        ...
    
    def get_status(self) -> dict:
        """Return status for admin API / UI consumption."""
        current = self.get_daily_loss()
        return {
            "date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "accumulated_loss_usd": current,
            "limit_usd": self._daily_limit_usd,
            "remaining_usd": max(0, self._daily_limit_usd - current),
            "limit_exceeded": current >= self._daily_limit_usd,
        }
```

Follow the atomic write pattern: write to `.tmp`, then `os.replace()`. Create parent directories if they don't exist. Handle corrupt JSON gracefully (log WARNING, reset to 0.0).

### 2.3 Integrate safety gates into `TradeExecutor.execute()`

In `src/execution/executor.py`, modify `execute()` (lines 60–109) to add two pre-flight checks BEFORE the existing order submission logic:

```python
def execute(self, order_calc, trade_params, kill_switch=None, daily_loss_tracker=None):
    # NEW: Kill switch check (before everything else)
    if kill_switch and kill_switch.is_engaged():
        self._save_audit(self._create_audit_entry(order_calc, trade_params, "KILLED"))
        return {"action": "KILLED", "reason": "Kill switch engaged"}
    
    # NEW: Daily loss limit check
    if daily_loss_tracker and daily_loss_tracker.is_limit_exceeded():
        self._save_audit(self._create_audit_entry(order_calc, trade_params, "DAILY_LIMIT"))
        return {"action": "DAILY_LIMIT", "reason": "Daily loss limit exceeded"}
    
    # EXISTING: Approval check, order submission, etc.
    ...
    
    # AFTER execution: if order resulted in a loss, track it
    # (This requires the trade result to include P&L — may need to be tracked
    # after position close rather than order open. For now, stub the hook.)
```

The `kill_switch` and `daily_loss_tracker` parameters are **optional** (`None` by default) to preserve backward compatibility with existing callers and tests.

### 2.4 Admin API endpoints

Add a new router file `src/admin/safety.py`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/safety", tags=["safety"])

@router.get("/status")
def get_safety_status(kill_switch, daily_loss_tracker):
    """Return current safety system status."""
    return {
        "kill_switch": kill_switch.get_status() if kill_switch else {"engaged": False},
        "daily_loss": daily_loss_tracker.get_status() if daily_loss_tracker else None,
    }

@router.post("/kill-switch/engage")
def engage_kill_switch(kill_switch, reason: str = ""):
    """Engage the kill switch — blocks all new orders."""
    kill_switch.engage(reason)
    return kill_switch.get_status()

@router.post("/kill-switch/disengage")
def disengage_kill_switch(kill_switch):
    """Disengage the kill switch — resumes normal trading."""
    kill_switch.disengage()
    return kill_switch.get_status()
```

Register this router in `src/admin/app.py` alongside the existing 11 routers.

### 2.5 Tests

New file `tests/test_kill_switch.py` (at least 5 tests):
- Engaged: file exists → `is_engaged()` returns True
- Not engaged: file doesn't exist → `is_engaged()` returns False
- Engage with reason: creates file with reason text
- Disengage: removes the file
- Executor integration: kill switch engaged → `execute()` returns `KILLED`, no broker call

New file `tests/test_daily_loss_tracker.py` (at least 8 tests):
- Persistence: write loss, new instance reads it back
- Day rollover: yesterday's losses reset to 0.0 today
- Atomic write: no `.tmp` files left behind
- Corruption recovery: corrupt JSON → graceful fallback to 0.0
- Only positive losses: negative/zero values rejected
- Limit check: accumulated >= limit → `is_limit_exceeded()` returns True
- File creation: parent directory created if missing
- Executor integration: limit exceeded → `execute()` returns `DAILY_LIMIT`, no broker call

---

## STEP 3: Queue Directory Configuration

**Problem:** The queue directory is currently set in `config/automation.yaml` as `queue_dir: "queue"` (a relative path). For Scanner integration, this needs to point to the Scanner's output directory, which is an absolute path on the host system.

### 3.1 Update `config/automation.yaml` comments

Add a comment explaining the queue_dir field:
```yaml
queue_reader:
  enabled: true
  poll_interval_seconds: 30
  # Path to the scanner's signal queue directory.
  # Contains pending/, processing/, completed/ subdirectories.
  # Can be absolute (e.g., /path/to/trifecta-market-scanner/queue)
  # or relative to the repo root.
  queue_dir: "queue"
```

### 3.2 Validate queue_dir on daemon startup

In `src/automation/daemon.py`, `validate()` method (around line 220–264), the queue_dir existence check already exists. Enhance it to:
1. Resolve relative paths against the repo root
2. Verify the `pending/`, `processing/`, `completed/` subdirectories exist (create them if missing)
3. Log the resolved absolute path at INFO level for debugging

```python
queue_dir = Path(self._config["queue_reader"]["queue_dir"]).resolve()
for subdir in ["pending", "processing", "completed"]:
    (queue_dir / subdir).mkdir(parents=True, exist_ok=True)
logger.info(f"Queue directory: {queue_dir}")
```

### 3.3 Ensure QueueReader uses the resolved path

In `src/automation/queue_reader.py`, verify that the `queue_dir` parameter is used as-is (it is — line ~45 takes it as a constructor arg). The daemon already passes it from config, so no change needed in QueueReader itself.

### 3.4 Tests

Add to existing `tests/test_queue_reader.py` or new `tests/test_queue_config.py` (at least 3 tests):
- Relative path: `queue_dir: "queue"` resolves correctly
- Absolute path: `/tmp/test_queue` works
- Auto-create subdirs: missing `pending/` is created on daemon validate

---

## STEP 4: Fix Failing Ollama Tests

**Problem:** Two tests fail when Ollama is not running locally. These tests should be skipped or mocked, not dependent on a local Ollama instance.

### 4.1 Identify the failing tests

Run the full test suite:
```bash
pytest tests/ -v 2>&1 | grep -E "(FAIL|ERROR)"
```

The 2 known failures are Ollama-dependent. They likely attempt to connect to `http://localhost:11434`.

### 4.2 Fix strategy

For each failing test:
1. If it's a unit test that should work without Ollama: **mock the Ollama call** using `unittest.mock.patch`
2. If it's an integration test that intentionally tests Ollama connectivity: **add a skip marker**:
   ```python
   @pytest.mark.skipif(
       not _ollama_available(),
       reason="Ollama not running at localhost:11434"
   )
   ```

Create a helper function:
```python
def _ollama_available() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False
```

### 4.3 Verify

After fixing, run:
```bash
pytest tests/ -v
```

All tests must pass (green) — both with and without Ollama running.

---

## STEP 5: Documentation and Completion

### 5.1 Create task report

Create `docs/TASK_022_REPORT.md` with:
- Summary of all features implemented
- File listing (new and modified files with line counts)
- Test counts (new tests per feature, total suite count)
- Exit criteria status (all items checked off)
- Architectural decisions made during implementation
- Known limitations or follow-up items

### 5.2 Update `PROJECT_BRIEF.md`

Add Task 022 to the task log table:
```
| 022 | Live Trading Readiness | Feature | Configurable paper/live mode, kill switch, daily loss tracker, queue config, Ollama test fixes |
```

Update the "Current State" section to reflect the new capabilities.

### 5.3 Update `ECOSYSTEM_CONTEXT.md`

In the Stock Trader Engine section, update the description to note that live trading is now configurable (paper/live with two-key safety lock).

### 5.4 Verify vendor cleanliness

```bash
cd vendor/TradingAgents && git status
# Must show: nothing to commit, working tree clean
```

---

## EXIT CRITERIA

All 12 must pass before reporting completion:

1. **All existing tests pass.** `pytest tests/ -v` — zero regressions from the existing ~43 test files.
2. **Paper mode is still the default.** `TradeExecutor()` with no `paper` arg → `TradingClient(paper=True)`.
3. **Live mode requires two keys.** Config `mode: live` alone → paper mode. `--confirm-live` alone → paper mode. Both together → live mode.
4. **Kill switch blocks orders.** `HALT_TRADING` file exists → `execute()` returns `KILLED`, no broker call made.
5. **Kill switch admin API works.** `POST /safety/kill-switch/engage` creates the file; `POST /safety/kill-switch/disengage` removes it; `GET /safety/status` returns current state.
6. **Daily loss tracked across restarts.** New `DailyLossTracker` instance with same file path reads accumulated loss from previous instance.
7. **Daily loss resets at midnight UTC.** Tracker returns 0.0 when the UTC date changes.
8. **Daily loss limit blocks orders.** When accumulated loss >= limit → `execute()` returns `DAILY_LIMIT`, no broker call.
9. **Queue directory is configurable.** Absolute path in `config/automation.yaml` → daemon reads from that path. Subdirectories auto-created if missing.
10. **Ollama tests fixed.** `pytest tests/ -v` passes with and without Ollama running locally.
11. **Config backward compatible.** Existing `config/automation.yaml` without new fields works. Missing `config/execution.yaml` works with safe defaults.
12. **Vendor submodule clean.** Zero modifications in `vendor/TradingAgents`.

**Target:** 24–30 new tests across 4–5 new test files. Total test file count should be ~47–48.

---

## REFERENCE: Key Files and Line Numbers

| File | Lines | Relevant Sections |
|------|-------|-------------------|
| `src/execution/executor.py` | 195 | `__init__` (27–53, **paper=True at 45–50**), `execute()` (60–109), `_submit_bracket_order` (111–138), `_submit_market_order` (140–162), `_create_audit_entry` (164–182), `_save_audit` (184–194) |
| `src/execution/position_manager.py` | 280 | Safety constants (15–19: MAX_POSITION_PCT=15.0, MIN_QUALITY=8.0, MAX_RISK=2.0, MIN_RR=1.5), `AccountState` (22–28), `Position` (31–41), `OrderCalculation` (44–83), `calculate_order` (149–279) |
| `src/execution/trade_params.py` | 383 | `TradeParams` dataclass (113–152), `extract_trade_params` (155–308), `is_actionable` property (138–147) |
| `src/run_analysis.py` | ~770 | `get_config` (37–65), `_make_trading_client` (70–75, **paper=True hardcoded**), `_run_execution_flow` (508–596), CLI args (704–706), main dispatch (754–770) |
| `src/automation/daemon.py` | ~265 | `_CONFIG_DEFAULTS` (25–55), `start()` (89–111), queue reader setup (154–176), `validate()` (220–264) |
| `src/automation/queue_reader.py` | — | Constructor takes `queue_dir`, polls `pending/`, moves to `processing/` then `completed/` |
| `src/admin/app.py` | — | FastAPI app factory, 11 existing routers, port 8420 |
| `src/admin/health.py` | 328 | `compute_health_color()` (33–70), `GET /health` (175–327) |
| `config/automation.yaml` | 36 | Scheduler, queue_reader (queue_dir at line ~15), accuracy, admin_api configs |
| `tests/test_executor.py` | 163 | `test_paper_only_hardcoded` (69–80), mock pattern: `patch("src.execution.executor.TradingClient")` |

---

## FUTURE CONTEXT (not in this sprint — read for architectural awareness)

### Stock Trader UI (tt-curser)

A React + Fastify frontend exists in a separate repo (`tt-curser`). It already has portfolio views, trade history, and settings. Future sprints will integrate the admin API safety endpoints into this UI. Keep this in mind:

1. **KillSwitch** — the `get_status() -> dict` method enables the UI to show a safety indicator and allow engage/disengage from the dashboard.
2. **DailyLossTracker** — the `get_status() -> dict` method enables the UI to show a real-time loss meter.
3. **Trading mode** — the admin API should expose whether the engine is in paper or live mode so the UI can show a prominent indicator.

### Crypto Trader Parity

The Crypto Trader Engine (separate repo: `trifecta-crypto-trader`) has an 8-layer safety gate stack. This sprint brings the Stock Trader to approximate parity with:
- Kill switch (HALT_TRADING file) ✅
- Daily loss limit with persistence ✅
- Position safety gates (already existed) ✅
- Audit-first logging (already existed) ✅

Features the Crypto Trader has that are NOT in this sprint (future consideration):
- Limit order support (market orders only for now)
- WebSocket SL/TP monitoring (no SL/TP monitor exists yet for stocks)
- Order recovery on restart (stub in config, implementation deferred)

### Ecosystem Agent

The Trifecta Ecosystem agent will eventually coordinate work across all repos. It uses Linear issues to track what needs to happen. This prompt covers TRI-31 (live trading), TRI-32 (push commits), TRI-33 (queue config), TRI-34 (test fixes). TRI-35 (Engine ↔ Platform UI integration) is NOT in scope here — it requires cross-repo coordination.

Also read `ECOSYSTEM_CONTEXT.md` in the repo root for the full integration picture.

---

## IMPLEMENTATION ORDER

Execute in this order — each step builds on the previous:

1. **Step 0: Housekeeping** — commit docs, push, create branch
2. **Step 1: Paper/live mode** — the core enabler; all subsequent safety features depend on mode awareness
3. **Step 2: Safety infrastructure** — kill switch + daily loss; must exist before anyone uses live mode
4. **Step 3: Queue config** — independent, small, low risk
5. **Step 4: Ollama test fixes** — independent, small, ensures clean test baseline
6. **Step 5: Documentation** — after all features are implemented

---

*End of development prompt. Execute all steps sequentially. Report completion with test counts and exit criteria status.*
