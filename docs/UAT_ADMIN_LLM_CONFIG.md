# User Acceptance Test — Admin LLM Configuration & A/B Comparison

**Version:** 1.0
**Date:** 2026-03-16
**Audience:** Administrator / Managing Director
**Prerequisites:** Admin API running on `localhost:8420`, Admin UI running on `localhost:5174`

---

## Starting the Services

Before running the UAT, make sure both the backend API and the admin frontend are running.

**Terminal 1 — Admin API:**
```
cd ~/path/to/trifecta-trader-poc    # Must be the project root, not a subfolder
python -m src.run_daemon --api
```
Alternatively, to start just the API without the scheduler/queue:
```
cd ~/path/to/trifecta-trader-poc
python -m src.admin.app
```
The API should report that it is listening on port 8420. Verify by visiting `http://localhost:8420/health` in a browser — you should see a JSON health response.

**Terminal 2 — Admin UI:**
```
cd trifecta-trader-poc/admin-ui
npm run dev
```
The Vite dev server should report that it is listening on port 5174. Open `http://localhost:5174` in your browser.

---

## Part 1: Navigating to LLM Configuration

**Step 1.1 — Open the admin dashboard.**
Open `http://localhost:5174` in your browser. You should see the Dashboard page with the left sidebar visible.

**Step 1.2 — Navigate to Configuration.**
In the left sidebar, click **Configuration** (gear icon). The page header should read "Configuration."

**Step 1.3 — Locate the LLM Configurations panel.**
Scroll down past the Automation, Supabase, and Watchlist panels. The fourth panel is titled **LLM Configurations** and shows a count of how many configs exist (e.g., "13 configs").

**Expected result:** You see a left-hand config list with names like `all_cloud`, `hybrid_qwen`, `hybrid_haiku_tools`, etc. The currently active config has a small blue dot (●) next to its name.

---

## Part 2: Viewing and Editing an Existing Config

**Step 2.1 — Select a config.**
Click any config name in the left list, for example `hybrid_haiku_tools`. The right-hand editor panel loads with that config's details.

**Step 2.2 — Review the three slot sections.**
The editor shows three bordered sections:

- **Tool Calling** — Provider dropdown + Model text field. This is the LLM used for function/tool calls.
- **Reasoning Quick** — Provider + Model for fast reasoning tasks (often a local Ollama model).
- **Reasoning Deep** — Provider + Model for deep analysis (often a cloud model like Claude Sonnet).

Each section has a Provider dropdown (options include `anthropic`, `ollama`, `openai`, `google`) and a Model text input showing the specific model identifier.

**Step 2.3 — Review the Enhancement section.**
Below the three slots is an **Enhancement** section with:

- **Enhance local** — checkbox (Enabled/Disabled). When enabled, local model outputs are post-processed.
- **Style** — dropdown (options like `financial_analysis`, `execution_params_only`, etc.).
- **Enhance deep** — checkbox for deep enhancement pass.
- **Deep style** — dropdown for the deep enhancement style.

**Step 2.4 — Make a test edit.**
Change the Reasoning Quick model from its current value to something recognizable, like `test-model-name`. Do NOT click Save yet.

**Expected result:** The text field updates to show your typed value. No network request is made until you click Save.

---

## Part 3: Running a Sanity Check

The sanity check tests whether the providers and models configured for a given config are actually reachable.

**Step 3.1 — Click the Sanity Check button.**
With a config selected, click the **Sanity Check** button (flask icon) in the action bar below the Enhancement section. A spinner appears while the check runs.

**Step 3.2 — Review the results.**
After a few seconds, a sanity check card appears below the action buttons. It shows three checks corresponding to the three slots:

- **Tool Calling** — provider/model, pass/fail, latency in ms
- **Reasoning Quick** — provider/model, pass/fail, latency in ms
- **Reasoning Deep** — provider/model, pass/fail, latency in ms

A green checkmark (✓) means the provider responded successfully. A red X (✗) means the provider could not be reached. A dash (—) means the check was skipped.

At the bottom, an **Overall** status appears: PASS (3/3), PARTIAL (1/3 or 2/3), or FAIL (0/3).

**Expected result:** If Ollama is not running locally, the Reasoning Quick slot will fail. Cloud providers (Anthropic, OpenAI) should pass if API keys are configured in the environment.

**Step 3.3 — Revert your test edit.**
If you changed the model name in Step 2.4, either reload the page or re-select the config from the list to discard unsaved changes.

---

## Part 4: Creating a New Config

This tests the Task 019 fix — configs created via the admin UI should be immediately usable from the CLI.

**Step 4.1 — Click "New Config."**
At the bottom of the config list on the left, click **+ New Config**. The right-hand panel switches to a creation form.

**Step 4.2 — Enter a name.**
Type a test name like `uat_test_config` in the Name field. The hint says "alphanumeric + underscores."

**Step 4.3 — Click "Create Config."**
Click the blue **Create Config** button. The new config appears in the left list and is auto-selected. All 10 fields are pre-filled with sensible defaults.

**Step 4.4 — Customize the new config.**
Set the fields to meaningful values for testing. For example:

| Slot | Provider | Model |
|------|----------|-------|
| Tool Calling | `anthropic` | `claude-haiku-4-5-20251001` |
| Reasoning Quick | `ollama` | `qwen2.5:14b` |
| Reasoning Deep | `anthropic` | `claude-sonnet-4-5-20250929` |

Leave Enhancement settings at their defaults for now.

**Step 4.5 — Save.**
Click the blue **Save** button. A green "Saved" toast appears briefly.

**Step 4.6 — Verify the new config count.**
The panel header should now show one more config than before (e.g., "14 configs" instead of "13 configs").

**Expected result:** The config is created, saved, and visible in the list.

---

## Part 5: Verifying CLI Accepts the New Config (Task 019)

This is the core verification for Task 019. Open a new terminal.

**Step 5.1 — Check that the CLI recognizes the new config.**
```
cd trifecta-trader-poc
python3 -c "
from src.hybrid_llm import CONFIGS
print('uat_test_config' in CONFIGS)
print(f'Total configs: {len(CONFIGS)}')
"
```

**Expected result:** Prints `True` and the total config count matches the UI.

**Step 5.2 — Verify argparse accepts it.**
```
python3 -c "
import argparse
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser = argparse.ArgumentParser()
parser.add_argument('--hybrid', type=str, choices=list(_hybrid_configs.keys()))
args = parser.parse_args(['--hybrid', 'uat_test_config'])
print('Accepted:', args.hybrid)
"
```

**Expected result:** Prints `Accepted: uat_test_config` with no error.

**Step 5.3 — Verify --help shows the new config.**
```
python3 -c "
import argparse
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser = argparse.ArgumentParser()
parser.add_argument('--hybrid', type=str, choices=list(_hybrid_configs.keys()))
parser.print_help()
" 2>&1 | grep uat_test_config
```

**Expected result:** The grep finds `uat_test_config` in the help output's choices list.

---

## Part 6: Running a Single Test Analysis

**Step 6.1 — Navigate to Test Run.**
In the left sidebar, click **Test Run** (flask icon). The page header reads "Test Run."

**Step 6.2 — Confirm you are in Single Run mode.**
At the top of the page, a mode toggle shows two buttons: **Single Run** (active/highlighted) and **A/B Compare**. Single Run should be selected by default.

**Step 6.3 — Fill in the test run form.**
The form has four fields:

- **Ticker** — Type a stock symbol, e.g., `NVDA`.
- **Hybrid Config** — A dropdown listing all available configs. Verify that `uat_test_config` appears in this dropdown. Select it.
- **Trade Date** — Defaults to today. Leave as-is or pick another date.
- **Publish to Supabase** — Checkbox. Leave unchecked for testing. If checked, a yellow warning appears: "Writes to production."

**Step 6.4 — Run the analysis.**
Click the orange **Run Analysis** button. The button text changes to "Analyzing NVDA..." and a task polling card appears below, showing the analysis status.

**Step 6.5 — Wait for results.**
The poller updates every 3 seconds. When complete, a result card appears showing:

- **Decision** — BUY, SELL, or HOLD in a colored badge (green/red/gray).
- **Quality score** — A composite score out of 10, with a breakdown bar chart.
- **Trade Parameters** — Entry price, stop loss, take profit, position size, etc.
- **Cost** — Total API cost in USD.
- **Elapsed** — Total time in seconds.
- **Published** — Yes or No.

A "Full Result" expandable JSON viewer is available at the bottom of the card for detailed inspection.

**Step 6.6 — Check Recent Runs.**
Below the result card, a **Recent Runs** table shows historical test runs with ticker, config, decision, quality, cost, and date.

**Expected result:** The analysis completes with a valid decision and quality score. If the Ollama model in `uat_test_config` is not available, the analysis may fail or use a fallback — this is expected and demonstrates the sanity check's value.

---

## Part 7: Running an A/B Comparison

**Step 7.1 — Switch to A/B Compare mode.**
Click the **A/B Compare** button in the mode toggle at the top. The form changes to show two config dropdowns.

**Step 7.2 — Fill in the A/B form.**
The form has five fields:

- **Ticker** — Type a stock symbol, e.g., `AAPL`.
- **Trade Date** — Defaults to today.
- **Config A** — Dropdown of all configs. The currently active config is marked "(current)." Select a config you trust, e.g., `hybrid_haiku_tools`.
- **Config B** — Dropdown of all configs. Select the test config `uat_test_config`, or another config you want to compare.
- **Publish to Supabase** — Leave unchecked for testing.

**Step 7.3 — Run the comparison.**
Click the purple **Run A/B Comparison** button. The button text changes to "Running A/B..." and two side-by-side panels appear below labeled **Config A** and **Config B**, each with a spinner.

**Step 7.4 — Review side-by-side results.**
As each analysis completes, its panel fills in with the same result card layout as Single Run (decision, quality, trade params, cost, elapsed). The two configs may finish at different times.

**Step 7.5 — Review the Comparison Summary.**
Once both analyses complete, a **Comparison Summary** bar appears above the side-by-side panels. It shows four key metrics:

- **Decision** — Whether both configs reached the same decision (e.g., "Both BUY ✓") or diverged (e.g., "A=BUY vs B=HOLD ⚠").
- **Quality delta** — The difference in composite quality scores between B and A (e.g., "+0.3 (B vs A)" in green, or "-1.2 (B vs A)" in red).
- **Cost** — Which config is more expensive and by what factor (e.g., "A is 2.3x more expensive").
- **Speed** — Which config is faster and by what percentage (e.g., "Config B is 34% faster").

**Expected result:** Both panels fill with results. The comparison summary gives a clear picture of the trade-offs between the two configurations.

---

## Part 8: Cloning a Config

**Step 8.1 — Go back to Configuration.**
Navigate to **Configuration** in the sidebar. Select `hybrid_haiku_tools` from the config list.

**Step 8.2 — Click Clone.**
Click the **Clone** button in the action bar. A text input appears below asking for the new config name.

**Step 8.3 — Enter a clone name.**
Type `hybrid_haiku_tools_v2` and click the green **Clone** button.

**Expected result:** The new config appears in the list and is auto-selected. It has identical settings to the original. A green toast says "Cloned to 'hybrid_haiku_tools_v2'."

---

## Part 9: Deleting Test Configs

**Step 9.1 — Select a test config.**
Click `uat_test_config` in the config list.

**Step 9.2 — Click Delete.**
Click the **Delete** button. It turns red and the label changes to **Confirm Delete** (two-click safety). Click it again to confirm.

**Expected result:** The config disappears from the list. The config count decreases by one.

**Step 9.3 — Verify the CLI rejects the deleted config.**
In the terminal:
```
python3 -c "
import argparse
from src.hybrid_llm import CONFIGS as _hybrid_configs
parser = argparse.ArgumentParser()
parser.add_argument('--hybrid', type=str, choices=list(_hybrid_configs.keys()))
try:
    parser.parse_args(['--hybrid', 'uat_test_config'])
    print('ERROR: Should have been rejected')
except SystemExit:
    print('Correctly rejected')
"
```

**Expected result:** Prints `Correctly rejected`.

**Step 9.4 — Clean up the clone.**
Repeat Steps 9.1–9.2 for `hybrid_haiku_tools_v2`.

**Note:** The currently active config (shown with a blue "active" badge) cannot be deleted. The Delete button is grayed out and shows a tooltip: "Cannot delete active config."

---

## Part 10: Verifying InfoTooltips

Throughout the Configuration page, small (i) icons appear next to each field label. These are InfoTooltips that explain what each setting does.

**Step 10.1 — Hover over a tooltip icon.**
Hover over the (i) icon next to any field in the LLM Configurations panel — for example, next to the "LLM Configurations" panel header itself, or next to any of the enhancement settings.

**Expected result:** A tooltip popover appears with a plain-English explanation of what that setting controls.

---

## Summary Checklist

| # | Test | Pass/Fail |
|---|------|-----------|
| 1 | Admin UI loads at `localhost:5174` | |
| 2 | Configuration page shows LLM Configurations panel with correct count | |
| 3 | Selecting a config loads its editor with three slot sections | |
| 4 | Sanity Check runs and shows per-slot pass/fail with latency | |
| 5 | New config can be created via "+ New Config" | |
| 6 | CLI accepts the new config name (`python3 -c` test) | |
| 7 | `--help` output includes the new config name | |
| 8 | New config appears in Test Run → Single Run dropdown | |
| 9 | Single Run analysis completes with result card | |
| 10 | A/B Compare mode shows two config dropdowns | |
| 11 | A/B Comparison runs and shows side-by-side results | |
| 12 | Comparison Summary shows decision match, quality delta, cost, speed | |
| 13 | Clone creates a copy with identical settings | |
| 14 | Delete requires two clicks (confirmation) and removes the config | |
| 15 | CLI rejects a deleted config name | |
| 16 | Active config cannot be deleted (button grayed out) | |
| 17 | InfoTooltips display on hover | |
