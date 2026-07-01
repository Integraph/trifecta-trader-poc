# Task 021 — UI Polish: Accessibility, A/B Polling Safety & Help Text Verification

**Type:** Bug Fix / Polish
**Depends on:** Task 018 (LLM Config Editor), Task 020 (Admin UI Bug Fixes)
**Repo:** trifecta-trader-poc
**Priority:** Medium

---

## Objective

Address three gaps identified during the Task 018 report review. All are polish-level issues — the features work correctly but have minor accessibility, robustness, and content accuracy gaps.

1. **InfoTooltip accessibility** — Add missing `aria-describedby` attribute to comply with the original Task 018 spec.
2. **A/B comparison polling safety** — Add a maximum polling timeout to prevent indefinite `setInterval` if an analysis task gets stuck.
3. **Help text content audit** — Verify that all 17 `SETTINGS_HELP` entries match the definitions specified in Task 018 §4.3.

---

## Background

Task 018 delivered a comprehensive set of features (YAML config management, CRUD API, sanity check, A/B comparison, tooltips). Tasks 019 and 020 fixed functional bugs. This task handles the remaining non-functional gaps found during the spec-vs-report review.

---

## Part 1 — InfoTooltip Accessibility (`aria-describedby`)

### 1.1 Current State

`admin-ui/src/components/shared/InfoTooltip.tsx` currently has:
- `role="tooltip"` on the popover div ✓
- `aria-label="Show help"` on the trigger button ✓
- `aria-expanded={open}` on the trigger button ✓

**Missing:** The spec (Task 018 §4.1) explicitly requires `aria-describedby` to link the trigger button to the tooltip content. This attribute tells screen readers which element contains the descriptive text for the button.

### 1.2 Required Changes

**`admin-ui/src/components/shared/InfoTooltip.tsx`:**

1. Generate a stable unique ID for each tooltip instance (use `useId()` from React 18 or a simple counter).
2. Add `id={tooltipId}` to the popover div.
3. Add `aria-describedby={open ? tooltipId : undefined}` to the trigger button (only set when tooltip is visible).

```tsx
// Example implementation
import { useId } from 'react';

function InfoTooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const tooltipId = useId();

  return (
    <span className="relative inline-flex">
      <button
        aria-label="Show help"
        aria-expanded={open}
        aria-describedby={open ? tooltipId : undefined}
        onClick={() => setOpen(!open)}
        // ... existing props
      >
        {/* (i) icon */}
      </button>
      {open && (
        <div
          id={tooltipId}
          role="tooltip"
          // ... existing props
        >
          {text}
        </div>
      )}
    </span>
  );
}
```

### 1.3 Verification

- Inspect the rendered HTML: when tooltip is open, the button's `aria-describedby` should reference the tooltip div's `id`.
- When tooltip is closed, `aria-describedby` should not be present (or be `undefined`).
- Each tooltip instance on the page should have a unique ID (no collisions).

---

## Part 2 — A/B Comparison Polling Timeout

### 2.1 Current State

`admin-ui/src/components/test-run/TestRunPage.tsx` uses `setInterval` at 3-second intervals to poll `GET /test-run/ab/{ab_id}`. The interval clears when `status === 'complete'`. However, there is **no maximum attempt counter or timeout ceiling**. If an analysis task hangs indefinitely (e.g., an LLM provider is unresponsive beyond the ThreadPoolExecutor timeout), the frontend will poll forever.

### 2.2 Required Changes

**`admin-ui/src/components/test-run/TestRunPage.tsx` — `ABCompareMode` component:**

Add a max-poll mechanism. After a reasonable ceiling (e.g., 5 minutes = 100 polls at 3s intervals), stop polling and show a timeout message.

```tsx
// Constants
const AB_POLL_INTERVAL_MS = 3000;
const AB_MAX_POLL_ATTEMPTS = 100; // 100 × 3s = 5 minutes

// Inside the polling logic
const pollCountRef = useRef(0);

// In the setInterval callback:
pollCountRef.current += 1;
if (pollCountRef.current >= AB_MAX_POLL_ATTEMPTS) {
  clearInterval(pollRef.current);
  pollRef.current = null;
  // Set a timeout error state
  setAbError('A/B comparison timed out after 5 minutes. The analysis may still be running on the server — check the Analyses page for results.');
  return;
}

// Reset counter when starting a new comparison
pollCountRef.current = 0;
```

### 2.3 UI for Timeout State

When the timeout fires:
- Stop the polling interval
- Show a yellow/amber warning banner (not a red error — the server-side tasks may still complete)
- Message: "A/B comparison timed out after 5 minutes. The analysis may still be running on the server — check the Analyses page for results."
- Show whatever partial results have been received (one side may have completed)
- Allow the user to click "Resume Polling" to restart with a fresh counter (in case the task is just slow, not stuck)

### 2.4 Cleanup on Unmount

Ensure the `useEffect` cleanup function clears the interval when the component unmounts or when the user switches back to Single Run mode. This should already be handled but verify it.

---

## Part 3 — Help Text Content Audit

### 3.1 Current State

`admin-ui/src/components/config/config-help.ts` contains 17 help text entries in `SETTINGS_HELP`. The Task 018 report confirms the count and section coverage, but the actual text content was not verified against the definitions in Task 018 spec §4.3.

### 3.2 Required Verification

Compare each `SETTINGS_HELP` entry against the spec definitions below. Fix any discrepancies in wording, missing caveats, or incorrect ranges.

**Scheduler section (6 entries):**

| Key | Required Content |
|-----|-----------------|
| `scheduler.enabled` | "Master switch for the daily watchlist scanner. When disabled, no scheduled scans will run. The queue reader and accuracy updater operate independently." |
| `scheduler.watchlist_hour` | "Time of day (24h format) in the configured timezone when the watchlist scan runs. Default 8:30 AM ET gives ~60 minutes before market open for analyses to complete." (Note: key was renamed from `hour` to `watchlist_hour` in Task 020 Bug 3) |
| `scheduler.watchlist_minute` | Same tooltip as hour — they describe the time pair together. |
| `scheduler.hybrid_config` | "Which LLM configuration to use for scheduled analyses. This determines which AI models handle tool-calling, quick reasoning, and deep reasoning (Risk Judge). Edit configs in the LLM Configuration section below." |
| `scheduler.watchlist` | "Name of the watchlist file to scan (from config/watchlists/). Each watchlist contains a list of ticker symbols to analyze." |
| `scheduler.publish` | "When enabled, analysis results are automatically published to Supabase for the Platform UI. Disable for testing or when Supabase is not configured." |

**Queue Reader section (4 entries):**

| Key | Required Content |
|-----|-----------------|
| `queue_reader.enabled` | "Master switch for the file-based queue reader. When enabled, the daemon polls the queue directory for candidate JSON files from the Market Scanner." |
| `queue_reader.poll_interval` | "How often (in seconds) to check for new queue candidates. Lower values mean faster processing but more filesystem I/O. Range: 5-300. Takes effect immediately." |
| `queue_reader.max_retries` | "Maximum number of retry attempts for a failed analysis before marking the candidate as permanently failed. Range: 0-10. Takes effect immediately." |
| `queue_reader.cooldown` | "Minimum wait time (in seconds) between consecutive analyses to respect LLM rate limits. Range: 10-600. Takes effect immediately." |

**Accuracy section (2 entries):**

| Key | Required Content |
|-----|-----------------|
| `accuracy.enabled` | "Master switch for the signal accuracy tracker. When enabled, tracks price movements at T+1, T+5, and T+10 trading days after each signal." |
| `accuracy.backfill_on_start` | "When enabled, automatically scores all existing untracked analyses when the daemon first starts. Useful after adding accuracy tracking to an existing deployment." |

**Admin API section (1 entry):**

| Key | Required Content |
|-----|-----------------|
| `admin_api.port` | "TCP port for the Admin API server. Default 8420. Requires daemon restart to take effect." |

**Supabase section (2 entries):**

| Key | Required Content |
|-----|-----------------|
| `supabase.write_enabled` | "When enabled, analyses are published to the Supabase signals table. Disable to run analyses without publishing. Takes effect immediately." |
| `supabase.signal_ttl_hours` | "How many hours a published signal remains active in Supabase before being considered stale. The Platform UI uses this to filter signals. Range: 1-168 (1 week)." |

**Section-level tooltips (2 entries):**

| Key | Required Content |
|-----|-----------------|
| `watchlist_manager` | "Watchlists define which tickers the scheduler analyzes. Each watchlist is a YAML file in config/watchlists/. The active watchlist is set in the Scheduler config above." |
| `llm_configs` | "Hybrid LLM configurations define which AI providers and models handle each agent role. Tool-calling agents need models that support function calling (e.g., Anthropic Claude, OpenAI GPT-4). Reasoning agents can use any model including local Ollama models." |

### 3.3 What to Fix

- If text differs from the spec definitions above, update to match the spec.
- If keys were renamed (e.g., `hour` → `watchlist_hour` per Task 020), ensure the tooltip keys match.
- Do NOT change working functionality — this is a content-only update to the help strings.

---

## Deliverables

| # | Deliverable | Scope |
|---|-------------|-------|
| 1 | `InfoTooltip.tsx` — add `aria-describedby` with unique IDs | TypeScript |
| 2 | `TestRunPage.tsx` — add max-poll timeout + timeout UI state | TypeScript |
| 3 | `config-help.ts` — verify/fix all 17 help text entries against spec | TypeScript |

---

## Exit Criteria

### Accessibility (3 criteria)

1. InfoTooltip trigger button has `aria-describedby` linking to the tooltip div when open
2. Each InfoTooltip instance on the page has a unique `id` (no collisions)
3. `aria-describedby` is not present (or undefined) when tooltip is closed

### A/B Polling Safety (4 criteria)

4. A/B comparison polling stops after 100 attempts (5 minutes at 3s intervals)
5. Timeout displays a warning message (not an error) with guidance to check Analyses page
6. Partial results from a completed side are still shown when timeout fires
7. "Resume Polling" button available after timeout to restart with fresh counter

### Help Text (2 criteria)

8. All 17 `SETTINGS_HELP` entries match the content defined in this spec (§3.2)
9. All tooltip keys match the current field names (accounting for Task 020 renames)

### Build & Quality (3 criteria)

10. `tsc --noEmit` — 0 errors
11. `npm run build` — 0 errors
12. No regressions in existing tests

---

## Implementation Notes

### React `useId()` for tooltips

React 18's `useId()` hook generates stable, unique IDs that work correctly with SSR and concurrent rendering. It's the recommended approach for accessibility IDs. Since the admin-ui already uses React 18, this is available without any dependencies.

### Polling timeout UX

The timeout should be a **warning**, not an error. The server-side analysis tasks continue running regardless of whether the frontend is polling. The user can always check the Analyses page for results. The "Resume Polling" button is a convenience for cases where the analysis is legitimately slow (e.g., deep reasoning with a large model) rather than stuck.

### Help text changes are low-risk

Modifying string constants in `config-help.ts` has no behavioral impact. The only risk is a typo breaking the build (unlikely with TypeScript's type checking on the object keys).

---

## File Inventory (expected changes)

### Modified files
- `admin-ui/src/components/shared/InfoTooltip.tsx` (accessibility fix)
- `admin-ui/src/components/test-run/TestRunPage.tsx` (polling timeout)
- `admin-ui/src/components/config/config-help.ts` (help text verification/fixes)

### No new files expected
