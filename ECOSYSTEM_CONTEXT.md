# Trifecta Trading Ecosystem — Integration Context

**Last updated:** June 29, 2026
**Maintained by:** Trifecta Ecosystem Agent
**Source of truth:** `AI/ECOSYSTEM_CONTEXT.md` — repo copies are mirrors, never edit them directly
**Purpose:** Give every development agent a holistic view of the entire application

---

## 1. System Overview

The Trifecta Trading Ecosystem is a multi-agent AI trading platform consisting of five components that form a signal-to-trade pipeline:

```
                        ┌─────────────────────┐
                        │   MARKET SCANNER     │
                        │  (Python/Streamlit)  │
                        │  Produces signals    │
                        └────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌───────────────────┐     ┌───────────────────┐
        │  STOCK TRADER     │     │  CRYPTO TRADER    │
        │    ENGINE         │     │    ENGINE         │
        │  (Python)         │     │  (Python/LangGraph)│
        │  Alpaca execution │     │  Kraken execution │
        └────────┬──────────┘     └────────┬──────────┘
                 │                         │
                 ▼                         ▼
        ┌───────────────────┐     ┌───────────────────┐
        │  STOCK TRADER UI  │     │  CRYPTO TRADER UI │
        │  (React/Vite +    │     │  (TBD — deferred  │
        │   Fastify)        │     │   to Ecosystem    │
        │  tt-curser        │     │   agent)          │
        └───────────────────┘     └───────────────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                    ┌─────────▼──────────┐
                    │     SUPABASE       │
                    │  (Shared data &    │
                    │   auth layer)      │
                    └────────────────────┘
```

---

## 2. Component Registry

| Component | Repo | Stack | Status | Admin UI |
|-----------|------|-------|--------|----------|
| Market Scanner | `trifecta-market-scanner` | Python, Streamlit, APScheduler | S7 complete (511 tests, 51 modules). All Linear issues Done. | Streamlit dashboard (port 8501), 6-tab admin settings page |
| Stock Trader Engine | `trifecta-trader-poc` | Python, FastAPI admin, Alpaca | **v0.3.0 vendor upgrade DONE (TRI-66, 2026-07-02): zero-mod restored, LangGraph 1.x pinned.** Engine-first MVP: **TRI-70 (local-model benchmark) is now the active priority**; live trading (TRI-31) DEFERRED under the paper-only MVP. | FastAPI admin API + React admin-ui/ |
| Stock Trader UI | `tt-curser` | React 18, TypeScript, Vite, Fastify, Zustand | Tasks 001-004 complete; **Task 005 was never implemented — commit 3bc491d is mislabeled (TRI-21, reverted to Todo)**; backend build red (TRI-40). UI work is deferred under the engine-first MVP. | Settings page with API keys, notifications, preferences |
| Crypto Trader Engine | `trifecta-crypto-trader` | Python, LangGraph, CCXT (Kraken) | M6 complete. M7 parent In Progress (TRI-7), sub-tasks in Backlog. | None — CLI only. Standalone UI deferred to Ecosystem agent |
| Crypto Trader UI | TBD | TBD (Ecosystem agent decides) | Not started | N/A |

---

## 3. Integration Points

### 3.1 Scanner → Traders (JSON File Queue)

The Scanner produces signals and writes them to a JSON file queue. Both trader engines read from this queue.

```
Scanner writes to:     queue/pending/{timestamp}_{ticker}_{asset_type}.json
Trader picks up:       queue/pending/ → moves to queue/processing/ → queue/completed/
```

**Signal JSON schema** (per file):
```json
{
  "ticker": "AAPL",
  "asset_type": "stock",
  "composite_score": 0.78,
  "signals": { ... },
  "timestamp": "2026-06-27T14:30:00Z"
}
```

The `asset_type` field (`"stock"` or `"crypto"`) determines which trader engine picks up the signal.

**Current state:**
- Stock Trader Engine: reads from queue via `src/automation/queue_reader.py` and `src/automation/daemon.py` ✅
- Crypto Trader Engine: does NOT yet read from the scanner queue ❌ (runs independently via CLI)
- **Linear item needed:** Crypto Trader queue integration

### 3.2 Scanner → Supabase (Signals Table)

The Scanner also writes signals to a Supabase `signals` table with 15 columns (snake_case). The `asset_type` column enables downstream filtering.

**Key columns:** `id`, `ticker`, `asset_type`, `composite_score`, `signal_type`, `confidence`, `timestamp`, `created_at`

### 3.3 Stock Trader Engine → Supabase

The Stock Trader writes trade results, portfolio state, and signal accuracy data to Supabase via `src/integration/supabase_writer.py`.

### 3.4 Stock Trader Engine → Alpaca

Executes stock trades via Alpaca's API. Paper trading by default. Components:
- `src/execution/executor.py` — order submission
- `src/execution/position_manager.py` — position tracking
- Market data also sourced from Alpaca

### 3.5 Stock Trader UI → Backend → Supabase + Alpaca

```
React UI (tt-curser) → Fastify backend → Alpaca API (market data, orders)
                                       → Supabase (auth, portfolio data, signals)
                                       → WebSocket server (real-time updates)
```

### 3.6 Crypto Trader Engine → Kraken

Executes crypto trades via CCXT/Kraken. Paper trading by default. 8-layer safety gate stack.
- `src/execution/live_trading.py` — live order submission
- `src/execution/paper_trading.py` — simulated fills
- Market data via CCXT REST (M7 adds WebSocket for SL/TP)

### 3.7 Crypto Trader UI → TBD

Not yet created. Will need to connect to the Crypto Trader Engine (likely via a new API layer or direct Supabase reads). Technology stack deferred to Ecosystem agent.

---

## 4. Shared Infrastructure

### 4.1 Supabase

All components share a single Supabase instance for:
- **Authentication:** OAuth (Google, GitHub), email/password
- **Signals table:** Scanner writes, traders read, UIs display
- **Trade history:** Both trader engines write, UIs display
- **Portfolio state:** Engines write, UIs display

Environment variables (all apps):
```
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...  (backend only)
```

### 4.2 Vendor Submodules

The two **engines** use vendor submodules as architectural references (the **Scanner does NOT** — it has no `vendor/` or `.gitmodules`; its third-party code is its pip libraries):
- Stock Trader Engine: `vendor/TradingAgents` (`TauricResearch/TradingAgents`) — **zero modifications RESTORED (TRI-66 Done, 2026-07-02): submodule at v0.3.0 `85946c2`, byte-identical to upstream (`diff v0.3.0` = 0).**
- Crypto Trader Engine: `vendor/ai-hedge-fund-crypto` (`51bitquant/ai-hedge-fund-crypto`) — ZERO modifications enforced. *(Currency: pinned to a Sept-2025 `main` commit — see TRI-63.)*
- Scanner: **no vendor submodule.**

**Hard constraint:** No agent may modify any file inside a vendor submodule.

### 4.3 Config Patterns

All Python components use:
- YAML config files (`config/` directory)
- Pydantic v2 for validation
- `.env` files for secrets (API keys, database credentials)
- Atomic write pattern for persistent state (write `.tmp` → `os.replace()`)

---

## 5. Admin UI Requirements

Each component needs admin controls accessible in two ways:

### 5.1 Standalone Admin (per-component)

Each engine has (or needs) its own admin interface for independent operation:

| Component | Standalone Admin | Status |
|-----------|-----------------|--------|
| Market Scanner | Streamlit 6-tab admin page (port 8501) | ✅ Complete (S7) |
| Stock Trader Engine | FastAPI admin API + React admin-ui/ | ✅ Exists |
| Stock Trader UI | Settings page (API keys, notifications, preferences) | ✅ Exists |
| Crypto Trader Engine | None | ❌ Needs standalone admin (future sprint) |
| Crypto Trader UI | N/A | ❌ Not started |

### 5.2 Unified Admin (future — Ecosystem agent scope)

A single admin interface where all component controls are accessible from one place. This will be planned by the Ecosystem agent. Options include:
- Extending tt-curser's Settings page to aggregate all component controls
- A separate "Ecosystem Dashboard" that proxies to each component's admin API
- Micro-frontend approach embedding each component's admin

**This is NOT in scope for any current sprint.** It will be scoped when the Ecosystem agent is created.

---

## 6. What Each Agent Needs to Know

### If you're working on the Market Scanner:

You produce signals that feed both trader engines. Your output format (JSON file queue + Supabase signals table) is the contract between you and the traders. Changes to the signal schema affect downstream consumers. The `asset_type` field is critical — it routes signals to the correct trader.

Your Streamlit admin UI (S7) is the standalone admin for the scanner. The unified admin will eventually proxy to your controls.

**Integration dependencies:** Supabase signals table schema, JSON queue file format, `asset_type` routing values.

### If you're working on the Stock Trader Engine:

You consume signals from the Scanner via the JSON file queue (`queue/pending/`). You execute trades via Alpaca. Your results flow to Supabase and are displayed in the Stock Trader UI (tt-curser).

You already have a FastAPI admin API and a React admin-ui. These are your standalone admin. The unified admin will eventually aggregate your controls.

**Integration dependencies:** Scanner queue format, Alpaca API, Supabase tables (trades, portfolio, signals), tt-curser's API expectations.

### If you're working on the Crypto Trader Engine:

You will eventually consume signals from the Scanner (queue integration not yet built — this is a pending item). You execute trades via CCXT/Kraken. You currently have NO admin UI and NO standalone frontend — both are deferred to the Ecosystem agent.

Your M7 sprint focuses on execution layer hardening (persist daily loss, limit orders, WebSocket SL/TP, order recovery). Admin UI and queue integration are future sprints.

**Integration dependencies (current):** Kraken API via CCXT, config/config.yaml, Ollama LLM endpoints.
**Integration dependencies (future):** Scanner queue format, Supabase tables, standalone UI (TBD).

### If you're working on the Stock Trader UI (tt-curser):

You are the primary user interface for the stock trading side. You connect to the Fastify backend, which talks to Alpaca (market data, orders) and Supabase (auth, data). You do NOT directly connect to the Scanner or Crypto Trader engines.

Future work may add: scanner signal views, crypto portfolio views (if unified UI approach is chosen), or links to other component UIs.

**Integration dependencies:** Fastify backend API, Supabase auth, Alpaca market data, WebSocket server.

### If you're working on the Ecosystem Agent:

You coordinate across all components. You decide: unified UI strategy, Crypto Trader UI technology, cross-repo architecture decisions, integration contracts, and development prioritization. You assign work to Claude Code (all development — engines and UI). **Cursor is the UAT agent** — it runs the user-facing UI test scripts against the running app and reports acceptance pass/fail (UI-testing only; it does not write application code). See §10.

---

## 7. Pending Integration Items

These items need to be tracked in Linear and addressed by the appropriate agent:

| Item | Component | Priority | Notes |
|------|-----------|----------|-------|
| Crypto Trader queue integration | Crypto Trader Engine | High | Read scanner signals from queue/pending/ like the Stock Trader does |
| Crypto Trader Supabase writer | Crypto Trader Engine | High | Write trade results to Supabase (like Stock Trader's supabase_writer.py) |
| Crypto Trader standalone admin UI | Crypto Trader Engine | Medium | CLI-only today; needs at minimum a config/status dashboard |
| Crypto Trader standalone frontend | Crypto Trader UI (new) | Medium | Technology TBD by Ecosystem agent |
| Unified admin interface | Ecosystem | Low | Aggregate all component admin controls into one place |
| Scanner signal schema contract | Ecosystem | Medium | Formalize the JSON queue format as a shared contract |
| Cross-component health dashboard | Ecosystem | Low | Single view showing status of all engines |

---

## 8. Linear Project Structure

| Linear Project | Project ID | Component |
|---------------|------------|-----------|
| Trifecta Market Scanner | `1cd949fe-bf53-4427-a21b-185e64b7fe44` | Scanner engine + Streamlit dashboard |
| Trifecta Crypto Trader | `ff0079da-c1b0-48cc-b303-5bd4ea1180db` | Crypto engine (M7 queued) |
| Trifecta Trader — Engine | `81c43bcd-8a52-4544-934b-aa81f9107253` | Stock trader engine |
| Trifecta Trader — Platform | `68afc7ea-4c9b-49e7-8550-711c7a171b48` | tt-curser React app |
| Trifecta Ecosystem | `4bc0fdf8-3fee-4063-9d7e-3f31d4912e87` | Cross-repo coordination |

All under team **TRI (Trifecta)**.

---

---

## 9. Document Management

### Source of truth locations

| Document | Canonical location | Copies |
|----------|-------------------|--------|
| `ECOSYSTEM_CONTEXT.md` | `AI/` folder | Mirrored to all 4 repo roots. Never edit a repo copy — update `AI/` and sync outward. |
| `TRIFECTA_ECOSYSTEM_AGENT.md` | `AI/` folder only | No repo copies. This is a Cowork operating prompt; Claude Code sessions don't need it. |
| Development prompts (M7, Live Trading, etc.) | Each repo's root | Also copied to `AI/` for Ecosystem agent access. Repo copy is canonical. |
| Task specs (UI_TASK_NNN.md) | `tt-curser/docs/` | Repo copy is canonical. `AI/` may have renamed copies for cross-session access. |

### Sync protocol

When updating `ECOSYSTEM_CONTEXT.md`:
1. Edit the `AI/` copy
2. Copy to all 4 repo roots: `trifecta-market-scanner/`, `trifecta-crypto-trader/`, `trifecta-trader-poc/`, `tt-curser/`
3. Never hand-edit a repo copy — it will be overwritten on next sync

---

## 10. Dev & Review Protocol (DEVELOP → QA → UAT)

All development runs through a three-stage gate. **DEVELOP** = Claude Code (tests + code); **QA** = Codex (independent adversarial review); **UAT** = Cursor (runs the user-facing UI test scripts against the running app). The **App Manager** for the affected app writes the test spec and the UAT script and runs the gate; the **Ecosystem agent / Arbiter** owns cross-app contracts, adjudicates, runs what it can to break ties, and signs off after re-verifying. (Where no dedicated App Manager exists yet, the Ecosystem agent acts as the manager.) The full coordination version lives in `AI/TRIFECTA_ECOSYSTEM_AGENT.md` §10, outside the repos — this is the dev-facing summary.

**Roles**
- **Claude Code (DEVELOP)** — writes tests and application code. Never marks its own work Done.
- **Codex (QA)** — independent **adversarial auditor** (far more than a test reviewer): red-teams test specs, audits the whole codebase (build, lint, dead code), audits cross-service contracts (this doc ↔ DB schema ↔ routes ↔ frontend clients), fault-injects safety controls to prove fail-closed, flags security/dependency hygiene, and reviews completed work. Gives `file:line` evidence, severity, and fix suggestions — **including code snippets/blocks** for substandard code — but writes **no committed application code**. Reports to `docs/`.
- **Cursor (UAT)** — runs the user-facing UI test scripts (`docs/TASK_NNN_UAT.md`) against the running app; reports acceptance pass/fail per scenario from the end-user's perspective → `docs/TASK_NNN_UAT_RESULT.md`. UI-testing only; writes no app code.
- **UAT for HEADLESS components (the engines + Scanner have no UI):** the UAT stage is **behavioral / paper-smoke acceptance**, not a UI walkthrough — run the real pipeline on paper (`--dry-run`) and confirm the actual outputs (signals / decisions / files / DB rows), recorded in `docs/TASK_NNN_UAT_RESULT.md` by the App Manager (Cursor optional). This is an **official provision of the protocol, not drift**: headless components *substitute* behavioral UAT for UI UAT — they do not skip the stage.
- **App Manager** (per app) — owns its app's test specs + UAT scripts, runs the gate, verifies its sub-agents' work (trusts no report), and assembles a verified sign-off package for the Arbiter. Does not declare Done.
- **Ecosystem agent / Arbiter** — owns cross-app contracts, adjudication, and final sign-off after independently re-verifying. (Acts as App Manager where none exists yet.)

**Per-task loop**
1. The App Manager (or Ecosystem agent for cross-app work) writes the **test spec** (units) **and the UAT script** (user-facing acceptance scenarios).
2. Claude Code writes the **failing tests first** — they must run **RED**.
3. Codex (QA) reviews the tests **independently** → `docs/TASK_NNN_TEST_REVIEW.md`.
4. Claude Code implements to green, runs the **full suite + build + lint** → `docs/TASK_NNN_REPORT.md`.
5. Codex (QA) reviews the implementation **independently** → `docs/TASK_NNN_CODEX_REVIEW.md`.
6. Cursor (UAT) runs the UI scripts against the running app → `docs/TASK_NNN_UAT_RESULT.md`.
7. Ecosystem agent reconciles → Done only when **DEVELOP + QA + UAT all pass**. Gaps → back to step 4 (or step 1 if the spec/script was wrong). Cap **2 rounds per stage**, then escalate to Jeff.

**Phase 0 (before a component's first sprint):** both Claude Code and Codex independently audit the existing test suite for stale/tautological/false-green tests; the Ecosystem agent reconciles and signs off. Reviewers get a **blind** first pass — the canary is held by the Ecosystem agent, not named in the prompt.

**Hard rules**
- **Safety-first with real money (overriding):** this is real capital. Default to the safest option without being asked — fail closed, block on uncertainty, paper unless live is explicitly confirmed, the more conservative reading of any ambiguity. Never trade safety for convenience, speed, or availability without Jeff's explicit say-so.
- **Independence:** Codex must not read Claude Code's report before forming its own. No anchoring.
- **Done means:** full suite green **AND** build/typecheck green **AND** lint green **AND** a passing Codex QA review **AND** a passing UAT run (Cursor UI UAT for UI components; **behavioral/paper-smoke UAT for headless components**). All three stages (DEVELOP/QA/UAT) must pass. Passing unit tests over code that doesn't compile, lint, or actually run is **not** Done.
- **A commit subject line is not evidence** — verify completion against the actual tree (files exist, behavior present).
- **Safety-critical paths fail closed** unless proven otherwise; prove each safety test actually fails when its guarded behavior is broken (mutation check).
- **Validate cross-service contracts** (this doc ↔ DB schema ↔ backend routes ↔ frontend clients), not just units. Shape/field/type mismatches are real bugs even when every unit test is green.
- **Reports live in repo `docs/`**; the Linear issue holds status + a link.

---

*This document should be updated whenever integration points change. Each development agent should read this before starting work on any sprint.*
