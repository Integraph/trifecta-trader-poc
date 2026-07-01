# Trifecta Trading Platform — Project Brief

**Last updated:** June 30, 2026
**Version:** v1.0.2 (Task 020 complete)
**Owner:** Jeff (jeff@integraphpro.com)
**Purpose:** Single source of truth for any agent working on this project. Read this FIRST.

> **June 30 pivot:** the project is now **engine-first MVP** and runs through the **DEVELOP → QA → UAT** gate (see "How Development Works" below; canonical protocol in `ECOSYSTEM_CONTEXT.md` §10). The old "Cursor does all coding / next = Task 021" model is **superseded**. Linear `TRI-xx` is the tracking system. Architecture map below is current; correct any workflow text you find that still describes the old model.

---

## What This Is

Trifecta is a **multi-agent AI stock and crypto trading platform**. It scans markets for opportunities using technical analysis, runs those opportunities through AI-powered analysis agents (using multiple LLM providers), and executes paper trades via broker APIs. It is a **proof of concept** progressing toward production readiness.

The system has five components. For full integration architecture, see **`ECOSYSTEM_CONTEXT.md`** in this repo's root.

---

## Components at a Glance

| Component | Repo | What It Does | Status |
|-----------|------|-------------|--------|
| **Market Scanner** | `trifecta-market-scanner` | Rule-based scanner using TA-Lib indicators. Produces signal candidates. Zero LLMs. | S7 complete (511 tests) |
| **Stock Trader Engine** | `trifecta-trader-poc` (this repo) | AI analysis pipeline. Consumes scanner signals, runs multi-agent LLM analysis, executes stock trades via Alpaca. | v1.0.2 (20 tasks complete) |
| **Stock Trader UI** | `tt-curser` | React + Fastify frontend for stock trading. Auth, portfolio, backtesting, real-time prices. | Active dev (Phase 5, UI Tasks 1–5) |
| **Crypto Trader Engine** | `trifecta-crypto-trader` | LangGraph-based crypto analysis + Kraken execution. 8-layer safety gates. | M6 complete, M7 in progress |
| **Crypto Trader UI** | TBD | Not started. Technology decision deferred. | Not started |

**Shared infrastructure:** Supabase (auth + data), JSON file queue (Scanner → Traders), YAML configs with atomic writes.

---

## This Repo: Stock Trader Engine (trifecta-trader-poc)

### Architecture

```
Scanner signals (queue/pending/*.json)
    ↓
Queue Reader (src/automation/queue_reader.py) — polls for new candidates
    ↓
Analysis Pipeline (src/run_analysis.py)
    ├── TradingAgentsGraph (vendor/TradingAgents) — multi-agent LLM analysis
    ├── Hybrid LLM Router (src/hybrid_llm.py) — routes agents to cloud/local models
    ├── Enhanced LLM (src/enhanced_llm.py) — prompt enhancement layer
    └── Quality Scorer (src/quality_scorer.py) — scores analysis quality
    ↓
Signal Processing (src/signal_processing.py) — extracts BUY/SELL/HOLD decision
    ↓
Trade Execution (src/execution/) — Alpaca paper trading
    ↓
Publishing (src/integration/supabase_writer.py) — results to Supabase
    ↓
Accuracy Tracking (src/accuracy/) — tracks T+1/T+5/T+10 price movements
```

### Key Subsystems

**Hybrid LLM System:** The core differentiator. Each analysis uses three LLM "slots": tool_calling (needs function-calling support), reasoning_quick, and reasoning_deep (Risk Judge). Each slot can be routed to a different provider/model. Configs are stored in `config/hybrid_llm.yaml` and managed via the admin UI.

**Admin Dashboard:** FastAPI backend (`src/admin/`) + React frontend (`admin-ui/`). Provides config management, scheduler control, queue monitoring, analysis browsing, accuracy reports, health checks, test runs, and A/B LLM comparison.

**Automation Pipeline:** APScheduler-based daemon (`src/automation/daemon.py`) that runs scheduled watchlist scans, queue reading, and accuracy updates. Controlled via `config/automation.yaml`.

### Tech Stack

- Python 3.11+, FastAPI, LangGraph/LangChain
- React 18, TypeScript, Vite (admin UI)
- Anthropic Claude (Sonnet/Haiku), Ollama (Qwen, Mistral), OpenAI, Google, XAI, OpenRouter
- Alpaca (stock broker API), Supabase (database), SQLite (portfolio tracker)
- APScheduler (automation), pytest (testing)

---

## Development History — Task Log

Each task has a spec (`docs/CURSOR_TASK_NNN_*.md`) and a report (`docs/TASK_NNN_REPORT.md`).

| Task | Title | Type | What It Delivered |
|------|-------|------|-------------------|
| 001 | Repository Setup | Setup | Project structure, dependencies, vendor submodule |
| 002 | Signal Processing Bug Fix | Bug Fix | Fixed signal extraction loop issue |
| 003 | Hybrid LLM Experiment | Feature | Multi-provider LLM routing (cloud + local models) |
| 004 | Live Hybrid Validation | Feature | Validation framework for hybrid configs |
| 005 | Local Model Scaling | Feature | Benchmarks for different local model sizes |
| 006 | Prompt Engineering + Multi-Ticker | Feature | Improved local model quality, multi-ticker validation |
| 007 | Position Management & Execution | Feature | Trade execution layer, Alpaca integration |
| 008 | Structured Execution Output | Feature | Standardized output format, first paper trade |
| 009 | Parameter Extraction Fix | Bug Fix | Fixed dual-source trade parameter parsing |
| 010 | Cost Optimization | Feature | Haiku tier routing, response caching, cost tracking |
| 011 | Qwen 3.5 Benchmark | Feature | Benchmarked Qwen 3.5 local models (9b/27b/35b) |
| 012 | Portfolio-Aware Execution | Feature | Portfolio tracking, watchlist batch mode |
| 013 | Signal Adapter & Supabase | Feature | Supabase writer, signal adapter layer |
| 014 | Pipeline Automation | Feature | Scheduler, queue reader, daemon mode |
| 015 | Signal Accuracy Tracker | Feature | T+1/T+5/T+10 accuracy tracking and scoring |
| 016 | Admin API | Feature | FastAPI admin backend (health, config, tasks, logs, events) |
| 017 | Admin Dashboard Frontend | Feature | React admin UI (all pages, WebSocket, data tables) |
| 018 | LLM Config Editor & Tooltips | Feature | YAML config externalization, CRUD API, sanity check, A/B comparison, info tooltips |
| 019 | Dynamic CLI Config Choices | Bug Fix | Fixed hardcoded argparse choices to read from YAML dynamically |
| 020 | Admin UI Bug Fixes | Bug Fix | 6 bugs: publish keyword, detail panel refresh, field name mismatches, health status logic, dependency checks |
| 021 | UI Polish & Accessibility | Bug Fix | **DEFERRED** (TRI-30) — engine's admin-UI polish, parked under the engine-first MVP. Do not run. |

---

## Current State (as of v1.0.2)

**What works:**
- Full analysis pipeline: scanner signal → AI analysis → trade execution → Supabase publishing
- 15 hybrid LLM configurations with CRUD management via admin UI
- Automated daemon with scheduler, queue reader, and accuracy updater
- Admin dashboard with all pages functional (health, config, scheduler, queue, analyses, accuracy, test runs, logs)
- A/B LLM comparison for side-by-side config testing
- 44+ task-specific tests + existing admin/accuracy/daemon tests
- Clean TypeScript build (0 errors)

**Engine-first MVP priorities (June 30):**
- **TRI-66 (Urgent)** — upgrade vendored TradingAgents v0.2.0 → v0.3.0; reconcile the 4-file local mod; restore zero-mod; update model strings (deep Risk-Judge = `claude-opus-4-8`, tools = `claude-haiku-4-5`); pin LangGraph. **First MVP action.**
- **TRI-70** — Step 0: re-benchmark current local models on the M3 Max (blocked by TRI-66; feeds TRI-69).
- **TRI-69** — define the "prove-it-before-real-money" out-of-sample edge gate (Ecosystem).
- **TRI-32** — push the 13 local-only commits to origin.
- Deferred: live trading (TRI-31), admin-UI polish / Task 021 (TRI-30). Separate: LangGraph 1.x (TRI-67), lockfiles (TRI-65).

**Known correction:** the vendor submodule is **NOT zero-mod** — pin `5de91bc` carries our Task-020 commit touching 4 files; TRI-66 restores it.

**Not in scope for this repo:**
- Crypto trading (separate repo: `trifecta-crypto-trader`)
- Stock Trader UI (separate repo: `tt-curser`)
- Unified admin dashboard (Ecosystem agent scope, future)

---

## How Development Works

### Roles (DEVELOP → QA → UAT gate — canonical in `ECOSYSTEM_CONTEXT.md` §10)

- **Jeff** — Product owner. Sets goals/priorities, makes irreversible & financial/regulatory calls, final arbiter.
- **Engine App Manager (Cowork/Claude)** — Owns this app end to end: writes specs + runtime acceptance checks, runs the gate, **verifies every stage against the tree (trusts no report)**, assembles a verified sign-off for the Arbiter. Does **not** write committed application code, and does **not** declare Done.
- **Ecosystem Arbiter** — Owns cross-app contracts; independently **re-verifies** the App Manager's work and signs off Done.
- **DEVELOP = Claude Code** — writes failing tests first (RED), implements to green, runs full `pytest` + a paper smoke (`run_batch --dry-run`) → `docs/TASK_*_REPORT.md`.
- **QA = Codex** — independent adversarial review; assumes the report is wrong until proven; gives `file:line` evidence and fix snippets but commits no app code → `docs/TASK_*_CODEX_REVIEW.md`.
- **UAT = runtime smoke** — the engine is **headless**, so UAT is **behavioral acceptance** (pipeline runs on paper, signals produced, decisions parseable on both a hybrid and a fully-local config), **not** a UI walkthrough → `docs/TASK_*_UAT_RESULT.md`.

### Workflow

1. The App Manager writes the spec **and** the runtime acceptance check, grounded in the actual code (verify mechanisms first).
2. **DEVELOP** (Claude Code) writes failing tests → implements to green → full suite + paper smoke → report.
3. **QA** (Codex) reviews tests, then implementation, independently — blind to Claude Code's report.
4. **UAT** runs the pipeline on paper and confirms real behavior.
5. The App Manager verifies every stage against the tree, then hands an evidence sign-off to the Arbiter, who **re-verifies and signs off Done**. **Done = DEVELOP green AND Codex APPROVED AND UAT passed AND Arbiter re-verified.** Cap 2 rounds/stage, then escalate to Jeff.

### Task Spec Format

Every task spec follows this structure:
- **Objective** — what and why
- **Background** — context the agent needs
- **Parts** — numbered sections with detailed requirements
- **Deliverables** — table of expected outputs
- **Exit Criteria** — numbered, testable acceptance criteria
- **Implementation Notes** — technical guidance and constraints
- **File Inventory** — new and modified files

### Where Files Go

- Task specs: `docs/CURSOR_TASK_NNN_*.md` + copy to `/mnt/AI/Cursor/`
- Task reports: `docs/TASK_NNN_REPORT.md`
- Task questions: `docs/TASK_NNN_QUESTIONS.md`
- Question responses: `docs/TASK_NNN_Q1_RESPONSE.md`

---

## Critical Instructions for Any Agent

### Before Starting Work

1. **Read this file first.** It tells you what the project is and where things stand.
2. **Read `ECOSYSTEM_CONTEXT.md`** for integration architecture across all repos.
3. **Read the last 2-3 task reports** to understand recent changes and current state.
4. **Tell the user what you understand** about the project before starting any work. If you can't accurately describe the system, you don't have enough context.

### During Work

5. **If you lose track of the overall system or how your current task fits into it, STOP and tell the user immediately.** Do not proceed on assumptions. This is a hard requirement.
6. **Do not try to be both the manager and the domain expert.** If you're reviewing work, review it. If you're writing specs, write them. Don't also try to hold the entire AI trading pipeline in your head.
7. **Update this file** when tasks are completed (add to the task log, update "Current State" section).

### Context Preservation

8. **This file + ECOSYSTEM_CONTEXT.md + the last task report = minimum viable context** for any new session. If any of these are missing or stale, flag it before starting work.
9. **Each chat session starts from zero.** The agent in the next window has never seen this conversation. Everything it needs must be in these documents.
10. **When in doubt, read the task spec.** Jeff's task specs are detailed and specific. The spec is the source of truth for what was requested. The report is the source of truth for what was delivered.
