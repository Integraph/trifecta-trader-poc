// Help text for every setting on the Configuration page.
// Displayed via InfoTooltip components next to each field label.

export const SETTINGS_HELP: Record<string, string> = {
  // ── Scheduler ─────────────────────────────────────────────────────────────
  'scheduler.enabled':
    'Master switch for the daily watchlist scanner. When disabled, no scheduled scans will run. The queue reader and accuracy updater operate independently.',

  'scheduler.watchlist_hour':
    'Hour of day (24h format) in the configured timezone when the watchlist scan runs. Default: 8 (8 AM). Paired with Minute below. Requires daemon restart.',

  'scheduler.watchlist_minute':
    'Minute of the hour when the watchlist scan runs. Default: 30 (giving 8:30 AM ET). ~60 minutes before market open allows analyses to complete before trading begins. Requires daemon restart.',

  'scheduler.hybrid_config':
    'Which LLM configuration to use for scheduled analyses. Determines which AI models handle tool-calling, quick reasoning, and deep reasoning (Risk Judge). Edit configs in the LLM Configuration section below.',

  'scheduler.watchlist':
    'Name of the watchlist file to scan (from config/watchlists/). Each watchlist contains ticker symbols to analyze. Requires daemon restart.',

  'scheduler.publish':
    'When enabled, analysis results are automatically published to Supabase for the Platform UI. Disable for testing or when Supabase is not configured.',

  // ── Queue Reader ──────────────────────────────────────────────────────────
  'queue_reader.enabled':
    'Master switch for the file-based queue reader. When enabled, the daemon polls the queue directory for candidate JSON files from the Market Scanner.',

  'queue_reader.poll_interval':
    'How often (in seconds) to check for new queue candidates. Lower values mean faster processing but more filesystem I/O. Valid range: 5–300. Takes effect immediately.',

  'queue_reader.max_retries':
    'Maximum number of retry attempts for a failed analysis before marking the candidate as permanently failed. Valid range: 0–10. Takes effect immediately.',

  'queue_reader.cooldown':
    'Minimum wait time (in seconds) between consecutive analyses to respect LLM rate limits and avoid thundering herd. Valid range: 10–600. Takes effect immediately.',

  // ── Accuracy ──────────────────────────────────────────────────────────────
  'accuracy.enabled':
    'Master switch for the signal accuracy tracker. When enabled, tracks price movements at T+1, T+5, and T+10 trading days after each signal to evaluate prediction quality.',

  'accuracy.backfill':
    'When enabled, automatically scores all existing untracked analyses when the daemon first starts. Useful after adding accuracy tracking to an existing deployment.',

  // ── Admin API ─────────────────────────────────────────────────────────────
  'admin_api.port':
    'TCP port for the Admin API server. Default: 8420. Changing this requires a daemon restart and updating the frontend API_BASE URL.',

  // ── Supabase ──────────────────────────────────────────────────────────────
  'supabase.write_enabled':
    'When enabled, analyses are published to the Supabase signals table. Disable to run analyses without publishing (e.g., during testing). Takes effect immediately.',

  'supabase.signal_ttl':
    'How many hours a published signal remains active in Supabase before being considered stale. The Platform UI uses this to filter out old signals. Valid range: 1–168 (1 week).',

  // ── LLM Configurations (section-level) ────────────────────────────────────
  'llm_configs.section':
    'Hybrid LLM configurations define which AI providers and models handle each agent role.\n\nTool-calling agents require models that support function calling (e.g., Anthropic Claude, OpenAI GPT-4). Reasoning agents can use any model including local Ollama models.\n\nEdit → Sanity Check → Save to deploy a new config.',

  // ── Watchlists (section-level) ────────────────────────────────────────────
  'watchlists.section':
    'Watchlists define which tickers the scheduler analyzes. Each watchlist is a YAML file in config/watchlists/. The active watchlist is set in the Scheduler config above.\n\nClick a watchlist to edit its ticker list. Changes take effect on the next scheduled run.',
};
