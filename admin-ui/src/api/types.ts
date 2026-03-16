// ── Health ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  color: 'green' | 'yellow' | 'red';
  timestamp: string;
  uptime_seconds: number | null;
  pid: number | null;
  subsystems: {
    daemon: {
      status: string;
      pid: number | null;
      start_time: string | null;
      uptime_seconds: number | null;
    };
    scheduler: {
      status: string;
      last_run: string | null;
      last_run_result: string | null;
      next_run: string | null;
      enabled: boolean;
    };
    queue_reader: {
      status: string;
      pending_count: number;
      processing_count: number;
      completed_today: number;
      last_poll: string | null;
    };
    accuracy_updater: {
      status: string;
      pending_outcomes: number;
      complete_outcomes: number;
      last_run: string | null;
    };
    supabase: {
      configured: boolean;
      write_enabled: boolean;
    };
    ollama: {
      reachable: boolean;
      model: string | null;
      latency_ms: number | null;
    };
  };
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

export interface LastRunDetail {
  timestamp: string;
  result: 'success' | 'error';
  tickers_processed: number;
  elapsed_seconds: number;
  decisions: Record<string, number>;
  error?: string | null;
}

export interface SchedulerStatus {
  enabled: boolean;
  is_running: boolean;
  schedule: string;
  next_run: string | null;
  last_run: LastRunDetail | null;
  config: {
    watchlist: string;
    hybrid_config: string;
    publish: boolean;
    hour: number;
    minute: number;
    timezone: string;
    weekdays_only: boolean;
  };
}

export interface SchedulerHistoryItem {
  trade_date: string;
  total_analyses: number;
  decisions: Record<string, number>;
  avg_quality: number | null;
  total_cost_usd: number | null;
  elapsed_seconds: number | null;
}

// ── Queue ─────────────────────────────────────────────────────────────────────

export interface QueueStatus {
  enabled: boolean;
  is_running: boolean;
  counts: {
    pending: number;
    processing: number;
    completed: number;
    failed: number;
  };
  config: {
    queue_dir: string;
    poll_interval_seconds: number;
    max_retries: number;
    cooldown_seconds: number;
  };
  last_poll: string | null;
}

export interface QueueCandidate {
  filename: string;
  ticker: string;
  priority: string;
  score: number | null;
  catalysts: string[];
  source: string | null;
  retry_count: number;
  queued_at: string | null;
}

export interface QueueCompleted {
  filename: string;
  ticker: string;
  decision: string | null;
  quality_score: number | null;
  elapsed_seconds: number | null;
  completed_at: string | null;
  analysis_result: Record<string, unknown> | null;
}

// ── Accuracy ──────────────────────────────────────────────────────────────────

export interface DecisionStats {
  count: number;
  directionally_correct: number;
  direction_accuracy_t1: number | null;
  direction_accuracy_t5: number | null;
  direction_accuracy_t10: number | null;
  avg_return_t1: number | null;
  avg_return_t5: number | null;
  avg_return_t10: number | null;
  target_hit_rate: number | null;
  stop_hit_rate: number | null;
}

export interface TierStats {
  count: number;
  avg_return_t5: number | null;
  direction_accuracy_t5: number | null;
}

export interface SignalRow {
  id: number;
  ticker: string;
  trade_date: string;
  decision: string;
  quality_score: number | null;
  return_t5: number | null;
  return_t10: number | null;
  direction_correct: boolean | null;
}

export interface AccuracySummary {
  period_days: number;
  total_signals: number;
  pending_outcomes: number;
  complete_outcomes: number;
  by_decision: Record<string, DecisionStats>;
  by_quality_tier: Record<string, TierStats>;
  best_signals: SignalRow[];
  worst_signals: SignalRow[];
}

export interface TickerReport {
  ticker: string;
  total: number;
  complete: number;
  pending: number;
  direction_accuracy: Record<string, number | null>;
  signals: SignalRow[];
}

// ── Analyses ──────────────────────────────────────────────────────────────────

export interface AnalysisRow {
  id: number;
  ticker: string;
  trade_date: string;
  decision: string;
  quality_score: number | null;
  entry_price: number | null;
  stop_loss: number | null;
  price_target: number | null;
  total_cost_usd: number | null;
  elapsed_seconds: number | null;
  outcome_status: string | null;
  created_at: string | null;
}

export interface AnalysisDetail extends AnalysisRow {
  hybrid_config: string | null;
  watchlist: string | null;
  raw_result: Record<string, unknown> | null;
  outcome: Record<string, unknown> | null;
}

export interface AnalysesStats {
  total_analyses: number;
  analyses_today: number;
  unique_tickers: number;
  avg_quality_score: number | null;
  total_cost_usd: number | null;
  by_decision: Record<string, number>;
}

// ── Tasks ─────────────────────────────────────────────────────────────────────

export interface TaskStatus {
  task_id: string;
  status: 'running' | 'complete' | 'error';
  started_at: string;
  completed_at: string | null;
  result: unknown;
  error: string | null;
}

// ── Config ────────────────────────────────────────────────────────────────────

export interface AutomationConfig {
  scheduler?: {
    enabled?: boolean;
    hour?: number;
    minute?: number;
    timezone?: string;
    weekdays_only?: boolean;
    hybrid_config?: string;
    watchlist?: string;
    publish?: boolean;
  };
  queue_reader?: {
    enabled?: boolean;
    queue_dir?: string;
    poll_interval_seconds?: number;
    max_retries?: number;
    cooldown_seconds?: number;
    target_trader?: string;
  };
  accuracy?: {
    enabled?: boolean;
    update_hour?: number;
    update_minute?: number;
    backfill_on_first_run?: boolean;
  };
  admin_api?: {
    enabled?: boolean;
    port?: number;
    host?: string;
  };
}

export interface SupabaseConfig {
  url?: string;
  write_enabled?: boolean;
  signal_ttl_hours?: number;
  table_name?: string;
}

export interface WatchlistItem {
  name: string;
  path: string;
  tickers: string[];
}

export interface HybridConfig {
  name: string;
  tool_provider: string | null;
  tool_model: string | null;
  reasoning_quick_provider: string | null;
  reasoning_quick_model: string | null;
  reasoning_deep_provider: string | null;
  reasoning_deep_model: string | null;
}

export interface HybridConfigFull {
  name: string;
  tool_provider: string;
  tool_model: string;
  reasoning_quick_provider: string;
  reasoning_quick_model: string;
  reasoning_deep_provider: string;
  reasoning_deep_model: string;
  enhance_local: boolean;
  enhance_style: string;
  enhance_deep: boolean;
  enhance_deep_style: string;
}

export interface HybridConfigsResponse {
  configs: HybridConfigFull[];
  active: string | null;
  providers: string[];
  enhance_styles: string[];
}

export interface SanityCheckSlot {
  provider: string;
  model: string;
  status: 'pass' | 'fail' | 'skip';
  latency_ms: number | null;
  error: string | null;
}

export interface SanityCheckResult {
  config_name: string;
  overall: 'pass' | 'partial' | 'fail';
  checks: {
    tool_calling: SanityCheckSlot;
    reasoning_quick: SanityCheckSlot;
    reasoning_deep: SanityCheckSlot;
  };
}

export interface ABCompareRequest {
  ticker: string;
  trade_date: string;
  config_a: string;
  config_b: string;
  publish?: boolean;
}

export interface ABSide {
  name: string;
  task_id: string;
  status: 'running' | 'complete' | 'error' | 'unknown';
  result: unknown;
  error: string | null;
}

export interface ABCompareResponse {
  ab_id: string;
  ticker: string;
  trade_date: string;
  status: 'running' | 'complete';
  started_at: string;
  config_a: ABSide;
  config_b: ABSide;
}

// ── Logs ──────────────────────────────────────────────────────────────────────

export interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
}

// ── Events ────────────────────────────────────────────────────────────────────

export interface LiveEvent {
  event: string;
  data: Record<string, unknown>;
  timestamp: string;
}
