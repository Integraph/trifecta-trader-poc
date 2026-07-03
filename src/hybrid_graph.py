"""
Patched TradingAgentsGraph that supports per-agent LLM routing.

This subclass overrides the LLM assignment so that:
- Tool-calling agents (analysts) get one LLM (typically cloud)
- Quick reasoning agents get another LLM (can be local)
- Deep reasoning agents (judges) get a third LLM (typically cloud)

Vendor code copied: The setup_graph() method from
vendor/TradingAgents/tradingagents/graph/setup.py is replicated here
with a three-LLM split instead of the original two-LLM split.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, StateGraph, START

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.agents import (
    create_msg_delete,
    AgentState,
    create_market_analyst,
    create_social_media_analyst,
    create_news_analyst,
    create_fundamentals_analyst,
    create_bull_researcher,
    create_bear_researcher,
    create_research_manager,
    create_trader,
    create_aggressive_debator,
    create_conservative_debator,
    create_neutral_debator,
    create_portfolio_manager,
)
from src.hybrid_llm import HybridLLMConfig, create_hybrid_llms
from src.data_cache import DataCache, ANALYST_TTL

logger = logging.getLogger(__name__)

# Maps analyst type → AgentState field name
ANALYST_STATE_FIELD = {
    "market":       "market_report",
    "social":       "sentiment_report",
    "news":         "news_report",
    "fundamentals": "fundamentals_report",
}

# Anthropic pricing (USD per 1M tokens). Cloud is BENCHMARK-REFERENCE only in
# TRI-70 (never a shipped/production model — too expensive per Jeff). Local /
# Ollama models cost $0 (see the provider-aware fallback in cost_breakdown()).
# TRI-70 pricing triad (Jeff, 2026-07-02):
#   - claude-opus-4-8 row added at $5/$25 (the chosen expensive deep reference)
#   - Haiku refreshed $0.80/$4.00 -> $1.00/$5.00
#   - _normalize_model "opus" now maps to claude-opus-4-8 (was the retired
#     opus-4-5 at $15/$75, so the new row would never be reached)
MODEL_PRICING = {
    "claude-sonnet-4-5-20250929": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":  {"input": 1.00,  "output": 5.00},
    "claude-opus-4-8":            {"input": 5.00,  "output": 25.00},
    # Retired (kept for reference; no longer reached via _normalize_model):
    "claude-opus-4-5-20251101":   {"input": 15.00, "output": 75.00},
}


class TokenUsageCallback(BaseCallbackHandler):
    """LangChain callback handler that tracks token usage per LLM call.

    Accumulates usage_metadata from each LLM response so we can compute
    cost breakdowns after the pipeline completes.
    """

    def __init__(self):
        super().__init__()
        self.usage_by_model: Dict[str, Dict] = {}
        self.call_count = 0

    def on_llm_end(self, response, **kwargs):
        """Called after each LLM API call completes."""
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen.message, "usage_metadata", None) or {}
                    model = getattr(gen.message, "response_metadata", {}).get(
                        "model", "unknown"
                    )
                    # Normalize model name (strip version suffix variations)
                    model_key = self._normalize_model(model)

                    if model_key not in self.usage_by_model:
                        self.usage_by_model[model_key] = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "calls": 0,
                        }

                    self.usage_by_model[model_key]["input_tokens"] += meta.get(
                        "input_tokens", 0
                    )
                    self.usage_by_model[model_key]["output_tokens"] += meta.get(
                        "output_tokens", 0
                    )
                    self.usage_by_model[model_key]["calls"] += 1
                    self.call_count += 1
        except Exception:
            pass  # Never let callback errors break the pipeline

    def cost_breakdown(self, cache_stats: dict = None) -> dict:
        """Compute cost breakdown from tracked token usage.

        Returns:
            Dict with per-model costs and totals.
        """
        breakdown = {"by_model": {}, "total_usd": 0.0}

        for model_key, usage in self.usage_by_model.items():
            # Provider-aware: a model_key not in MODEL_PRICING is a local/Ollama
            # model, which is free — cost $0. (Was a Sonnet $3/$15 fallback,
            # which wrongly billed local runs as if they were cloud; TRI-70.)
            pricing = MODEL_PRICING.get(model_key, {"input": 0.0, "output": 0.0})
            input_cost = usage["input_tokens"] / 1_000_000 * pricing["input"]
            output_cost = usage["output_tokens"] / 1_000_000 * pricing["output"]
            total_cost = input_cost + output_cost

            breakdown["by_model"][model_key] = {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "calls": usage["calls"],
                "cost_usd": round(total_cost, 6),
            }
            breakdown["total_usd"] += total_cost

        breakdown["total_usd"] = round(breakdown["total_usd"], 6)

        if cache_stats:
            breakdown["cache_hits"] = cache_stats.get("hits", 0)
            breakdown["cache_misses"] = cache_stats.get("misses", 0)
            breakdown["cache_hit_rate_pct"] = cache_stats.get("hit_rate_pct", 0.0)

        return breakdown

    @staticmethod
    def _normalize_model(model: str) -> str:
        """Map raw model name to a canonical key."""
        model_lower = model.lower()
        if "haiku" in model_lower:
            return "claude-haiku-4-5-20251001"
        if "sonnet" in model_lower:
            return "claude-sonnet-4-5-20250929"
        if "opus" in model_lower:
            return "claude-opus-4-8"
        return model


def make_cached_analyst(
    analyst_creator_fn,
    analyst_type: str,
    llm,
    cache: DataCache,
    ticker: str,
    trade_date: str,
):
    """Return a cache-aware analyst node function.

    On cache HIT:  injects cached report into state without LLM call.
    On cache MISS: runs analyst normally and caches the final report.

    The cache-hit path returns an AIMessage with no tool_calls, which
    causes ConditionalLogic.should_continue_<type> to route directly to
    the Msg Clear node (skipping the tool-call loop entirely).
    """
    state_field = ANALYST_STATE_FIELD[analyst_type]
    cache_key = cache.key_for(ticker, analyst_type, trade_date)
    cached_value = cache.get(cache_key)

    if cached_value:
        logger.info("Analyst cache HIT: %s — skipping LLM call", cache_key)

        def cached_node(state: dict) -> dict:
            return {
                state_field: cached_value,
                "messages": [AIMessage(content=cached_value)],
            }

        return cached_node

    # Cache miss — create real analyst and wrap to cache on completion
    real_node = analyst_creator_fn(llm)
    ttl = ANALYST_TTL.get(analyst_type, 3600)

    def caching_node(state: dict) -> dict:
        result = real_node(state)
        report = result.get(state_field, "")
        if report:
            cache.set(cache_key, report, ttl=ttl)
        return result

    return caching_node


class HybridGraphSetup:
    """Graph setup that accepts three LLM instances for hybrid routing.

    This replicates the graph wiring from GraphSetup.setup_graph() but uses
    three LLMs instead of two:
    - tool_llm: for analyst nodes that call external tools
    - reasoning_quick_llm: for debaters, researchers, and trader
    - reasoning_deep_llm: for judge nodes (Research Manager, Risk Manager)

    Optionally accepts a DataCache and ticker/date to enable analyst-level
    output caching — skipping LLM calls for analysts whose outputs are cached.
    """

    def __init__(
        self,
        tool_llm,
        reasoning_quick_llm,
        reasoning_deep_llm,
        tool_nodes: Dict,
        conditional_logic: ConditionalLogic,
        cache: Optional[DataCache] = None,
        ticker: str = "",
        trade_date: str = "",
    ):
        self.tool_llm = tool_llm
        self.reasoning_quick_llm = reasoning_quick_llm
        self.reasoning_deep_llm = reasoning_deep_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.cache = cache
        self.ticker = ticker
        self.trade_date = trade_date

    def setup_graph(self, selected_analysts=None):
        """Set up and compile the agent workflow graph with three-LLM routing.

        This method mirrors GraphSetup.setup_graph() from
        vendor/TradingAgents/tradingagents/graph/setup.py but assigns LLMs
        based on agent role rather than giving all quick agents the same LLM.

        When a DataCache is provided, each analyst is wrapped with a cache-aware
        node that injects cached reports (skipping the LLM call) on a cache hit.
        """
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]

        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        analyst_creators = {
            "market": create_market_analyst,
            "social": create_social_media_analyst,
            "news": create_news_analyst,
            "fundamentals": create_fundamentals_analyst,
        }

        # v0.3.0 made analyst node naming SPEC-DRIVEN. The "social" key now maps to
        # agent_node="Sentiment Analyst" / clear_node="Msg Clear Sentiment", and
        # conditional_logic.should_continue_social() returns those spec labels. So we
        # must build nodes and edges from the execution plan — a naive
        # f"{key.capitalize()} Analyst" pattern produces "Msg Clear Social", which the
        # v0.3.0 router can't find (KeyError 'Msg Clear Sentiment').
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for spec in plan.specs:
            # Optionally wrap analyst with cache-aware node (hybrid optimization)
            if self.cache and self.ticker and self.trade_date:
                analyst_nodes[spec.key] = make_cached_analyst(
                    analyst_creator_fn=analyst_creators[spec.key],
                    analyst_type=spec.key,
                    llm=self.tool_llm,
                    cache=self.cache,
                    ticker=self.ticker,
                    trade_date=self.trade_date,
                )
            else:
                analyst_nodes[spec.key] = analyst_creators[spec.key](self.tool_llm)
            delete_nodes[spec.key] = create_msg_delete()
            tool_nodes[spec.key] = self.tool_nodes[spec.key]

        # Researchers and trader use reasoning_quick_llm.
        # v0.3.0: agent factories no longer take a per-agent memory object —
        # memory is centralized in the vendor's file-based TradingMemoryLog and
        # injected into state as past_context by TradingAgentsGraph.propagate()
        # (which our propagate() delegates to). See TASK_TRI-66_REPORT.md.
        bull_researcher_node = create_bull_researcher(self.reasoning_quick_llm)
        bear_researcher_node = create_bear_researcher(self.reasoning_quick_llm)
        trader_node = create_trader(self.reasoning_quick_llm)

        # Judges use reasoning_deep_llm
        research_manager_node = create_research_manager(self.reasoning_deep_llm)

        # Risk debaters use reasoning_quick_llm
        aggressive_analyst = create_aggressive_debator(self.reasoning_quick_llm)
        neutral_analyst = create_neutral_debator(self.reasoning_quick_llm)
        conservative_analyst = create_conservative_debator(self.reasoning_quick_llm)

        # Risk judge uses reasoning_deep_llm.
        # v0.3.0 removed create_risk_manager(llm, memory) -> create_portfolio_manager(llm),
        # and the final node is labeled "Portfolio Manager" (should_continue_risk_analysis
        # routes here, not to the old "Risk Judge" label).
        portfolio_manager_node = create_portfolio_manager(self.reasoning_deep_llm)

        workflow = StateGraph(AgentState)

        # Analyst nodes — use the v0.3.0 spec labels (agent_node / clear_node / tool_node)
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_nodes[spec.key])
            workflow.add_node(spec.clear_node, delete_nodes[spec.key])
            workflow.add_node(spec.tool_node, tool_nodes[spec.key])

        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        workflow.add_edge(START, plan.specs[0].agent_node)

        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            if i < len(plan.specs) - 1:
                workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Conservative Analyst": "Conservative Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Neutral Analyst": "Neutral Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Aggressive Analyst": "Aggressive Analyst", "Portfolio Manager": "Portfolio Manager"},
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow.compile()


class HybridTradingGraph(TradingAgentsGraph):
    """TradingAgentsGraph with per-agent LLM provider routing.

    Adds two capabilities over the base graph:
    1. Three-LLM routing: tool-calling, quick-reasoning, deep-reasoning agents
       each get their own LLM (enabling cost-optimised hybrid configs).
    2. Analyst output caching: via DataCache, analyst reports are cached by
       ticker+date with TTLs. On cache hit, the LLM call is skipped entirely.
    3. Token usage tracking: TokenUsageCallback accumulates per-model token
       counts so run_analysis can report a cost breakdown.

    Usage:
        from src.hybrid_llm import CONFIGS
        from src.hybrid_graph import HybridTradingGraph

        graph = HybridTradingGraph(
            hybrid_config=CONFIGS["hybrid_haiku_tools"],
            config={...},
        )
        final_state, decision = graph.propagate("AAPL", "2026-03-02")
        breakdown = graph.cost_breakdown()   # dict with token counts and USD
    """

    def __init__(
        self,
        hybrid_config: HybridLLMConfig,
        selected_analysts=None,
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
        cache_dir: str = "cache",
        use_cache: bool = True,
    ):
        self._hybrid_config = hybrid_config
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]
        self._selected_analysts = selected_analysts
        self._use_cache = use_cache

        # DataCache — created now, wired into graph in propagate()
        self._cache = DataCache(cache_dir=cache_dir) if use_cache else None

        # Token usage tracker — attached as a LangChain callback
        self._token_callback = TokenUsageCallback()

        # Let parent __init__ do all the standard setup
        super().__init__(
            selected_analysts=selected_analysts,
            debug=debug,
            config=config,
            callbacks=callbacks,
        )

        # Rebuild graph with hybrid LLM routing (no ticker/date yet — set in propagate())
        self._llms = create_hybrid_llms(self._hybrid_config)
        self._rebuild_graph(ticker="", trade_date="")

        logger.info(
            "HybridTradingGraph initialized: %s", self._hybrid_config.to_dict()
        )

    def _rebuild_graph(self, ticker: str, trade_date: str) -> None:
        """(Re)build the compiled graph, optionally wiring in the cache."""
        self.graph_setup = HybridGraphSetup(
            tool_llm=self._llms["tool_calling_llm"],
            reasoning_quick_llm=self._llms["reasoning_quick_llm"],
            reasoning_deep_llm=self._llms["reasoning_deep_llm"],
            tool_nodes=self.tool_nodes,
            conditional_logic=self.conditional_logic,
            cache=self._cache,
            ticker=ticker,
            trade_date=trade_date,
        )
        self.graph = self.graph_setup.setup_graph(self._selected_analysts)

    def propagate(self, company_name: str, trade_date: str):
        """Run the trading pipeline, with caching and token tracking.

        Overrides TradingAgentsGraph.propagate() to:
        1. Rebuild the compiled graph with the current ticker/date so the
           cache-aware analyst wrappers have the right keys.
        2. Inject the TokenUsageCallback so every LLM call is tracked.
        3. Delegate to the parent propagate() for actual graph execution.
        """
        # Rebuild graph with ticker+date so cache keys are correct
        self._rebuild_graph(ticker=company_name, trade_date=str(trade_date))

        # Attach our callback to all LLMs
        for llm_key, llm_obj in self._llms.items():
            try:
                llm_obj.callbacks = [self._token_callback]
            except Exception:
                pass

        result = super().propagate(company_name, trade_date)
        return result

    def cost_breakdown(self) -> dict:
        """Return cost breakdown from the last (or current) pipeline run."""
        cache_stats = self._cache.stats() if self._cache else {}
        return self._token_callback.cost_breakdown(cache_stats=cache_stats)
