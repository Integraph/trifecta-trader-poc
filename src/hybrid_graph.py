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
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph, START

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic
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
    create_risk_manager,
)
from src.hybrid_llm import HybridLLMConfig, create_hybrid_llms

logger = logging.getLogger(__name__)


class HybridGraphSetup:
    """Graph setup that accepts three LLM instances for hybrid routing.

    This replicates the graph wiring from GraphSetup.setup_graph() but uses
    three LLMs instead of two:
    - tool_llm: for analyst nodes that call external tools
    - reasoning_quick_llm: for debaters, researchers, and trader
    - reasoning_deep_llm: for judge nodes (Research Manager, Risk Manager)
    """

    def __init__(
        self,
        tool_llm,
        reasoning_quick_llm,
        reasoning_deep_llm,
        tool_nodes: Dict,
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        self.tool_llm = tool_llm
        self.reasoning_quick_llm = reasoning_quick_llm
        self.reasoning_deep_llm = reasoning_deep_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic

    def setup_graph(self, selected_analysts=None):
        """Set up and compile the agent workflow graph with three-LLM routing.

        This method mirrors GraphSetup.setup_graph() from
        vendor/TradingAgents/tradingagents/graph/setup.py but assigns LLMs
        based on agent role rather than giving all quick agents the same LLM.
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

        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in selected_analysts:
            # Analysts use tool_llm (they call external tools)
            analyst_nodes[analyst_type] = analyst_creators[analyst_type](self.tool_llm)
            delete_nodes[analyst_type] = create_msg_delete()
            tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        # Researchers and trader use reasoning_quick_llm
        bull_researcher_node = create_bull_researcher(
            self.reasoning_quick_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.reasoning_quick_llm, self.bear_memory
        )
        trader_node = create_trader(self.reasoning_quick_llm, self.trader_memory)

        # Judges use reasoning_deep_llm
        research_manager_node = create_research_manager(
            self.reasoning_deep_llm, self.invest_judge_memory
        )

        # Risk debaters use reasoning_quick_llm
        aggressive_analyst = create_aggressive_debator(self.reasoning_quick_llm)
        neutral_analyst = create_neutral_debator(self.reasoning_quick_llm)
        conservative_analyst = create_conservative_debator(self.reasoning_quick_llm)

        # Risk judge uses reasoning_deep_llm
        risk_manager_node = create_risk_manager(
            self.reasoning_deep_llm, self.risk_manager_memory
        )

        workflow = StateGraph(AgentState)

        for analyst_type in selected_analysts:
            workflow.add_node(
                f"{analyst_type.capitalize()} Analyst", analyst_nodes[analyst_type]
            )
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i + 1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
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
            {"Conservative Analyst": "Conservative Analyst", "Risk Judge": "Risk Judge"},
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Neutral Analyst": "Neutral Analyst", "Risk Judge": "Risk Judge"},
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Aggressive Analyst": "Aggressive Analyst", "Risk Judge": "Risk Judge"},
        )

        workflow.add_edge("Risk Judge", END)

        return workflow.compile()


class HybridTradingGraph(TradingAgentsGraph):
    """TradingAgentsGraph with per-agent LLM provider routing.

    Usage:
        from src.hybrid_llm import CONFIGS
        from src.hybrid_graph import HybridTradingGraph

        graph = HybridTradingGraph(
            hybrid_config=CONFIGS["hybrid_qwen"],
            config={...},
        )
        result = graph.propagate("AAPL", "2026-02-27")
    """

    def __init__(
        self,
        hybrid_config: HybridLLMConfig,
        selected_analysts=None,
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        self._hybrid_config = hybrid_config
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]
        self._selected_analysts = selected_analysts

        # Let parent __init__ do all the standard setup
        super().__init__(
            selected_analysts=selected_analysts,
            debug=debug,
            config=config,
            callbacks=callbacks,
        )

        # Now rebuild the graph with hybrid LLM routing
        llms = create_hybrid_llms(self._hybrid_config)

        self.graph_setup = HybridGraphSetup(
            tool_llm=llms["tool_calling_llm"],
            reasoning_quick_llm=llms["reasoning_quick_llm"],
            reasoning_deep_llm=llms["reasoning_deep_llm"],
            tool_nodes=self.tool_nodes,
            bull_memory=self.bull_memory,
            bear_memory=self.bear_memory,
            trader_memory=self.trader_memory,
            invest_judge_memory=self.invest_judge_memory,
            risk_manager_memory=self.risk_manager_memory,
            conditional_logic=self.conditional_logic,
        )

        self.graph = self.graph_setup.setup_graph(self._selected_analysts)

        logger.info(
            "HybridTradingGraph initialized: %s", self._hybrid_config.to_dict()
        )
