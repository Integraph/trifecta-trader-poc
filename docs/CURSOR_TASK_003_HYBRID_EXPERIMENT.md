# Cursor Task 003: Hybrid LLM Experiment — Local vs Cloud Agents

## Objective
Determine which agents in the TradingAgents pipeline can run on local Ollama models without unacceptable quality loss, and build the infrastructure to support per-agent LLM routing (hybrid mode).

## Context
- Tasks 001 and 002 established a working pipeline using Anthropic Claude for all agents
- Early testing showed 7B/8B local models (llama3.1:8b, mistral:7b) **cannot** handle tool calling — they describe tool usage in text instead of invoking tools
- The user now has two larger local models that may handle tool calling better:
  - `qwen2.5:14b` (~9 GB, quantized)
  - `mistral-small:22b` (~12 GB, quantized)
- The pipeline has **4 tool-calling agents** (analysts) and **8 pure-reasoning agents** (debaters, judges, trader)
- Currently, all agents must use the **same LLM provider** — both `deep_thinking_llm` and `quick_thinking_llm` are created from the same `llm_provider` config value
- The goal is to find the optimal cost/quality split: cloud where needed, local where possible

## Important Rules
- **DO NOT modify files in `vendor/TradingAgents/` directly** — keep upstream clean
- All new code goes in `src/` and `tests/`
- If a vendor modification is absolutely required, document it in the report
- **DO NOT run the full pipeline** unless explicitly told to — it costs money and takes ~20 minutes
- All experiments that require API calls should be **small, isolated tests** documented in test files

## Architecture Reference

### Agent Classification

**Tool-Calling Agents (4)** — these call external tools via `llm.bind_tools()`:
| Agent | Tools Used | LLM Instance |
|-------|-----------|---------------|
| Market Analyst | `get_stock_data`, `get_indicators` | `quick_thinking_llm` |
| Social Media Analyst | `get_news` | `quick_thinking_llm` |
| News Analyst | `get_news`, `get_global_news`, `get_insider_transactions` | `quick_thinking_llm` |
| Fundamentals Analyst | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` | `quick_thinking_llm` |

**Pure Reasoning Agents (8)** — these use `llm.invoke()` with no tool binding:
| Agent | LLM Instance | Role |
|-------|--------------|------|
| Bull Researcher | `quick_thinking_llm` | Argues bull case in debate |
| Bear Researcher | `quick_thinking_llm` | Argues bear case in debate |
| Research Manager | `deep_thinking_llm` | Judges debate, creates investment plan |
| Trader | `quick_thinking_llm` | Creates trading proposal |
| Aggressive Debator | `quick_thinking_llm` | Risk debate — aggressive stance |
| Conservative Debator | `quick_thinking_llm` | Risk debate — conservative stance |
| Neutral Debator | `quick_thinking_llm` | Risk debate — balanced stance |
| Risk Manager (Judge) | `deep_thinking_llm` | Final risk-adjusted decision |

### Current LLM Routing (vendor code)

In `vendor/TradingAgents/tradingagents/graph/trading_graph.py`:

```python
# Lines 81-95: Both use the SAME provider
deep_client = create_llm_client(provider=self.config["llm_provider"], model=self.config["deep_think_llm"], ...)
quick_client = create_llm_client(provider=self.config["llm_provider"], model=self.config["quick_think_llm"], ...)
```

In `vendor/TradingAgents/tradingagents/graph/setup.py`:

```python
# Quick LLM: all analysts, debaters, bull/bear researchers, trader
analyst_nodes["market"] = create_market_analyst(self.quick_thinking_llm)
bull_researcher_node = create_bull_researcher(self.quick_thinking_llm, self.bull_memory)
trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)
aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)

# Deep LLM: only the two judges
research_manager_node = create_research_manager(self.deep_thinking_llm, self.invest_judge_memory)
risk_manager_node = create_risk_manager(self.deep_thinking_llm, self.risk_manager_memory)
```

### LLM Client Factory

In `vendor/TradingAgents/tradingagents/llm_clients/factory.py`:

```python
def create_llm_client(provider, model, **kwargs):
    if provider == "ollama":
        return OpenAIClient(model=model, base_url="http://localhost:11434/v1", api_key="ollama", ...)
    elif provider == "anthropic":
        return AnthropicClient(model=model, ...)
    # etc.
```

---

## Step 1: Test Tool Calling on Larger Local Models

### Goal
Determine if `qwen2.5:14b` and `mistral-small:22b` can execute OpenAI-style tool calls through LangChain.

### Implementation

Create `tests/test_local_tool_calling.py`:

```python
"""Test whether larger Ollama models can handle tool calling."""

import pytest
import json
from langchain_openai import ChatOpenAI

# Skip if Ollama is not running
pytestmark = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama not running on localhost:11434"
)

def _ollama_available():
    """Check if Ollama is accessible."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False

def _get_test_tool():
    """Create a simple test tool definition."""
    from langchain_core.tools import tool

    @tool
    def get_stock_price(ticker: str, date: str) -> str:
        """Get the stock price for a given ticker on a given date.

        Args:
            ticker: The stock ticker symbol (e.g., 'AAPL')
            date: The date in YYYY-MM-DD format
        """
        return json.dumps({"ticker": ticker, "date": date, "close": 185.50, "open": 183.20})

    return get_stock_price


OLLAMA_MODELS = [
    "qwen2.5:14b",
    "mistral-small:22b",
]


@pytest.mark.parametrize("model_name", OLLAMA_MODELS)
def test_tool_calling_basic(model_name):
    """Test if the model can produce a valid tool call."""
    llm = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    tool = _get_test_tool()
    llm_with_tools = llm.bind_tools([tool])

    result = llm_with_tools.invoke(
        "What is the stock price of AAPL on 2026-02-27?"
    )

    # Check if the model produced tool calls
    has_tool_calls = hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    if has_tool_calls:
        # Verify the tool call structure
        tc = result.tool_calls[0]
        assert tc["name"] == "get_stock_price", f"Wrong tool name: {tc['name']}"
        assert "ticker" in tc["args"], f"Missing 'ticker' arg: {tc['args']}"
        print(f"\n✅ {model_name}: Tool calling WORKS")
        print(f"   Tool call: {tc['name']}({tc['args']})")
    else:
        # Model responded with text instead of tool call
        print(f"\n❌ {model_name}: Tool calling FAILED")
        print(f"   Response (first 200 chars): {result.content[:200]}")
        pytest.fail(
            f"{model_name} did not produce tool calls. "
            f"Response: {result.content[:200]}"
        )


@pytest.mark.parametrize("model_name", OLLAMA_MODELS)
def test_tool_calling_multi_tool(model_name):
    """Test if the model can choose the right tool when multiple are available."""
    from langchain_core.tools import tool

    @tool
    def get_stock_price(ticker: str) -> str:
        """Get the current stock price for a ticker symbol."""
        return json.dumps({"ticker": ticker, "price": 185.50})

    @tool
    def get_company_news(company: str) -> str:
        """Get recent news articles about a company."""
        return json.dumps({"company": company, "articles": []})

    llm = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    llm_with_tools = llm.bind_tools([get_stock_price, get_company_news])

    # Ask for stock price — should call get_stock_price, not get_company_news
    result = llm_with_tools.invoke("What is AAPL trading at?")

    has_tool_calls = hasattr(result, "tool_calls") and len(result.tool_calls) > 0
    if has_tool_calls:
        tc = result.tool_calls[0]
        assert tc["name"] == "get_stock_price", (
            f"Model called wrong tool: {tc['name']} (expected get_stock_price)"
        )
        print(f"\n✅ {model_name}: Multi-tool selection WORKS — chose {tc['name']}")
    else:
        print(f"\n❌ {model_name}: Multi-tool selection FAILED — no tool calls")
        pytest.fail(f"{model_name} did not produce tool calls")


@pytest.mark.parametrize("model_name", OLLAMA_MODELS)
def test_reasoning_quality(model_name):
    """Test reasoning quality for debate-style prompts (no tool calling needed)."""
    llm = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    prompt = """You are a bull-case researcher analyzing AAPL stock.

Given the following data:
- Current Price: $270
- P/E Ratio: 33x
- Revenue Growth: 8% YoY
- Services Revenue: $100B (growing 15% YoY)
- Cash on Hand: $160B
- Recent News: Apple Vision Pro sales exceeded expectations

Make the strongest possible bull case for buying AAPL. Be specific with numbers.
Keep your response under 300 words."""

    result = llm.invoke(prompt)
    content = result.content

    # Quality checks
    assert len(content) > 100, f"Response too short ({len(content)} chars)"
    assert any(word in content.upper() for word in ["AAPL", "APPLE"]), "Should mention AAPL/Apple"
    assert any(char.isdigit() for char in content), "Should contain numbers/data"

    # Check it's actually bullish (not bearish)
    bull_words = sum(1 for w in ["buy", "bullish", "upside", "growth", "strong", "opportunity"]
                     if w in content.lower())
    assert bull_words >= 2, f"Response doesn't seem bullish enough (only {bull_words} bull keywords)"

    print(f"\n✅ {model_name}: Reasoning quality OK")
    print(f"   Length: {len(content)} chars")
    print(f"   Bull keywords found: {bull_words}")
    print(f"   First 200 chars: {content[:200]}")
```

### Run

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc
pytest tests/test_local_tool_calling.py -v -s 2>&1 | head -100
```

Record the results for each model. This determines the approach for Step 2.

---

## Step 2: Build Hybrid LLM Router

### Goal
Create `src/hybrid_llm.py` — a wrapper that allows assigning different LLM providers to different agents.

### Implementation

Create `src/hybrid_llm.py`:

```python
"""
Hybrid LLM router for assigning different providers to different agents.

This module creates a patched version of TradingAgentsGraph that supports
per-agent LLM provider routing. Tool-calling agents use cloud LLMs (Anthropic),
while pure-reasoning agents can use local Ollama models.
"""

import logging
from typing import Any, Dict, Optional
from tradingagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)


class HybridLLMConfig:
    """Configuration for hybrid LLM routing.

    Defines which provider/model to use for each agent category:
    - tool_calling: Agents that use bind_tools() — must support tool calling
    - reasoning_quick: Pure reasoning agents on the quick path
    - reasoning_deep: Pure reasoning agents on the deep path (judges)
    """

    def __init__(
        self,
        # Tool-calling agents (analysts) — needs reliable tool calling
        tool_provider: str = "anthropic",
        tool_model: str = "claude-sonnet-4-5-20250929",
        # Quick reasoning agents (debaters, bull/bear, trader)
        reasoning_quick_provider: str = "ollama",
        reasoning_quick_model: str = "qwen2.5:14b",
        # Deep reasoning agents (research manager, risk manager)
        reasoning_deep_provider: str = "anthropic",
        reasoning_deep_model: str = "claude-sonnet-4-5-20250929",
    ):
        self.tool_provider = tool_provider
        self.tool_model = tool_model
        self.reasoning_quick_provider = reasoning_quick_provider
        self.reasoning_quick_model = reasoning_quick_model
        self.reasoning_deep_provider = reasoning_deep_provider
        self.reasoning_deep_model = reasoning_deep_model

    def to_dict(self) -> Dict[str, Any]:
        """Return a summary dict for logging/reporting."""
        return {
            "tool_calling": f"{self.tool_provider}/{self.tool_model}",
            "reasoning_quick": f"{self.reasoning_quick_provider}/{self.reasoning_quick_model}",
            "reasoning_deep": f"{self.reasoning_deep_provider}/{self.reasoning_deep_model}",
        }


# Predefined configurations for A/B testing
CONFIGS = {
    # Baseline: all Claude (the control)
    "all_cloud": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="anthropic",
        reasoning_quick_model="claude-sonnet-4-5-20250929",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    # Hybrid A: local reasoning (quick), cloud for tools and judges
    "hybrid_qwen": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    # Hybrid B: local reasoning (quick), cloud for tools and judges
    "hybrid_mistral": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="mistral-small:22b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    # Aggressive hybrid: local for everything except tool-calling analysts
    "hybrid_aggressive_qwen": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="ollama",
        reasoning_deep_model="qwen2.5:14b",
    ),

    # Aggressive hybrid: local for everything except tool-calling analysts
    "hybrid_aggressive_mistral": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="mistral-small:22b",
        reasoning_deep_provider="ollama",
        reasoning_deep_model="mistral-small:22b",
    ),
}


def create_hybrid_llms(hybrid_config: HybridLLMConfig) -> Dict[str, Any]:
    """Create three LLM instances for hybrid routing.

    Returns:
        Dict with keys: 'tool_calling_llm', 'reasoning_quick_llm', 'reasoning_deep_llm'
    """
    llms = {}

    # Tool-calling LLM (analysts)
    tool_client = create_llm_client(
        provider=hybrid_config.tool_provider,
        model=hybrid_config.tool_model,
    )
    llms["tool_calling_llm"] = tool_client.get_llm()

    # Quick reasoning LLM (debaters, researchers, trader)
    reasoning_quick_client = create_llm_client(
        provider=hybrid_config.reasoning_quick_provider,
        model=hybrid_config.reasoning_quick_model,
    )
    llms["reasoning_quick_llm"] = reasoning_quick_client.get_llm()

    # Deep reasoning LLM (judges)
    reasoning_deep_client = create_llm_client(
        provider=hybrid_config.reasoning_deep_provider,
        model=hybrid_config.reasoning_deep_model,
    )
    llms["reasoning_deep_llm"] = reasoning_deep_client.get_llm()

    logger.info("Hybrid LLM routing: %s", hybrid_config.to_dict())
    return llms
```

### Create the Patched Graph Setup

Create `src/hybrid_graph.py`:

```python
"""
Patched TradingAgentsGraph that supports per-agent LLM routing.

This subclass overrides the LLM assignment so that:
- Tool-calling agents (analysts) get one LLM (typically cloud)
- Quick reasoning agents get another LLM (can be local)
- Deep reasoning agents (judges) get a third LLM (typically cloud)
"""

import logging
from typing import Any, Dict, List, Optional

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.setup import GraphSetup
from tradingagents.agents import (
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


class HybridTradingGraph(TradingAgentsGraph):
    """TradingAgentsGraph with per-agent LLM provider routing.

    Usage:
        from src.hybrid_llm import CONFIGS
        from src.hybrid_graph import HybridTradingGraph

        graph = HybridTradingGraph(
            hybrid_config=CONFIGS["hybrid_qwen"],
            config={...},  # standard TradingAgents config
        )
        result = graph.propagate(ticker="AAPL", ...)
    """

    def __init__(
        self,
        hybrid_config: HybridLLMConfig,
        selected_analysts=None,
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        # Store hybrid config before parent init
        self._hybrid_config = hybrid_config

        # Initialize parent — this creates self.deep_thinking_llm and
        # self.quick_thinking_llm using the config's llm_provider.
        # We'll override the graph setup to use our hybrid LLMs instead.
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]

        super().__init__(
            selected_analysts=selected_analysts,
            debug=debug,
            config=config,
            callbacks=callbacks,
        )

    def _setup_graph(self):
        """Override graph setup to use hybrid LLM routing.

        Called from __init__ after LLM clients are created.
        We replace the standard dual-LLM setup with our triple-LLM hybrid.
        """
        # Create the three hybrid LLMs
        llms = create_hybrid_llms(self._hybrid_config)
        tool_llm = llms["tool_calling_llm"]
        reasoning_quick_llm = llms["reasoning_quick_llm"]
        reasoning_deep_llm = llms["reasoning_deep_llm"]

        # Override the parent's LLM assignments
        # Analysts get the tool-calling LLM
        # NOTE: The parent's setup_graph() uses self.quick_thinking_llm for analysts
        # and self.deep_thinking_llm for judges. We override both.
        self.quick_thinking_llm = tool_llm  # Used by analysts (tool calling)
        self.deep_thinking_llm = reasoning_deep_llm  # Used by judges

        # Now we need to rebuild the graph with correct routing.
        # The problem: setup_graph() gives ALL quick agents the same LLM.
        # Analysts need tool_llm, but debaters/researchers need reasoning_quick_llm.
        # We have to patch the node creation after setup.

        # Let the parent build the graph first with tool_llm as quick
        self.graph_setup = GraphSetup(self)
        graph = self.graph_setup.setup_graph(
            self.selected_analysts, self.debug
        )

        # Now patch the reasoning nodes to use the reasoning LLM
        # The graph nodes are functions that close over the LLM.
        # We need to recreate them with the correct LLM.
        logger.info(
            "Hybrid routing: tool_calling=%s/%s, reasoning_quick=%s/%s, reasoning_deep=%s/%s",
            self._hybrid_config.tool_provider,
            self._hybrid_config.tool_model,
            self._hybrid_config.reasoning_quick_provider,
            self._hybrid_config.reasoning_quick_model,
            self._hybrid_config.reasoning_deep_provider,
            self._hybrid_config.reasoning_deep_model,
        )

        return graph
```

**IMPORTANT:** The patching approach above is a starting point. The exact implementation depends on how `setup_graph()` works internally. Here's what you need to investigate:

1. Read `vendor/TradingAgents/tradingagents/graph/setup.py` — specifically `setup_graph()` method
2. Determine if nodes can be replaced after graph construction, or if the graph must be rebuilt
3. If nodes can't be replaced post-construction, you'll need to **copy and modify** `setup_graph()` in `src/hybrid_graph.py` to accept three LLMs instead of two

The cleanest approach is likely:

```python
class HybridGraphSetup(GraphSetup):
    """Patched GraphSetup that accepts three LLM instances."""

    def __init__(self, graph, tool_llm, reasoning_quick_llm, reasoning_deep_llm):
        super().__init__(graph)
        self._tool_llm = tool_llm
        self._reasoning_quick_llm = reasoning_quick_llm
        self._reasoning_deep_llm = reasoning_deep_llm

    def setup_graph(self, selected_analysts, debug):
        # Copy the parent's setup_graph but replace LLM assignments:
        # - Analysts: self._tool_llm
        # - Bull/Bear/Trader/Debaters: self._reasoning_quick_llm
        # - Research Manager/Risk Manager: self._reasoning_deep_llm
        ...
```

You will need to read the full `setup_graph()` method and replicate it with the three-way split. Document any vendor code you had to copy.

---

## Step 3: Create Quality Comparison Framework

### Goal
Build a scoring system that compares pipeline outputs across different LLM configurations.

### Implementation

Create `src/quality_scorer.py`:

```python
"""
Quality scorer for comparing pipeline outputs across LLM configurations.

Evaluates the quality of trading analysis along several dimensions:
- Decision consistency (does the extracted decision match the reasoning?)
- Reasoning depth (length, specificity, use of data)
- Data grounding (does the analysis reference real data points?)
- Risk awareness (are risk parameters specified?)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QualityScore:
    """Quality assessment of a pipeline run."""
    config_name: str
    ticker: str
    trade_date: str

    # Decision quality
    decision: str  # BUY/HOLD/SELL/UNKNOWN
    decision_consistent: bool  # Does extracted decision match reasoning?

    # Reasoning quality (0-10 scale)
    reasoning_depth: int  # Length and structure of analysis
    data_grounding: int  # References to specific numbers/data
    risk_awareness: int  # Presence of risk parameters (stop-loss, targets)

    # Performance
    total_tokens_approx: int = 0  # Rough token count
    estimated_cost_usd: float = 0.0  # Estimated API cost

    # Raw data
    full_output_length: int = 0
    num_data_points_cited: int = 0
    has_stop_loss: bool = False
    has_price_target: bool = False
    has_position_sizing: bool = False

    @property
    def composite_score(self) -> float:
        """Weighted composite quality score (0-10)."""
        return (
            (3.0 if self.decision_consistent else 0.0) +  # 30% weight
            self.reasoning_depth * 0.3 +  # 30% weight
            self.data_grounding * 0.2 +  # 20% weight
            self.risk_awareness * 0.2  # 20% weight
        )


def score_pipeline_output(
    config_name: str,
    ticker: str,
    trade_date: str,
    final_trade_decision: str,
    extracted_decision: str,
    full_output: Optional[str] = None,
) -> QualityScore:
    """Score a pipeline output for quality.

    Args:
        config_name: Name of the hybrid config used
        ticker: Stock ticker
        trade_date: Analysis date
        final_trade_decision: Raw text from the pipeline
        extracted_decision: Decision extracted by our signal processor
        full_output: Full pipeline output text (if available)

    Returns:
        QualityScore with all metrics
    """
    text = final_trade_decision or ""

    # Decision consistency
    from src.signal_processing import extract_decision
    our_decision = extract_decision(text)
    decision_consistent = our_decision == extracted_decision

    # Reasoning depth (0-10)
    word_count = len(text.split())
    if word_count > 500:
        reasoning_depth = min(10, 5 + (word_count - 500) // 200)
    elif word_count > 200:
        reasoning_depth = 3 + (word_count - 200) // 100
    elif word_count > 50:
        reasoning_depth = 1 + word_count // 50
    else:
        reasoning_depth = 0

    # Data grounding (0-10) — count specific numbers/percentages
    numbers = re.findall(r'\$[\d,.]+|\d+\.?\d*%|\d{1,3}(?:,\d{3})+', text)
    num_data_points = len(numbers)
    data_grounding = min(10, num_data_points // 2)

    # Risk awareness (0-10)
    risk_score = 0
    has_stop_loss = bool(re.search(r'stop.?loss', text, re.IGNORECASE))
    has_price_target = bool(re.search(r'(?:price\s+)?target|upside|downside', text, re.IGNORECASE))
    has_position_sizing = bool(re.search(r'position\s+siz|allocation|portfolio\s+weight', text, re.IGNORECASE))

    if has_stop_loss:
        risk_score += 4
    if has_price_target:
        risk_score += 3
    if has_position_sizing:
        risk_score += 3
    risk_awareness = min(10, risk_score)

    return QualityScore(
        config_name=config_name,
        ticker=ticker,
        trade_date=trade_date,
        decision=extracted_decision,
        decision_consistent=decision_consistent,
        reasoning_depth=reasoning_depth,
        data_grounding=data_grounding,
        risk_awareness=risk_awareness,
        full_output_length=len(text),
        num_data_points_cited=num_data_points,
        has_stop_loss=has_stop_loss,
        has_price_target=has_price_target,
        has_position_sizing=has_position_sizing,
    )


def compare_scores(scores: List[QualityScore]) -> str:
    """Generate a comparison report from multiple pipeline runs.

    Args:
        scores: List of QualityScore from different configurations

    Returns:
        Formatted comparison table as a string
    """
    if not scores:
        return "No scores to compare."

    lines = []
    lines.append("=" * 80)
    lines.append("HYBRID LLM QUALITY COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    header = f"{'Config':<25} {'Decision':<8} {'Consistent':<11} {'Depth':<6} {'Data':<6} {'Risk':<6} {'Score':<6} {'Cost':<8}"
    lines.append(header)
    lines.append("-" * 80)

    for s in sorted(scores, key=lambda x: x.composite_score, reverse=True):
        line = (
            f"{s.config_name:<25} "
            f"{s.decision:<8} "
            f"{'✅' if s.decision_consistent else '❌':<11} "
            f"{s.reasoning_depth:<6} "
            f"{s.data_grounding:<6} "
            f"{s.risk_awareness:<6} "
            f"{s.composite_score:<6.1f} "
            f"${s.estimated_cost_usd:<7.4f}"
        )
        lines.append(line)

    lines.append("-" * 80)
    lines.append("")

    # Winner
    best = max(scores, key=lambda x: x.composite_score)
    cheapest = min(scores, key=lambda x: x.estimated_cost_usd)
    lines.append(f"Best quality:  {best.config_name} (score: {best.composite_score:.1f})")
    lines.append(f"Lowest cost:   {cheapest.config_name} (${cheapest.estimated_cost_usd:.4f})")

    return "\n".join(lines)
```

### Tests

Create `tests/test_quality_scorer.py`:

```python
"""Tests for the quality scoring system."""

from src.quality_scorer import score_pipeline_output, compare_scores


class TestQualityScorer:
    """Test quality scoring logic."""

    def test_high_quality_output(self):
        """A detailed output with risk params should score high."""
        text = """
        ## Analysis Summary

        AAPL is trading at $270 with a P/E of 33x. Revenue grew 8% YoY
        with services reaching $100B. Market cap is $4.1T.

        The technical setup shows RSI at 62, MACD bullish crossover,
        and price above the 50-day SMA at $265.

        ### Risk Management
        - Stop-loss: $258 (-4.4%)
        - Price target: $295 (+9.3%)
        - Position sizing: 5% of portfolio

        ## FINAL TRANSACTION PROPOSAL: **HOLD**

        With active risk management and trim target at $280+.
        """
        score = score_pipeline_output(
            config_name="test",
            ticker="AAPL",
            trade_date="2026-02-27",
            final_trade_decision=text,
            extracted_decision="HOLD",
        )

        assert score.decision == "HOLD"
        assert score.decision_consistent
        assert score.reasoning_depth >= 3
        assert score.data_grounding >= 3
        assert score.risk_awareness >= 7
        assert score.has_stop_loss
        assert score.has_price_target
        assert score.has_position_sizing
        assert score.composite_score >= 5.0

    def test_low_quality_output(self):
        """A vague output should score low."""
        text = "I think we should buy it."
        score = score_pipeline_output(
            config_name="test_low",
            ticker="AAPL",
            trade_date="2026-02-27",
            final_trade_decision=text,
            extracted_decision="BUY",
        )

        assert score.reasoning_depth <= 2
        assert score.data_grounding <= 1
        assert score.risk_awareness == 0
        assert score.composite_score < 4.0

    def test_inconsistent_decision(self):
        """When extracted decision doesn't match, consistency is false."""
        text = "FINAL TRANSACTION PROPOSAL: **HOLD**"
        score = score_pipeline_output(
            config_name="test_inconsistent",
            ticker="AAPL",
            trade_date="2026-02-27",
            final_trade_decision=text,
            extracted_decision="SELL",  # Doesn't match HOLD in text
        )

        assert not score.decision_consistent
        assert score.composite_score < 5.0  # Penalized for inconsistency

    def test_comparison_report(self):
        """Compare scores generates a formatted report."""
        scores = [
            score_pipeline_output("all_cloud", "AAPL", "2026-02-27",
                "FINAL TRANSACTION PROPOSAL: **HOLD**\nStop-loss: $258\nTarget: $295",
                "HOLD"),
            score_pipeline_output("hybrid_qwen", "AAPL", "2026-02-27",
                "FINAL TRANSACTION PROPOSAL: **HOLD**",
                "HOLD"),
        ]
        report = compare_scores(scores)
        assert "HYBRID LLM QUALITY COMPARISON" in report
        assert "all_cloud" in report
        assert "hybrid_qwen" in report
```

---

## Step 4: Isolated Reasoning Test

### Goal
Run the same reasoning prompt through Claude and local models side-by-side, without running the full pipeline. This tests quality without API cost.

### Implementation

Create `tests/test_reasoning_comparison.py`:

```python
"""
Side-by-side reasoning comparison: Claude vs local models.

This test sends the same prompt to multiple models and compares output quality.
It does NOT run the full pipeline — it tests individual agent prompts in isolation.

NOTE: This test makes real API calls. Run with:
    pytest tests/test_reasoning_comparison.py -v -s --run-comparison
"""

import pytest
import os
import json
from dataclasses import dataclass
from typing import List

# Only run if explicitly requested (costs money for Anthropic calls)
pytestmark = pytest.mark.skipif(
    "--run-comparison" not in pytest.config.args if hasattr(pytest, "config") else True,
    reason="Reasoning comparison tests require --run-comparison flag"
)


SAMPLE_MARKET_DATA = """
=== MARKET DATA FOR AAPL (2026-02-27) ===
Current Price: $270.15
52-Week High: $295.80
52-Week Low: $218.42
P/E Ratio: 33.2x
Forward P/E: 28.5x
Revenue (TTM): $412B
Revenue Growth: 8.2% YoY
Net Income (TTM): $105B
EPS: $6.85
Free Cash Flow: $110B
Debt/Equity: 1.45
RSI (14-day): 62.3
MACD: Bullish crossover 3 days ago
50-day SMA: $265.20
200-day SMA: $252.80
Volume: 48M (vs 55M avg)
Insider Activity: Tim Cook sold $25M in shares (planned sale)
Recent News: Vision Pro 2 announcement next month, EU DMA compliance costs $2B
"""

BULL_PROMPT = f"""You are a bull-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BULL case for buying AAPL. Include:
1. Specific data points from the market data above
2. Growth catalysts
3. Valuation justification
4. Risk mitigation factors

Keep your response under 400 words. Be specific with numbers."""

BEAR_PROMPT = f"""You are a bear-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BEAR case for selling AAPL. Include:
1. Specific data points from the market data above
2. Risk factors and headwinds
3. Valuation concerns
4. Technical warning signs

Keep your response under 400 words. Be specific with numbers."""

TRADER_PROMPT = f"""You are a portfolio trader making a final trading decision.

{SAMPLE_MARKET_DATA}

Bull Argument: Strong services growth at 15% YoY, Vision Pro 2 catalyst, massive $110B free cash flow supports buybacks and dividends.

Bear Argument: Elevated 33x P/E, insider selling, EU regulatory costs of $2B, volume below average suggesting institutional caution.

Based on the bull and bear arguments and the market data, make your final trading decision.
Your response MUST end with a line in this exact format:
FINAL TRANSACTION PROPOSAL: <BUY|HOLD|SELL>

Include specific risk management parameters (stop-loss, price target, position size)."""


def _create_model_configs():
    """Define models to compare."""
    configs = []

    # Always include local models (free)
    configs.append({
        "name": "qwen2.5:14b",
        "provider": "ollama",
        "model": "qwen2.5:14b",
    })
    configs.append({
        "name": "mistral-small:22b",
        "provider": "ollama",
        "model": "mistral-small:22b",
    })

    # Include Anthropic if API key is set
    if os.environ.get("ANTHROPIC_API_KEY"):
        configs.append({
            "name": "claude-sonnet-4.5",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        })

    return configs


def _invoke_model(provider: str, model: str, prompt: str) -> str:
    """Invoke a model and return the response text."""
    if provider == "ollama":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    result = llm.invoke(prompt)
    return result.content


@pytest.fixture(scope="module")
def comparison_results():
    """Run all comparisons and collect results."""
    configs = _create_model_configs()
    results = {}

    for config in configs:
        name = config["name"]
        results[name] = {}

        for prompt_name, prompt in [
            ("bull", BULL_PROMPT),
            ("bear", BEAR_PROMPT),
            ("trader", TRADER_PROMPT),
        ]:
            try:
                response = _invoke_model(config["provider"], config["model"], prompt)
                results[name][prompt_name] = response
            except Exception as e:
                results[name][prompt_name] = f"ERROR: {e}"

    return results


def test_bull_case_quality(comparison_results):
    """Compare bull case reasoning across models."""
    print("\n" + "=" * 80)
    print("BULL CASE COMPARISON")
    print("=" * 80)

    for model_name, responses in comparison_results.items():
        response = responses.get("bull", "NOT RUN")
        word_count = len(response.split()) if not response.startswith("ERROR") else 0
        has_numbers = bool(__import__("re").findall(r'\$[\d,.]+|\d+\.?\d*%', response))

        print(f"\n--- {model_name} ---")
        print(f"Words: {word_count}, Has numbers: {has_numbers}")
        print(f"First 300 chars: {response[:300]}")

        if not response.startswith("ERROR"):
            assert word_count > 50, f"{model_name} bull case too short"


def test_trader_decision_format(comparison_results):
    """Check that each model produces a properly formatted decision."""
    print("\n" + "=" * 80)
    print("TRADER DECISION FORMAT")
    print("=" * 80)

    from src.signal_processing import extract_decision

    for model_name, responses in comparison_results.items():
        response = responses.get("trader", "NOT RUN")
        decision = extract_decision(response) if not response.startswith("ERROR") else "ERROR"

        print(f"\n--- {model_name} ---")
        print(f"Decision: {decision}")
        print(f"Last 200 chars: {response[-200:]}")

        if not response.startswith("ERROR"):
            assert decision != "UNKNOWN", (
                f"{model_name} did not produce a parseable decision. "
                f"Response tail: {response[-300:]}"
            )
```

### Run

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

# Local models only (free):
pytest tests/test_reasoning_comparison.py -v -s -k "not anthropic"

# With Anthropic comparison (costs a few cents):
pytest tests/test_reasoning_comparison.py -v -s --run-comparison
```

---

## Step 5: Update run_analysis.py for Hybrid Mode

### Goal
Add a `--hybrid` CLI flag to `run_analysis.py` so the user can choose between standard and hybrid configurations.

### Implementation

In `src/run_analysis.py`, add:

```python
import argparse
from src.hybrid_llm import CONFIGS, HybridLLMConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run trading analysis")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument(
        "--hybrid",
        choices=list(CONFIGS.keys()),
        default=None,
        help="Use hybrid LLM routing. Options: " + ", ".join(CONFIGS.keys()),
    )
    parser.add_argument("--no-debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hybrid:
        # Use hybrid routing
        from src.hybrid_graph import HybridTradingGraph
        hybrid_config = CONFIGS[args.hybrid]
        ta = HybridTradingGraph(
            hybrid_config=hybrid_config,
            debug=not args.no_debug,
            config={...},  # standard config
        )
    else:
        # Standard single-provider mode
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        ta = TradingAgentsGraph(
            debug=not args.no_debug,
            config={...},
        )

    # ... rest of pipeline ...
```

---

## Step 6: Tests for Hybrid Infrastructure

### Implementation

Create `tests/test_hybrid_llm.py`:

```python
"""Tests for hybrid LLM routing infrastructure."""

from src.hybrid_llm import HybridLLMConfig, CONFIGS, create_hybrid_llms
import pytest


class TestHybridLLMConfig:
    """Test configuration objects."""

    def test_default_config(self):
        config = HybridLLMConfig()
        assert config.tool_provider == "anthropic"
        assert config.reasoning_quick_provider == "ollama"
        assert config.reasoning_deep_provider == "anthropic"

    def test_predefined_configs_exist(self):
        assert "all_cloud" in CONFIGS
        assert "hybrid_qwen" in CONFIGS
        assert "hybrid_mistral" in CONFIGS
        assert "hybrid_aggressive_qwen" in CONFIGS
        assert "hybrid_aggressive_mistral" in CONFIGS

    def test_all_cloud_config(self):
        config = CONFIGS["all_cloud"]
        assert config.tool_provider == "anthropic"
        assert config.reasoning_quick_provider == "anthropic"
        assert config.reasoning_deep_provider == "anthropic"

    def test_hybrid_qwen_config(self):
        config = CONFIGS["hybrid_qwen"]
        assert config.tool_provider == "anthropic"
        assert config.reasoning_quick_provider == "ollama"
        assert config.reasoning_quick_model == "qwen2.5:14b"
        assert config.reasoning_deep_provider == "anthropic"

    def test_to_dict(self):
        config = CONFIGS["hybrid_qwen"]
        d = config.to_dict()
        assert "tool_calling" in d
        assert "reasoning_quick" in d
        assert "reasoning_deep" in d
        assert "ollama" in d["reasoning_quick"]

    def test_custom_config(self):
        config = HybridLLMConfig(
            tool_provider="anthropic",
            tool_model="claude-sonnet-4-5-20250929",
            reasoning_quick_provider="ollama",
            reasoning_quick_model="llama3.1:8b",
            reasoning_deep_provider="ollama",
            reasoning_deep_model="mistral-small:22b",
        )
        assert config.reasoning_quick_model == "llama3.1:8b"
        assert config.reasoning_deep_model == "mistral-small:22b"


class TestCreateHybridLLMs:
    """Test LLM creation (requires running services)."""

    @pytest.mark.skipif(
        True,  # Skip by default — requires Ollama and/or API keys
        reason="Requires running Ollama and API keys"
    )
    def test_create_all_cloud(self):
        llms = create_hybrid_llms(CONFIGS["all_cloud"])
        assert "tool_calling_llm" in llms
        assert "reasoning_quick_llm" in llms
        assert "reasoning_deep_llm" in llms
```

---

## Verification

### Run All Tests

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc
pytest tests/ -v
```

**Expected:** All existing tests (25) plus new tests pass.

### Run Tool Calling Tests (requires Ollama)

```bash
pytest tests/test_local_tool_calling.py -v -s
```

**Expected:** Results showing whether qwen2.5:14b and mistral-small:22b can handle tool calls.

### DO NOT Run Full Pipeline

The full pipeline costs money. Only run it when explicitly asked. All quality testing should be done via the isolated reasoning tests in Step 4.

### Commit

```bash
git add .
git commit -m "Add hybrid LLM routing and quality comparison framework"
git push
```

---

## Verification Checklist

- [ ] `tests/test_local_tool_calling.py` created and run — results recorded for each model
- [ ] `src/hybrid_llm.py` created with HybridLLMConfig and predefined CONFIGS
- [ ] `src/hybrid_graph.py` created with HybridTradingGraph (or approach documented if patching isn't feasible)
- [ ] `src/quality_scorer.py` created with scoring and comparison logic
- [ ] `tests/test_quality_scorer.py` created and passing
- [ ] `tests/test_hybrid_llm.py` created and passing
- [ ] `tests/test_reasoning_comparison.py` created (may skip Anthropic calls unless --run-comparison)
- [ ] `src/run_analysis.py` updated with `--hybrid` flag
- [ ] All existing tests still pass (25 from Tasks 001/002)
- [ ] Changes committed and pushed

---

## Report

After completing all steps, create `docs/TASK_003_REPORT.md` containing:
1. Which steps succeeded and which had issues
2. **Tool calling test results** for each local model (CRITICAL — this determines the hybrid strategy)
3. Output of `pytest tests/ -v` (all tests)
4. Any modifications to the approach documented above
5. Whether vendor code was modified (and if so, which files/lines)
6. The output of `git log --oneline` showing all commits
7. Recommendations for which hybrid config to use for full pipeline testing
