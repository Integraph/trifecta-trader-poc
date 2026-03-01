"""Test whether larger Ollama models can handle tool calling."""

import pytest
import json
from langchain_openai import ChatOpenAI


def _ollama_available():
    """Check if Ollama is accessible."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama not running on localhost:11434"
)


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

try:
    import urllib.request as _urllib_req
    import json as _json_mod
    _resp = _urllib_req.urlopen("http://localhost:11434/api/tags", timeout=2)
    _data = _json_mod.loads(_resp.read())
    _available = [m["name"] for m in _data.get("models", [])]
    for _model in ["qwen2.5:32b", "qwen2.5:72b", "llama3.3:70b", "command-r:35b"]:
        if any(_model in m for m in _available) and _model not in OLLAMA_MODELS:
            OLLAMA_MODELS.append(_model)
except Exception:
    pass


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

    has_tool_calls = hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    if has_tool_calls:
        tc = result.tool_calls[0]
        assert tc["name"] == "get_stock_price", f"Wrong tool name: {tc['name']}"
        assert "ticker" in tc["args"], f"Missing 'ticker' arg: {tc['args']}"
        print(f"\n  {model_name}: Tool calling WORKS")
        print(f"   Tool call: {tc['name']}({tc['args']})")
    else:
        print(f"\n  {model_name}: Tool calling FAILED")
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

    result = llm_with_tools.invoke("What is AAPL trading at?")

    has_tool_calls = hasattr(result, "tool_calls") and len(result.tool_calls) > 0
    if has_tool_calls:
        tc = result.tool_calls[0]
        assert tc["name"] == "get_stock_price", (
            f"Model called wrong tool: {tc['name']} (expected get_stock_price)"
        )
        print(f"\n  {model_name}: Multi-tool selection WORKS -- chose {tc['name']}")
    else:
        print(f"\n  {model_name}: Multi-tool selection FAILED -- no tool calls")
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

    assert len(content) > 100, f"Response too short ({len(content)} chars)"
    assert any(word in content.upper() for word in ["AAPL", "APPLE"]), "Should mention AAPL/Apple"
    assert any(char.isdigit() for char in content), "Should contain numbers/data"

    bull_words = sum(1 for w in ["buy", "bullish", "upside", "growth", "strong", "opportunity"]
                     if w in content.lower())
    assert bull_words >= 2, f"Response doesn't seem bullish enough (only {bull_words} bull keywords)"

    print(f"\n  {model_name}: Reasoning quality OK")
    print(f"   Length: {len(content)} chars")
    print(f"   Bull keywords found: {bull_words}")
    print(f"   First 200 chars: {content[:200]}")
