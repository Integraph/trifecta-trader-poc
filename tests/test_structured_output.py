"""Test that enhanced local models produce structured EXECUTION PARAMETERS blocks.

Run with: pytest tests/test_structured_output.py -v --run-comparison
"""

import pytest
import re

pytestmark = pytest.mark.skipif(
    "not config.getoption('--run-comparison')",
    reason="Requires --run-comparison flag and running Ollama",
)


def _model_available(model_name: str) -> bool:
    """Check if model is available in Ollama."""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        return model_name in result.stdout
    except Exception:
        return False


# Sample analysis data for the model to work with
SAMPLE_ANALYSIS_DATA = """
AAPL Current Analysis:
- Current Price: $272.15
- 50-day SMA: $265.20
- 200-day SMA: $252.80
- RSI(14): 58.3
- P/E Ratio: 28.5x
- Forward P/E: 26.2x
- EPS (TTM): $9.55
- Revenue Growth: 6.2% YoY
- Free Cash Flow: $108.4B
- Debt/Equity: 1.87
- 52-week range: $230.15 - $285.40
- Average Volume: 48.2M
- MACD: 1.35 (signal: 0.89)
- Analyst consensus: 32 buy, 8 hold, 2 sell

Based on this data, provide a complete trading analysis for AAPL with a BUY recommendation.
"""


class TestStructuredOutput:

    @pytest.mark.skipif(
        not _model_available("qwen2.5:14b"),
        reason="qwen2.5:14b not available in Ollama",
    )
    def test_qwen14b_produces_execution_params_block(self):
        """Qwen 14B with enhanced prompt should produce ## EXECUTION PARAMETERS."""
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_openai import ChatOpenAI
        from src.enhanced_llm import create_enhanced_llm

        base_llm = ChatOpenAI(
            model="qwen2.5:14b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.1,
        )
        enhanced = create_enhanced_llm(base_llm, style="financial_analysis")

        response = enhanced.invoke(SAMPLE_ANALYSIS_DATA)
        text = response.content if hasattr(response, 'content') else str(response)

        # Must contain the structured block
        assert "EXECUTION PARAMETERS" in text, (
            f"Model did not produce EXECUTION PARAMETERS block. Output:\n{text[:500]}"
        )

        # Extract the block and verify key fields
        block_match = re.search(
            r'(?:##\s*)?EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
            text,
            re.DOTALL | re.IGNORECASE,
        )
        assert block_match, "Could not parse EXECUTION PARAMETERS block"
        block = block_match.group(1)

        # Check required fields are present
        assert re.search(r'Stop-Loss:\s*\$[\d,]+', block), f"No stop-loss in block:\n{block}"
        assert re.search(r'Price Target:\s*\$[\d,]+', block), f"No target in block:\n{block}"
        assert re.search(r'Position Size:\s*[\d.]+%', block), f"No position size in block:\n{block}"

        print(f"\n{'='*60}")
        print("Structured output from qwen2.5:14b:")
        print(f"{'='*60}")
        print(text[-800:])  # Print last 800 chars to show the block

    @pytest.mark.skipif(
        not _model_available("qwen2.5:14b"),
        reason="qwen2.5:14b not available in Ollama",
    )
    def test_extracted_params_from_live_model(self):
        """Full integration: enhanced model -> extractor -> TradeParams."""
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_openai import ChatOpenAI
        from src.enhanced_llm import create_enhanced_llm
        from src.execution.trade_params import extract_trade_params

        base_llm = ChatOpenAI(
            model="qwen2.5:14b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.1,
        )
        enhanced = create_enhanced_llm(base_llm, style="financial_analysis")

        response = enhanced.invoke(SAMPLE_ANALYSIS_DATA)
        text = response.content if hasattr(response, 'content') else str(response)

        params = extract_trade_params(
            ticker="AAPL",
            decision="BUY",
            quality_score=9.5,
            decision_text=text,
            current_price=272.15,
        )

        print(f"\nExtracted params:")
        print(f"  Stop-loss:   ${params.stop_loss}")
        print(f"  Target:      ${params.price_target}")
        print(f"  Position %:  {params.position_pct}%")
        print(f"  Entry:       ${params.entry_price}")
        print(f"  R/R ratio:   {params.risk_reward_ratio}")
        print(f"  Confidence:  {params.confidence}")
        print(f"  Actionable:  {params.is_actionable}")
        print(f"  Bracket:     {params.has_bracket_params}")

        # With structured output, these should all be present
        assert params.stop_loss is not None, "Stop-loss not extracted"
        assert params.price_target is not None, "Price target not extracted"
        assert params.position_pct is not None, "Position size not extracted"
        assert params.is_actionable, "Should be actionable with all params present"
        assert params.has_bracket_params, "Should have bracket params"
