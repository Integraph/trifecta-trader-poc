"""Test dual-source extraction against real pipeline output from Task 009.

This test uses actual text from the March 2, 2026 AAPL pipeline run.
"""

import pytest
from src.execution.trade_params import extract_trade_params, extract_trade_params_dual


# Actual Trader output from the AAPL pipeline run (appears in TASK_008_REPORT.md)
ACTUAL_TRADER_OUTPUT = """## KEY METRICS

- **Earnings Per Share (EPS):** $17.24
- **Forward Price-to-Earnings Ratio (P/E):** 33.47
- **Price-to-Book Ratio (P/B):** 44.08
- **Free Cash Flow:** $53.93 billion
- **Revenue Growth Rate (YoY Q1 FY2026):** 15.7%
- **Services Segment Margins:** 30%
- **Current Stock Price:** $264.18
- **Apple iPhone 17e Starting Price:** $599
- **Price Below 50 Day Simple Moving Average (SMA):** Current price is below $266.42 SMA
- **Stop-Loss Support Level:** $255

## TECHNICALS AND MACROS

- **Relative Strength Index (RSI):** 58, suggesting moderate strength but not overbought territory
- **MACD Crossover:** Bearish crossover below the signal line at -1.79
- **50 Day SMA:** $266.42; price is currently trading below it at $264.18
- **Near-Term Price Support:** $255

## RISK MANAGEMENT AND VALUATION

**Valuation Metrics:**

- Current P/E ratio of 33.47 suggests the stock is richly valued
- Free cash flow of $53.93 billion indicates strong financial health

**Risk Factors and Execution Parameters:**

1. **Near-term Risks:**
    - Potential iPhone ASP compression due to lower pricing on the model 17e
    - Geopolitical tensions impacting supply chains

2. **Technical Analysis:**
    - Price has broken below the 50 SMA, indicating potential weakness.
    - Support at $255 can be key for near-term price recovery.

3. **Execution Parameters:**

- Decision: SELL (40-50% position reduction)
- Entry Price: $264.18
- Stop-Loss: $256.50 (3% below current market price for exiting the trade if conditions worsen)
- Price Target: $248.00 (to capture 6% downside risk and wait for a more favorable entry)
- Risk/Reward Ratio: 1:2 (for every dollar of potential loss, aiming to profit two dollars or more)
- Position Size: Reduce to 2-3% of portfolio from the current suggested level, based on lower confidence in near-term upside

## FINAL TRANSACTION PROPOSAL: **SELL**"""


class TestDualExtractionWithRealData:

    def test_trader_output_extraction(self):
        """Trader output should have extractable execution parameters."""
        params = extract_trade_params(
            "AAPL", "SELL", 9.4, ACTUAL_TRADER_OUTPUT, current_price=264.18
        )
        assert params.stop_loss is not None, "Stop-loss not found in Trader output"
        assert params.price_target is not None, "Target not found in Trader output"
        print(f"Trader extraction: SL=${params.stop_loss}, Target=${params.price_target}")

    def test_dual_extraction_makes_actionable(self):
        """Dual extraction with empty Risk Judge should use Trader values."""
        judge_text = "Sell recommendation for AAPL. Target around $268. Position 40% reduction."

        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4,
            final_decision_text=judge_text,
            trader_plan_text=ACTUAL_TRADER_OUTPUT,
            current_price=264.18,
        )

        assert params.stop_loss is not None, "Should get stop-loss from Trader fallback"
        assert params.price_target is not None
        assert params.is_actionable, (
            f"Should be actionable: SL=${params.stop_loss}, "
            f"Target=${params.price_target}, "
            f"Quality={params.quality_score}"
        )
        print(f"\nDual extraction result:")
        print(f"  Stop-loss:   ${params.stop_loss}")
        print(f"  Target:      ${params.price_target}")
        print(f"  Position:    {params.position_pct}%")
        print(f"  R/R ratio:   {params.risk_reward_ratio}")
        print(f"  Actionable:  {params.is_actionable}")
