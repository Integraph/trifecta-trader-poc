"""Tests for the run comparison utility."""

import json
from pathlib import Path
from src.compare_runs import load_result, result_to_score


class TestCompareRuns:

    def _make_result(self, config_name="test", decision="HOLD", elapsed=120.0):
        return {
            "ticker": "AAPL",
            "trade_date": "2026-02-27",
            "provider": "anthropic",
            "hybrid_config": config_name,
            "decision": decision,
            "upstream_decision": decision,
            "final_trade_decision_text": (
                f"FINAL TRANSACTION PROPOSAL: **{decision}**\n"
                "Stop-loss: $258\nTarget: $295\n"
                "Position sizing: 5% of portfolio\n"
                "Revenue growth of 8.2% and P/E of 33x."
            ),
            "elapsed_seconds": elapsed,
            "quality_score": {
                "composite": 7.5,
                "reasoning_depth": 3,
                "data_grounding": 4,
                "risk_awareness": 10,
                "decision_consistent": True,
                "has_stop_loss": True,
                "has_price_target": True,
                "has_position_sizing": True,
            },
        }

    def test_load_result(self, tmp_path):
        result = self._make_result()
        filepath = tmp_path / "test_result.json"
        with open(filepath, "w") as f:
            json.dump(result, f)

        loaded = load_result(str(filepath))
        assert loaded["ticker"] == "AAPL"
        assert loaded["decision"] == "HOLD"

    def test_result_to_score(self):
        result = self._make_result("all_cloud", "HOLD")
        score = result_to_score(result)
        assert score.config_name == "all_cloud"
        assert score.decision == "HOLD"
        assert score.decision_consistent is True
        assert score.has_stop_loss is True
        assert score.composite_score > 0

    def test_result_to_score_missing_quality(self):
        """Handle result without quality_score block."""
        result = {
            "ticker": "AAPL",
            "trade_date": "2026-02-27",
            "provider": "anthropic",
            "hybrid_config": None,
            "decision": "BUY",
            "final_trade_decision_text": "",
        }
        score = result_to_score(result)
        assert score.config_name == "anthropic"
        assert score.decision == "BUY"
        assert score.reasoning_depth == 0
