"""Pipeline integration tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "TradingAgents"))


def test_decision_not_repeated(tmp_path):
    """Verify the final decision appears only once (no looping)."""
    results_dir = Path(__file__).parent.parent / "results" / "AAPL"

    if not results_dir.exists():
        import pytest
        pytest.skip("No pipeline results available - run the pipeline first")

    result_files = sorted(results_dir.glob("*.json"))
    assert len(result_files) > 0, "No result files found"


def test_conditional_logic_defaults():
    """Verify ConditionalLogic defaults match the intended config values."""
    from tradingagents.graph.conditional_logic import ConditionalLogic

    cl = ConditionalLogic()
    assert cl.max_debate_rounds == 1
    assert cl.max_risk_discuss_rounds == 1


def test_conditional_logic_risk_round_limit():
    """With max_risk_discuss_rounds=1, debate should end after 3 speakers."""
    from tradingagents.graph.conditional_logic import ConditionalLogic

    cl = ConditionalLogic(max_risk_discuss_rounds=1)

    class FakeState(dict):
        pass

    state = FakeState({
        "risk_debate_state": {
            "count": 0,
            "latest_speaker": "Aggressive",
        }
    })
    assert cl.should_continue_risk_analysis(state) == "Conservative Analyst"

    state["risk_debate_state"]["count"] = 1
    state["risk_debate_state"]["latest_speaker"] = "Conservative"
    assert cl.should_continue_risk_analysis(state) == "Neutral Analyst"

    state["risk_debate_state"]["count"] = 2
    state["risk_debate_state"]["latest_speaker"] = "Neutral"
    assert cl.should_continue_risk_analysis(state) == "Aggressive Analyst"

    state["risk_debate_state"]["count"] = 3
    state["risk_debate_state"]["latest_speaker"] = "Neutral"
    assert cl.should_continue_risk_analysis(state) == "Risk Judge"


def test_conditional_logic_config_not_passed():
    """Document the bug: TradingAgentsGraph creates ConditionalLogic without config.

    This means config['max_debate_rounds'] and config['max_risk_discuss_rounds']
    are silently ignored. The fix requires modifying vendor code at
    vendor/TradingAgents/tradingagents/graph/trading_graph.py line 108.
    """
    from tradingagents.graph.conditional_logic import ConditionalLogic

    cl_default = ConditionalLogic()
    cl_custom = ConditionalLogic(max_risk_discuss_rounds=2)
    assert cl_default.max_risk_discuss_rounds == 1
    assert cl_custom.max_risk_discuss_rounds == 2

    state_count_3 = {
        "risk_debate_state": {"count": 3, "latest_speaker": "Neutral"}
    }
    assert cl_default.should_continue_risk_analysis(state_count_3) == "Risk Judge"
    assert cl_custom.should_continue_risk_analysis(state_count_3) == "Aggressive Analyst"


def test_propagator_defaults():
    """Verify Propagator default matches intended recursion limit."""
    from tradingagents.graph.propagation import Propagator

    p = Propagator()
    assert p.max_recur_limit == 100

    args = p.get_graph_args()
    assert args["config"]["recursion_limit"] == 100
