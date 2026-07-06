"""TRI-69 — golden-fixture tests for the scoring math (integrity doc §5 style).

Hand-computed scenarios on synthetic prices shaped exactly like
yf.download(auto_adjust=True) output (MultiIndex columns: field x ticker).
Negative/mutation-style checks: zeroing the cost must move the number
(proves cost is applied); a SELL must flip the sign; the slippage sweep
must be portfolio-level monotonically non-increasing.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import score_tri69  # noqa: E402


def make_prices():
    """10 trading days after 2026-03-13 for TICK and SPY.

    TICK: entry Open(t+1)=100, Close(t+10)=110 -> r=+10%
    SPY : entry Open(t+1)=100, Close(t+10)=102 -> r=+2%
    """
    days = pd.bdate_range("2026-03-16", periods=10)
    cols = pd.MultiIndex.from_product([["Open", "Close"], ["TICK", "SPY"]])
    df = pd.DataFrame(index=days, columns=cols, dtype=float)
    df[("Open", "TICK")] = [100 + i for i in range(10)]
    df[("Close", "TICK")] = [101 + i for i in range(9)] + [110.0]
    df[("Open", "SPY")] = [100 + 0.2 * i for i in range(10)]
    df[("Close", "SPY")] = [100.2 + 0.2 * i for i in range(9)] + [102.0]
    return df


class TestWindowReturn:
    def test_golden_long_window(self):
        prices = make_prices()
        r, entry, exitd = score_tri69.window_return(prices, "TICK", "2026-03-13", 10)
        assert entry == "2026-03-16" and exitd == "2026-03-27"
        assert r == pytest.approx(0.10)

    def test_unrealized_window_is_flagged_not_dropped(self):
        prices = make_prices()
        r, why, _ = score_tri69.window_return(prices, "TICK", "2026-03-13", 11)
        assert r is None and "unrealized" in why


class TestScore:
    ROWS = [{"ticker": "TICK", "date": "2026-03-13", "decision": "BUY"}]

    def test_golden_buy_net_excess(self):
        # gross = +1 * (0.10 - 0.02) = 0.08 ; net = 0.08 - 0.0020 = 0.078
        out = score_tri69.score(self.ROWS, make_prices(), 1.0, "T+10")
        assert out["directional_n"] == 1
        assert out["mean_of_date_means"] == pytest.approx(0.078)
        assert out["hit_rate"] == 1.0

    def test_sell_flips_sign(self):
        rows = [{"ticker": "TICK", "date": "2026-03-13", "decision": "SELL"}]
        out = score_tri69.score(rows, make_prices(), 1.0, "T+10")
        assert out["mean_of_date_means"] == pytest.approx(-0.082)
        assert out["hit_rate"] == 0.0

    def test_hold_excluded(self):
        rows = self.ROWS + [{"ticker": "TICK", "date": "2026-03-13", "decision": "HOLD"}]
        out = score_tri69.score(rows, make_prices(), 1.0, "T+10")
        assert out["directional_n"] == 1  # HOLD contributes nothing

    def test_cost_actually_applied(self):
        """Mutation-style negative: zero the cost -> the number must move."""
        with_cost = score_tri69.score(self.ROWS, make_prices(), 1.0, "T+10")
        no_cost = score_tri69.score(self.ROWS, make_prices(), 0.0, "T+10")
        assert no_cost["mean_of_date_means"] > with_cost["mean_of_date_means"]
        assert no_cost["mean_of_date_means"] == pytest.approx(0.08)

    def test_slippage_sweep_monotone_non_increasing(self):
        prices = make_prices()
        means = [score_tri69.score(self.ROWS, prices, m, "T+10")["mean_of_date_means"]
                 for m in (1.0, 2.0, 3.0)]
        assert means[0] > means[1] > means[2]


class TestLongOnlyView:
    """Integrity §5: long-only — SELL on an unheld name is a P&L no-op."""

    def test_sell_is_noop_buy_is_long_net_of_cost(self):
        rows = [
            {"ticker": "TICK", "date": "2026-03-13", "decision": "BUY"},
            {"ticker": "SPY",  "date": "2026-03-13", "decision": "SELL"},
        ]
        out = score_tri69.long_only_view(rows, make_prices(), 1.0, "T+10")
        d = out["per_date"]["2026-03-13"]
        # strategy: mean(BUY TICK 0.10-0.002, SELL no-op 0.0) = 0.049
        assert d["strategy"] == pytest.approx(0.049)
        # buy-and-hold both names net of cost: mean(0.098, 0.018) = 0.058
        assert d["buy_and_hold"] == pytest.approx(0.058)


class TestSignFlip:
    def test_all_positive_five_dates_exact_min_p(self):
        assert score_tri69.sign_flip_pvalue([0.01] * 5) == pytest.approx(1 / 32)

    def test_eight_dates_power_floor(self):
        """Pre-registered D=8: floor 1/256; one flat-negative date of equal
        magnitude still clears alpha=0.05 (p = 9/256)."""
        assert score_tri69.sign_flip_pvalue([0.01] * 8) == pytest.approx(1 / 256)
        p = score_tri69.sign_flip_pvalue([0.01] * 7 + [-0.01])
        assert p == pytest.approx(9 / 256)
        assert p < 0.05

    def test_symmetric_data_not_significant(self):
        p = score_tri69.sign_flip_pvalue([0.01, -0.01, 0.02, -0.02, 0.0])
        assert p > 0.4

    def test_single_negative_reduces_significance(self):
        p_all = score_tri69.sign_flip_pvalue([0.02, 0.02, 0.02, 0.02, 0.02])
        p_one_neg = score_tri69.sign_flip_pvalue([0.02, 0.02, 0.02, 0.02, -0.02])
        assert p_one_neg > p_all


class TestBaseRate:
    def test_base_rate_is_best_constant_strategy(self):
        # TICK beats SPY -> all-BUY accuracy 1.0 on this one-decision set
        p0 = score_tri69.base_rate(
            [{"ticker": "TICK", "date": "2026-03-13", "decision": "SELL"}],
            make_prices(), "T+10")
        assert p0 == 1.0  # max(frac_up=1.0, 1-frac_up=0.0)
