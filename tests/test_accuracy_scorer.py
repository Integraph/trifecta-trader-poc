"""Tests for src/accuracy/scorer.py (Task 015).

All tests use deterministic price scenarios so assertions are exact.

Covers:
- BUY direction correct when price goes up
- BUY direction wrong when price goes down
- SELL direction correct when price goes down
- SELL direction wrong when price goes up
- HOLD always direction_correct=1
- Target hit for BUY (high reached target)
- Target not hit for BUY (high below target)
- Target hit for SELL (low reached target)
- Target not hit for SELL (low above target)
- Stop hit for BUY (low breached stop)
- Stop not hit for BUY (low above stop)
- Stop hit for SELL (high breached stop)
- Stop not hit for SELL (high below stop)
- target_hit_first: target before stop (chronological)
- target_hit_first: stop before target
- target_hit_first: both on same day → False
- target_hit_first: neither hit → False
- Return calculations for BUY and SELL (positive = profitable)
- max_favorable_pct for BUY and SELL
- max_adverse_pct for BUY and SELL (always negative)
- Missing entry_price → direction/return fields absent from result
- Missing stop/target → hit fields absent from result
- score_outcome() integrates all helpers correctly
"""

import pytest
from src.accuracy.scorer import (
    AccuracyScorer,
    _direction_correct,
    _target_hit,
    _stop_hit,
    _target_hit_first,
    _calculate_return,
    _max_favorable,
    _max_adverse,
)


# ── 1. _direction_correct() ───────────────────────────────────────────────────

class TestDirectionCorrect:

    def test_buy_correct_when_price_up(self):
        assert _direction_correct("BUY", entry=100.0, actual=105.0) is True

    def test_buy_wrong_when_price_down(self):
        assert _direction_correct("BUY", entry=100.0, actual=95.0) is False

    def test_buy_wrong_when_price_unchanged(self):
        assert _direction_correct("BUY", entry=100.0, actual=100.0) is False

    def test_sell_correct_when_price_down(self):
        assert _direction_correct("SELL", entry=100.0, actual=95.0) is True

    def test_sell_wrong_when_price_up(self):
        assert _direction_correct("SELL", entry=100.0, actual=105.0) is False

    def test_hold_always_correct(self):
        assert _direction_correct("HOLD", entry=100.0, actual=95.0)  is True
        assert _direction_correct("HOLD", entry=100.0, actual=105.0) is True


# ── 2. _target_hit() ─────────────────────────────────────────────────────────

class TestTargetHit:

    def test_buy_target_hit_when_high_reaches(self):
        assert _target_hit("BUY", price_target=110.0, high_t10=112.0, low_t10=95.0) is True

    def test_buy_target_not_hit_when_high_below(self):
        assert _target_hit("BUY", price_target=110.0, high_t10=108.0, low_t10=95.0) is False

    def test_buy_target_hit_exactly_at_boundary(self):
        assert _target_hit("BUY", price_target=110.0, high_t10=110.0, low_t10=95.0) is True

    def test_sell_target_hit_when_low_reaches(self):
        assert _target_hit("SELL", price_target=90.0, high_t10=105.0, low_t10=88.0) is True

    def test_sell_target_not_hit_when_low_above(self):
        assert _target_hit("SELL", price_target=90.0, high_t10=105.0, low_t10=92.0) is False


# ── 3. _stop_hit() ────────────────────────────────────────────────────────────

class TestStopHit:

    def test_buy_stop_hit_when_low_breaches(self):
        assert _stop_hit("BUY", stop_loss=90.0, high_t10=105.0, low_t10=88.0) is True

    def test_buy_stop_not_hit_when_low_above(self):
        assert _stop_hit("BUY", stop_loss=90.0, high_t10=105.0, low_t10=92.0) is False

    def test_buy_stop_hit_exactly_at_stop(self):
        assert _stop_hit("BUY", stop_loss=90.0, high_t10=105.0, low_t10=90.0) is True

    def test_sell_stop_hit_when_high_breaches(self):
        assert _stop_hit("SELL", stop_loss=110.0, high_t10=112.0, low_t10=95.0) is True

    def test_sell_stop_not_hit_when_high_below(self):
        assert _stop_hit("SELL", stop_loss=110.0, high_t10=108.0, low_t10=95.0) is False


# ── 4. _target_hit_first() ────────────────────────────────────────────────────

class TestTargetHitFirst:

    def test_buy_target_before_stop(self):
        # Day 1: neither. Day 2: target hit. Day 3: stop hit.
        highs = [102.0, 112.0, 105.0]
        lows  = [98.0,  100.0,  88.0]
        assert _target_hit_first("BUY", price_target=110.0, stop_loss=90.0,
                                  daily_highs=highs, daily_lows=lows) is True

    def test_buy_stop_before_target(self):
        # Day 1: stop hit. Day 2: target hit.
        highs = [102.0, 112.0]
        lows  = [88.0,  100.0]
        assert _target_hit_first("BUY", price_target=110.0, stop_loss=90.0,
                                  daily_highs=highs, daily_lows=lows) is False

    def test_buy_both_same_day_returns_false(self):
        # Day 1: both high reaches target AND low breaches stop
        highs = [112.0]
        lows  = [88.0]
        assert _target_hit_first("BUY", price_target=110.0, stop_loss=90.0,
                                  daily_highs=highs, daily_lows=lows) is False

    def test_buy_neither_hit_returns_false(self):
        highs = [105.0, 107.0, 106.0]
        lows  = [98.0,  99.0,  97.0]
        assert _target_hit_first("BUY", price_target=115.0, stop_loss=85.0,
                                  daily_highs=highs, daily_lows=lows) is False

    def test_sell_target_before_stop(self):
        # Day 1: neither. Day 2: sell target hit (low reaches target). Day 3: sell stop hit.
        highs = [102.0, 100.0, 112.0]
        lows  = [98.0,  88.0,  95.0]
        assert _target_hit_first("SELL", price_target=90.0, stop_loss=110.0,
                                  daily_highs=highs, daily_lows=lows) is True

    def test_sell_stop_before_target(self):
        # Day 1: sell stop hit (high breaches stop). Day 2: sell target hit.
        highs = [112.0, 100.0]
        lows  = [98.0,  88.0]
        assert _target_hit_first("SELL", price_target=90.0, stop_loss=110.0,
                                  daily_highs=highs, daily_lows=lows) is False


# ── 5. _calculate_return() ────────────────────────────────────────────────────

class TestCalculateReturn:

    def test_buy_positive_return_when_price_up(self):
        r = _calculate_return("BUY", entry=100.0, actual=110.0)
        assert abs(r - 10.0) < 0.001

    def test_buy_negative_return_when_price_down(self):
        r = _calculate_return("BUY", entry=100.0, actual=90.0)
        assert abs(r - (-10.0)) < 0.001

    def test_sell_positive_return_when_price_down(self):
        r = _calculate_return("SELL", entry=100.0, actual=90.0)
        assert abs(r - 10.0) < 0.001

    def test_sell_negative_return_when_price_up(self):
        r = _calculate_return("SELL", entry=100.0, actual=110.0)
        assert abs(r - (-10.0)) < 0.001

    def test_zero_entry_returns_zero(self):
        assert _calculate_return("BUY", entry=0.0, actual=100.0) == 0.0


# ── 6. _max_favorable() ───────────────────────────────────────────────────────

class TestMaxFavorable:

    def test_buy_favorable_is_upside(self):
        r = _max_favorable("BUY", entry=100.0, high_t10=115.0, low_t10=90.0)
        assert abs(r - 15.0) < 0.001

    def test_sell_favorable_is_downside(self):
        r = _max_favorable("SELL", entry=100.0, high_t10=115.0, low_t10=90.0)
        assert abs(r - 10.0) < 0.001

    def test_always_non_negative(self):
        # Even if price moved against signal, favorable should be 0 or positive
        r = _max_favorable("BUY", entry=100.0, high_t10=100.0, low_t10=95.0)
        assert r >= 0


# ── 7. _max_adverse() ─────────────────────────────────────────────────────────

class TestMaxAdverse:

    def test_buy_adverse_is_downside_negative(self):
        r = _max_adverse("BUY", entry=100.0, high_t10=115.0, low_t10=90.0)
        assert r < 0
        assert abs(r - (-10.0)) < 0.001

    def test_sell_adverse_is_upside_negative(self):
        r = _max_adverse("SELL", entry=100.0, high_t10=115.0, low_t10=90.0)
        assert r < 0
        assert abs(r - (-15.0)) < 0.001


# ── 8. score_outcome() integration ───────────────────────────────────────────

class TestScoreOutcome:

    def _buy_outcome(self, p_t1=105.0, p_t5=108.0, p_t10=112.0,
                     high10=115.0, low10=92.0,
                     target=110.0, stop=90.0):
        return {
            "decision":    "BUY",
            "entry_price": 100.0,
            "stop_loss":   stop,
            "price_target": target,
            "price_t1":    p_t1,
            "price_t5":    p_t5,
            "price_t10":   p_t10,
            "high_t10":    high10,
            "low_t10":     low10,
        }

    def test_buy_all_direction_correct(self):
        scorer = AccuracyScorer()
        scores = scorer.score_outcome(self._buy_outcome())
        assert scores["direction_correct_t1"]  == 1
        assert scores["direction_correct_t5"]  == 1
        assert scores["direction_correct_t10"] == 1

    def test_buy_target_hit(self):
        scorer = AccuracyScorer()
        scores = scorer.score_outcome(self._buy_outcome(high10=115.0, target=110.0))
        assert scores["target_hit"] == 1

    def test_buy_stop_not_hit(self):
        scorer = AccuracyScorer()
        scores = scorer.score_outcome(self._buy_outcome(low10=92.0, stop=90.0))
        assert scores["stop_hit"] == 0

    def test_hold_direction_always_correct(self):
        scorer = AccuracyScorer()
        outcome = {
            "decision":    "HOLD",
            "entry_price": 100.0,
            "stop_loss":   None,
            "price_target": None,
            "price_t1":    90.0,   # Price went down — HOLD still "correct"
            "price_t5":    85.0,
            "price_t10":   80.0,
            "high_t10":    102.0,
            "low_t10":     80.0,
        }
        scores = scorer.score_outcome(outcome)
        assert scores["direction_correct_t1"]  == 1
        assert scores["direction_correct_t5"]  == 1
        assert scores["direction_correct_t10"] == 1

    def test_missing_entry_price_skips_direction_fields(self):
        scorer  = AccuracyScorer()
        outcome = {
            "decision":    "BUY",
            "entry_price": None,
            "stop_loss":   90.0,
            "price_target": 110.0,
            "price_t1":    105.0,
            "price_t5":    108.0,
            "price_t10":   112.0,
            "high_t10":    115.0,
            "low_t10":     92.0,
        }
        scores = scorer.score_outcome(outcome)
        assert "direction_correct_t1"  not in scores
        assert "direction_correct_t5"  not in scores
        assert "direction_correct_t10" not in scores
        assert "return_t1_pct"         not in scores

    def test_missing_stop_target_skips_hit_fields(self):
        scorer  = AccuracyScorer()
        outcome = {
            "decision":    "BUY",
            "entry_price": 100.0,
            "stop_loss":   None,
            "price_target": None,
            "price_t1":    105.0,
            "price_t5":    108.0,
            "price_t10":   112.0,
            "high_t10":    115.0,
            "low_t10":     92.0,
        }
        scores = scorer.score_outcome(outcome)
        assert "target_hit"      not in scores
        assert "stop_hit"        not in scores
        assert "target_hit_first" not in scores

    def test_target_hit_first_uses_daily_arrays(self):
        scorer  = AccuracyScorer()
        outcome = self._buy_outcome(target=110.0, stop=90.0, high10=115.0, low10=88.0)
        # Day-by-day: target hit on day 2, stop hit on day 3
        daily_highs = [103.0, 112.0, 102.0]
        daily_lows  = [98.0,  100.0,  87.0]
        scores = scorer.score_outcome(outcome, daily_highs=daily_highs, daily_lows=daily_lows)
        assert scores["target_hit_first"] == 1

    def test_return_t5_pct_positive_for_profitable_buy(self):
        scorer = AccuracyScorer()
        scores = scorer.score_outcome(self._buy_outcome(p_t5=110.0))
        assert scores["return_t5_pct"] > 0

    def test_max_adverse_is_negative(self):
        scorer = AccuracyScorer()
        scores = scorer.score_outcome(self._buy_outcome(low10=88.0))
        assert scores["max_adverse_pct"] < 0
