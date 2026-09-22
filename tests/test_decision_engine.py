from __future__ import annotations

import time

import pytest

from strategy.decision_engine import (
    NO_TRADE_CONFLICTING_SIGNALS,
    NO_TRADE_HIGH_VOLATILITY,
    NO_TRADE_MINIMUM_DATA,
    NO_TRADE_NO_SETUP,
    NO_TRADE_POOR_ECONOMICS,
    NO_TRADE_REGIME_UNKNOWN,
    NO_TRADE_RISK,
    NO_TRADE_UNCONFIRMED,
    TRADE_BEARISH,
    TRADE_BULLISH,
    apply_economics_and_risk,
    evaluate_signal,
)
from strategy.edge_engine import Proposal, assess_level1_economics


def _base_kwargs(**overrides):
    kwargs = dict(symbol="R_100", timestamp=time.time(), sufficient_data=True,
                 regime="RANGE", volatility_regime="NORMAL",
                 bullish_score=0.0, bearish_score=0.0,
                 setup_threshold=55.0, regime_multiplier=1.0,
                 disable_on_high_vol=True, confirmed_direction=None,
                 confirmation_reason="")
    kwargs.update(overrides)
    return kwargs


def test_insufficient_data_is_no_trade():
    d = evaluate_signal(**_base_kwargs(sufficient_data=False))
    assert d.reason_code == NO_TRADE_MINIMUM_DATA
    assert not d.will_trade


def test_unknown_regime_is_hard_no_trade_even_with_a_strong_score():
    d = evaluate_signal(**_base_kwargs(regime="UNKNOWN", bullish_score=99.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    assert d.reason_code == NO_TRADE_REGIME_UNKNOWN


def test_high_volatility_refuses_regardless_of_score():
    d = evaluate_signal(**_base_kwargs(volatility_regime="HIGH_VOLATILITY",
                                       bullish_score=99.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    assert d.reason_code == NO_TRADE_HIGH_VOLATILITY


def test_high_volatility_gate_is_configurable_off():
    d = evaluate_signal(**_base_kwargs(volatility_regime="HIGH_VOLATILITY",
                                       disable_on_high_vol=False,
                                       bullish_score=99.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    assert d.will_trade


def test_neither_direction_qualifying_is_no_setup():
    d = evaluate_signal(**_base_kwargs(bullish_score=10.0, bearish_score=20.0))
    assert d.reason_code == NO_TRADE_NO_SETUP


def test_both_directions_qualifying_refuses_rather_than_picking_arbitrarily():
    d = evaluate_signal(**_base_kwargs(bullish_score=80.0, bearish_score=75.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    assert d.reason_code == NO_TRADE_CONFLICTING_SIGNALS


def test_qualifying_but_unconfirmed_is_no_trade():
    d = evaluate_signal(**_base_kwargs(bullish_score=80.0, confirmed_direction=None,
                                       confirmation_reason="awaiting confirmation"))
    assert d.reason_code == NO_TRADE_UNCONFIRMED


def test_qualifying_and_confirmed_bullish_trades():
    d = evaluate_signal(**_base_kwargs(bullish_score=80.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    assert d.decision == TRADE_BULLISH
    assert d.will_trade


def test_qualifying_and_confirmed_bearish_trades():
    d = evaluate_signal(**_base_kwargs(bearish_score=80.0,
                                       confirmed_direction="bearish",
                                       confirmation_reason="follow-through"))
    assert d.decision == TRADE_BEARISH


def test_regime_multiplier_raises_the_effective_threshold():
    """Section 15: a reversal against a strong trend needs stronger
    evidence than one inside a range."""
    kwargs = _base_kwargs(bullish_score=65.0, regime="TREND_UP",
                          regime_multiplier=1.4, confirmed_direction="bullish",
                          confirmation_reason="follow-through")
    d = evaluate_signal(**kwargs)
    assert d.threshold_used == pytest.approx(55.0 * 1.4)
    assert d.reason_code == NO_TRADE_NO_SETUP   # 65 < 77


def test_same_score_qualifies_in_range_regime():
    kwargs = _base_kwargs(bullish_score=65.0, regime="RANGE", regime_multiplier=1.0,
                          confirmed_direction="bullish",
                          confirmation_reason="follow-through")
    d = evaluate_signal(**kwargs)
    assert d.decision == TRADE_BULLISH


# --- Level 1 economics: no fabricated EV -----------------------------------

def _proposal(payout_multiple: float) -> Proposal:
    return Proposal(contract_type="CALL", symbol="R_100", stake=1.0,
                    payout=payout_multiple, ask_price=1.0, currency="USD",
                    proposal_id="p1", received_at=time.time())


def test_level1_economics_has_no_expected_value():
    """The central honesty requirement: Level 1 has no probability model,
    so it must not report a computed EV -- not 0.0, not anything else that
    could be mistaken for 'computed and found neutral'."""
    edge = assess_level1_economics(_proposal(1.9), min_payout_multiple=1.8)
    assert edge.expected_value is None


def test_payout_floor_met_and_not_met():
    good = assess_level1_economics(_proposal(1.9), min_payout_multiple=1.8)
    assert good.economically_sound
    bad = assess_level1_economics(_proposal(1.5), min_payout_multiple=1.8)
    assert not bad.economically_sound
    assert "below floor" in bad.reason


def test_apply_economics_and_risk_downgrades_on_poor_economics():
    d = evaluate_signal(**_base_kwargs(bullish_score=80.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    assert d.will_trade
    edge = assess_level1_economics(_proposal(1.5), min_payout_multiple=1.8)
    risk_ok = type("R", (), {"allowed": True, "reason": ""})()
    d2 = apply_economics_and_risk(d, edge_assessment=edge, risk_decision=risk_ok)
    assert not d2.will_trade
    assert d2.reason_code == NO_TRADE_POOR_ECONOMICS


def test_apply_economics_and_risk_downgrades_on_risk_refusal():
    d = evaluate_signal(**_base_kwargs(bearish_score=80.0,
                                       confirmed_direction="bearish",
                                       confirmation_reason="follow-through"))
    edge = assess_level1_economics(_proposal(1.9), min_payout_multiple=1.8)
    risk_no = type("R", (), {"allowed": False, "reason": "daily loss limit"})()
    d2 = apply_economics_and_risk(d, edge_assessment=edge, risk_decision=risk_no)
    assert d2.reason_code == NO_TRADE_RISK
    assert d2.explanation == "daily loss limit"


def test_apply_economics_and_risk_is_a_noop_on_an_already_no_trade_decision():
    d = evaluate_signal(**_base_kwargs())   # NO_TRADE_NO_SETUP
    edge = assess_level1_economics(_proposal(1.9), min_payout_multiple=1.8)
    risk_ok = type("R", (), {"allowed": True, "reason": ""})()
    d2 = apply_economics_and_risk(d, edge_assessment=edge, risk_decision=risk_ok)
    assert d2.reason_code == NO_TRADE_NO_SETUP


# --- research mode: the safety-critical case ---------------------------

def test_research_mode_downgrades_an_otherwise_perfect_trade_to_no_trade():
    """The exact scenario that matters: every other gate passed -- strong
    signal, good economics, risk clear -- and research mode still refuses.
    Checked LAST, after economics/risk, so the log shows what it would have
    traded rather than exiting before the rest of the pipeline is exercised."""
    d = evaluate_signal(**_base_kwargs(bullish_score=90.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    assert d.will_trade   # sanity: this WOULD trade outside research mode
    edge = assess_level1_economics(_proposal(1.95), min_payout_multiple=1.8)
    risk_ok = type("R", (), {"allowed": True, "reason": ""})()
    d2 = apply_economics_and_risk(d, edge_assessment=edge, risk_decision=risk_ok,
                                  research_mode=True)
    assert not d2.will_trade
    assert d2.reason_code == "NO_TRADE_RESEARCH_MODE"
    assert "buy() is never called" in d2.explanation


def test_research_mode_false_is_unaffected_default_behavior():
    """Regression check: the fix must not change demo/live behavior."""
    d = evaluate_signal(**_base_kwargs(bullish_score=90.0,
                                       confirmed_direction="bullish",
                                       confirmation_reason="follow-through"))
    edge = assess_level1_economics(_proposal(1.95), min_payout_multiple=1.8)
    risk_ok = type("R", (), {"allowed": True, "reason": ""})()
    d2 = apply_economics_and_risk(d, edge_assessment=edge, risk_decision=risk_ok,
                                  research_mode=False)
    assert d2.will_trade
