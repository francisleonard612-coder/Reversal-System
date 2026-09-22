"""
Tests for execution/executor.py, focused on the research-mode backstop
(spec Section 39: "Never silently switch to live trading").

The scenario that matters is not "does the normal path work" -- it's "if
the first gate (strategy.decision_engine.apply_economics_and_risk) is
somehow bypassed or buggy, does buy() still not get called." That's tested
here by monkeypatching the first gate to NOT downgrade the decision, and
confirming execute_trade refuses anyway.
"""
from __future__ import annotations

import asyncio
import time

import pytest

import execution.executor as executor_module
from execution.executor import ExecutionError, execute_trade
from strategy.decision_engine import Decision, TRADE_BULLISH


class _FakeClient:
    def __init__(self):
        self.buy_called = False

    async def proposal(self, **kwargs):
        return {"id": "prop-1", "ask_price": 1.0, "payout": 1.95}

    async def buy(self, *args, **kwargs):
        self.buy_called = True
        return {"contract_id": 999, "buy_price": 1.0, "start_spot": 100.0}


class _FakeDb:
    def record_trade_open(self, **kwargs):
        return 1

    def log_event(self, *args, **kwargs):
        pass


class _FakeRisk:
    def can_trade(self, stake):
        return type("R", (), {"allowed": True, "reason": ""})()


def _decision() -> Decision:
    return Decision(symbol="R_100", timestamp=time.time(), decision=TRADE_BULLISH,
                    reason_code=TRADE_BULLISH, explanation="", bullish_score=90.0,
                    confirmed=True)


def test_research_mode_never_calls_buy_via_the_normal_path():
    async def run():
        client = _FakeClient()
        d = await execute_trade(
            client, _FakeDb(), decision=_decision(), signal_id=1, symbol="R_100",
            currency="USD", duration=5, duration_unit="t", stake=1.0,
            min_payout_multiple=1.8, risk_manager=_FakeRisk(), research_mode=True)
        decision, edge, buy_resp = d
        assert not client.buy_called
        assert buy_resp is None
        assert decision.reason_code == "NO_TRADE_RESEARCH_MODE"
    asyncio.run(run())


def test_demo_mode_calls_buy_normally():
    """Regression: the research-mode fix must not block real execution when
    research_mode is False."""
    async def run():
        client = _FakeClient()
        decision, edge, buy_resp = await execute_trade(
            client, _FakeDb(), decision=_decision(), signal_id=1, symbol="R_100",
            currency="USD", duration=5, duration_unit="t", stake=1.0,
            min_payout_multiple=1.8, risk_manager=_FakeRisk(), research_mode=False)
        assert client.buy_called
        assert buy_resp is not None
        assert decision.will_trade
    asyncio.run(run())


def test_backstop_refuses_even_if_the_first_gate_is_bypassed(monkeypatch):
    """THE test that matters. Simulates a bug or bypass in
    apply_economics_and_risk (the first gate) by monkeypatching it to NOT
    downgrade the decision even though research_mode=True -- and confirms
    execute_trade's own independent check still refuses to call buy()."""
    async def broken_apply_economics_and_risk(decision, *, edge_assessment,
                                               risk_decision, research_mode=False):
        # Deliberately ignores research_mode -- simulates the first gate
        # failing to do its job.
        return decision

    import strategy.decision_engine as de
    monkeypatch.setattr(de, "apply_economics_and_risk",
                        lambda decision, **kw: decision)

    async def run():
        client = _FakeClient()
        with pytest.raises(ExecutionError, match="refusing to call buy"):
            await execute_trade(
                client, _FakeDb(), decision=_decision(), signal_id=1,
                symbol="R_100", currency="USD", duration=5, duration_unit="t",
                stake=1.0, min_payout_multiple=1.8, risk_manager=_FakeRisk(),
                research_mode=True)
        assert not client.buy_called, (
            "buy() must never be called even when the first gate is bypassed")
    asyncio.run(run())
