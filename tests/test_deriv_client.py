"""
Tests for what's actually DIFFERENT in this repo's deriv/client.py relative
to the sibling Even/Odd bot's proven connection layer: Rise/Fall
availability checks, CALL/PUT proposals, and float (not text) tick prices.
The reconnect/rate-limit/idempotent-buy internals are unchanged ported code
already covered by that sibling suite and are not re-tested here.
"""
from __future__ import annotations

import asyncio

import pytest

from deriv.client import AUTH_LEGACY, DerivAPIError, DerivClient


def _client(**kw) -> DerivClient:
    kw.setdefault("auth_mode", AUTH_LEGACY)
    return DerivClient(app_id="1", api_token="tok", ws_url="wss://x", **kw)


def test_verify_rise_fall_confirms_when_both_types_present():
    async def run():
        c = _client()

        async def ok_send(payload):
            return {"contracts_for": {"available": [
                {"contract_type": "CALL"}, {"contract_type": "PUT"}]}}

        c._send = ok_send
        ok, why = await c.verify_rise_fall_available("R_100")
        assert ok
        assert "confirmed available" in why
    asyncio.run(run())


def test_verify_rise_fall_distinguishes_request_failure_from_confirmed_unavailable():
    async def run():
        c = _client()

        async def failing_send(payload):
            raise DerivAPIError("RateLimit", "too many requests")

        c._send = failing_send
        ok, why = await c.verify_rise_fall_available("R_100")
        assert not ok
        assert "REQUEST FAILED" in why
        assert "never confirmed either way" in why
    asyncio.run(run())


def test_verify_rise_fall_reports_confirmed_missing_types():
    async def run():
        c = _client()

        async def ok_send(payload):
            return {"contracts_for": {"available": [{"contract_type": "DIGITEVEN"}]}}

        c._send = ok_send
        ok, why = await c.verify_rise_fall_available("R_100")
        assert not ok
        assert "CONFIRMED missing" in why
        assert "CALL" in why and "PUT" in why
    asyncio.run(run())


def test_proposal_sends_call_or_put_with_no_barrier():
    async def run():
        c = _client()
        sent: dict = {}

        async def fake_send(payload):
            sent.update(payload)
            return {"proposal": {"id": "p1", "payout": 1.9}}

        c._send = fake_send
        await c.proposal(symbol="R_100", contract_type="CALL", amount=1.0,
                         currency="USD")
        assert sent["contract_type"] == "CALL"
        assert sent["underlying_symbol"] == "R_100"
        assert "symbol" not in sent
        assert "barrier" not in sent
    asyncio.run(run())


def test_tick_price_is_a_real_float_not_text():
    """The one deliberate departure from the sibling Even/Odd client: candle
    OHLC needs real arithmetic, so quote is parsed once here rather than
    kept as text."""
    async def run():
        c = _client()
        c._tick_queues["R_100"] = asyncio.Queue(maxsize=10)
        c._route_tick({"tick": {"symbol": "R_100", "epoch": 1, "quote": "1234.50"}})
        tick = c._tick_queues["R_100"].get_nowait()
        assert isinstance(tick.price, float)
        assert tick.price == 1234.50
    asyncio.run(run())


def test_malformed_quote_parses_to_zero_rather_than_raising():
    async def run():
        c = _client()
        c._tick_queues["R_100"] = asyncio.Queue(maxsize=10)
        c._route_tick({"tick": {"symbol": "R_100", "epoch": 1, "quote": None}})
        tick = c._tick_queues["R_100"].get_nowait()
        assert tick.price == 0.0
    asyncio.run(run())
