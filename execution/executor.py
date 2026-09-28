"""
Execution engine (spec Section 38). CALL/PUT (Rise/Fall) only -- this build
does not support Higher/Lower or other barrier contracts.

Before execution verifies exactly the list Section 38 asks for: instrument,
contract type, duration, stake, payout, account mode, data freshness, risk
approval, signal timestamp. Reuses the sibling Even/Odd bot's idempotency
and no-retry-on-ambiguous-buy discipline (deriv/client.py rule 4) --
nothing about that rule is digit-specific.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from deriv.client import BuyAmbiguousError, DerivAPIError, DerivClient
from strategy.decision_engine import Decision
from strategy.edge_engine import (EdgeAssessment, Proposal, assess_level1_economics,
                                  assess_level2_economics)


class ExecutionError(RuntimeError):
    pass


async def get_proposal(client: DerivClient, *, symbol: str, direction: str,
                       stake: float, currency: str, duration: int,
                       duration_unit: str) -> Proposal:
    contract_type = "CALL" if direction == "bullish" else "PUT"
    raw = await client.proposal(symbol=symbol, contract_type=contract_type,
                                amount=stake, currency=currency,
                                duration=duration, duration_unit=duration_unit)
    if "id" not in raw or "ask_price" not in raw:
        raise ExecutionError(f"malformed proposal response: {raw}")
    return Proposal(
        contract_type=contract_type, symbol=symbol, stake=stake,
        payout=float(raw.get("payout", 0.0)), ask_price=float(raw["ask_price"]),
        currency=currency, proposal_id=raw["id"], received_at=time.time())


async def execute_trade(client: DerivClient, db, *, decision: Decision,
                        signal_id: int, symbol: str, currency: str,
                        duration: int, duration_unit: str, stake: float,
                        min_payout_multiple: float, risk_manager,
                        research_mode: bool = False,
                        max_proposal_age_seconds: float = 5.0,
                        level2_model=None, level2_features: dict | None = None,
                        level2_min_ev: float = 0.0,
                        level2_require_lower_bound: bool = True,
                        ) -> tuple[Decision, EdgeAssessment | None, dict | None]:
    """Full flow: proposal -> economics/risk check -> buy -> record. Returns
    the (possibly downgraded to NO_TRADE) decision, the edge assessment for
    logging, and the raw buy response (None if never bought).

    `research_mode` is checked in TWO INDEPENDENT PLACES: once in
    strategy.decision_engine.apply_economics_and_risk (below), and again
    here, immediately before the buy() call, as a hard backstop that does
    not trust the decision object's own state. A bug that let a wrongly-
    flagged Decision reach this function would still be caught here --
    this is deliberately not DRY, because the one thing worth duplicating
    is "does this function place a real order."
    """
    from strategy.decision_engine import apply_economics_and_risk

    direction = "bullish" if decision.decision == "TRADE_BULLISH" else "bearish"

    try:
        proposal = await get_proposal(
            client, symbol=symbol, direction=direction, stake=stake,
            currency=currency, duration=duration, duration_unit=duration_unit)
    except (DerivAPIError, ExecutionError) as exc:
        decision.decision = "NO_TRADE"
        decision.reason_code = "NO_TRADE_BAD_PROPOSAL"
        decision.explanation = f"proposal request failed: {exc}"
        return decision, None, None

    if level2_model is not None and level2_features is not None:
        # A promoted model is active: price the trade with a real P(win)
        # against the real quoted payout (true EV), not just a payout floor.
        raw_p, cal_p, lower_p = level2_model.predict(level2_features)
        edge = assess_level2_economics(
            proposal, probability=cal_p, probability_lower=lower_p,
            min_payout_multiple=min_payout_multiple, min_ev=level2_min_ev,
            require_lower_bound_edge=level2_require_lower_bound)
        try:
            db.record_prediction(
                signal_id=signal_id, symbol=symbol, model_id=level2_model.model_id,
                model_version=level2_model.version, probability=raw_p,
                calibrated_probability=cal_p, probability_lower=lower_p,
                payout_multiple=proposal.payout_multiple, expected_value=edge.expected_value)
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not block a decision
            db.log_event("WARNING", "level2", f"could not record prediction: {exc}")
    else:
        edge = assess_level1_economics(proposal, min_payout_multiple=min_payout_multiple)
    risk_decision = risk_manager.can_trade(stake)
    decision = apply_economics_and_risk(decision, edge_assessment=edge,
                                        risk_decision=risk_decision,
                                        research_mode=research_mode,
                                        max_proposal_age_seconds=max_proposal_age_seconds)
    if not decision.will_trade:
        return decision, edge, None

    if research_mode:
        # Should be unreachable -- apply_economics_and_risk already downgrades
        # to NO_TRADE_RESEARCH_MODE above. Reaching here means that layer was
        # bypassed or is wrong, and this refuses anyway rather than trust it.
        raise ExecutionError(
            "refusing to call buy(): research_mode=True but the decision "
            "still reads as tradeable -- this should never happen and "
            "indicates a bug in apply_economics_and_risk, not a reason to proceed")

    idempotency_key = str(uuid.uuid4())
    buy_started = time.time()
    try:
        buy_resp = await client.buy(proposal.proposal_id, proposal.ask_price,
                                    idempotency_key=idempotency_key)
    except BuyAmbiguousError as exc:
        db.log_event("ERROR", "execution",
                     f"buy outcome unknown, reconciling via portfolio: {exc}",
                     {"idempotency_key": idempotency_key, "symbol": symbol})
        # Previously the trail ended here: a buy that DID go through was
        # never recorded, never settled, and its symbol was freed to trade
        # again on top of a live contract. Look for it instead (never retry).
        buy_resp = await reconcile_ambiguous_buy(
            client, db, symbol=symbol, contract_type=proposal.contract_type,
            since=buy_started - 5)
        if buy_resp is None:
            decision.decision = "NO_TRADE"
            decision.reason_code = "NO_TRADE_BUY_AMBIGUOUS"
            decision.explanation = f"{exc} -- no matching contract found in portfolio"
            return decision, edge, None
        db.log_event("WARNING", "execution",
                     f"ambiguous buy DID go through: contract {buy_resp['contract_id']}",
                     {"idempotency_key": idempotency_key, "symbol": symbol})
    except DerivAPIError as exc:
        decision.decision = "NO_TRADE"
        decision.reason_code = "NO_TRADE_BUY_REJECTED"
        decision.explanation = str(exc)
        return decision, edge, None

    contract_id = buy_resp.get("contract_id")
    if contract_id is None:
        raise ExecutionError(f"buy succeeded but no contract_id returned: {buy_resp}")

    db.record_trade_open(
        signal_id=signal_id, symbol=symbol, contract_id=contract_id,
        idempotency_key=idempotency_key, contract_type=proposal.contract_type,
        stake=stake, payout=proposal.payout,
        buy_price=float(buy_resp.get("buy_price", proposal.ask_price)),
        entry_spot=float(buy_resp.get("start_spot", 0.0) or 0.0))
    # Previously never called: trades_today, cooldown and the concurrency
    # limit all read counters that nothing ever incremented.
    if hasattr(risk_manager, "register_open"):
        risk_manager.register_open(stake)

    return decision, edge, buy_resp


async def reconcile_ambiguous_buy(client, db, *, symbol: str, contract_type: str,
                                  since: float, attempts: int = 3) -> dict | None:
    """Finds a contract bought by an ambiguous (timed-out) buy call: same
    symbol and type, purchased after `since`, not already in our journal."""
    try:
        known = db.known_contract_ids()
    except Exception:  # noqa: BLE001
        known = set()
    for attempt in range(attempts):
        try:
            contracts = await client.portfolio()
        except Exception:  # noqa: BLE001
            await asyncio.sleep(2)
            continue
        for c in contracts:
            cid = c.get("contract_id")
            sym = c.get("symbol") or c.get("underlying_symbol") or c.get("underlying")
            if (cid is not None and cid not in known and sym == symbol
                    and str(c.get("contract_type", "")).upper() == contract_type
                    and float(c.get("purchase_time") or 0) >= since):
                return {"contract_id": cid, "buy_price": c.get("buy_price"),
                        "start_spot": c.get("entry_spot") or c.get("entry_tick")}
        return None
    return None


async def monitor_settlement(client: DerivClient, db, risk_manager, staking,
                             contract_id: int, *, expected_seconds: float | None = None,
                             grace_seconds: float = 60.0, logger=None) -> dict:
    """Waits for a contract's real result and books it exactly once.

    PREVIOUSLY: the wait was a fixed 120s, and on silence the trade was
    booked as a LOSS with pnl 0 -- inflating the losing streak, corrupting
    win rate, and releasing the symbol while the contract could still be
    live. Contract streams are also not re-subscribed after a reconnect, so
    any disconnect during a trade produced exactly that fake loss.

    NOW: the stream wait lasts the contract's own length plus a grace
    period; if the result still hasn't arrived, it polls
    proposal_open_contract with backoff until it has. Nothing is booked
    until Deriv says the contract is sold, and db.record_trade_result is
    idempotent, so risk/staking counters move exactly once.
    """
    timeout = (expected_seconds or 0.0) + grace_seconds
    poc = {}
    try:
        poc = await client.wait_for_settlement(contract_id, timeout=max(timeout, 30.0))
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("contract %s: settlement stream failed (%s), polling", contract_id, exc)
    delay = 5.0
    while not poc.get("is_sold"):
        try:
            poc = await client.proposal_open_contract(contract_id) or {}
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("contract %s: poll failed (%s)", contract_id, exc)
            poc = {}
        if poc.get("is_sold"):
            break
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60.0)

    profit = float(poc.get("profit", 0.0) or 0.0)
    status = str(poc.get("status") or "").lower()
    won = (status == "won") if status in ("won", "lost") else profit > 0
    exit_spot = poc.get("exit_tick") or poc.get("exit_spot")
    first = db.record_trade_result(contract_id, won=won, pnl=profit,
                                   exit_spot=float(exit_spot) if exit_spot else None)
    if first is not False:   # True, or None from an older backend without the return value
        risk_manager.register_result(profit)
        staking.register_result(won)
    return poc
