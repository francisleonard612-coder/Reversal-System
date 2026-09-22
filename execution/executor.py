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

import time
import uuid

from deriv.client import BuyAmbiguousError, DerivAPIError, DerivClient
from strategy.decision_engine import Decision
from strategy.edge_engine import EdgeAssessment, Proposal, assess_level1_economics


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
                        research_mode: bool = False
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

    edge = assess_level1_economics(proposal, min_payout_multiple=min_payout_multiple)
    risk_decision = risk_manager.can_trade(stake)
    decision = apply_economics_and_risk(decision, edge_assessment=edge,
                                        risk_decision=risk_decision,
                                        research_mode=research_mode)
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
    try:
        buy_resp = await client.buy(proposal.proposal_id, proposal.ask_price,
                                    idempotency_key=idempotency_key)
    except BuyAmbiguousError as exc:
        db.log_event("ERROR", "execution",
                     f"buy outcome unknown, needs reconciliation: {exc}",
                     {"idempotency_key": idempotency_key, "symbol": symbol})
        decision.decision = "NO_TRADE"
        decision.reason_code = "NO_TRADE_BUY_AMBIGUOUS"
        decision.explanation = str(exc)
        return decision, edge, None
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

    return decision, edge, buy_resp


async def monitor_settlement(client: DerivClient, db, risk_manager, staking,
                             contract_id: int) -> None:
    poc = await client.wait_for_settlement(contract_id)
    if not poc:
        risk_manager.register_result(0.0)
        staking.register_result(False)
        db.record_trade_result(contract_id, won=False, pnl=0.0,
                               error="settlement timeout")
        return
    profit = float(poc.get("profit", 0.0))
    won = profit > 0
    risk_manager.register_result(profit)
    staking.register_result(won)
    db.record_trade_result(contract_id, won=won, pnl=profit,
                           exit_spot=float(poc.get("exit_tick", 0.0) or 0.0))
