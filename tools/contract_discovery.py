"""
What can actually be traded, on which symbol, for how long?  Asks Deriv.

    python tools/contract_discovery.py                         # forex + stock indices
    python tools/contract_discovery.py --markets forex
    python tools/contract_discovery.py --symbols frxEURUSD,OTC_SPC

For every symbol in the chosen markets it calls `contracts_for` and prints,
per contract type (CALL/PUT = Rise/Fall, EXPIRYRANGE = Ends Between,
RANGE = Stays Between, ...), the shortest and longest duration Deriv
offers. Settles the design question "is the minimum 15 minutes or 1 day?"
per symbol and per contract type, instead of guessing. Also writes
data/contract_discovery.csv. Read-only: nothing is bought.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TYPE_NAMES = {
    "CALL": "Rise/Fall", "PUT": "Rise/Fall", "CALLE": "Rise/Fall (equal)", "PUTE": "Rise/Fall (equal)",
    "EXPIRYRANGE": "Ends Between", "EXPIRYMISS": "Ends Outside",
    "RANGE": "Stays Between", "UPORDOWN": "Goes Outside",
    "ONETOUCH": "Touch", "NOTOUCH": "No Touch",
}
_DUR = re.compile(r"^(\d+)([tsmhd])$")
_SECONDS = {"t": 0, "s": 1, "m": 60, "h": 3600, "d": 86400}


def dur_seconds(d: str) -> float:
    m = _DUR.match(str(d))
    if not m:
        return float("inf")
    n, u = int(m.group(1)), m.group(2)
    return n * 1e-6 if u == "t" else n * _SECONDS[u]      # ticks sort before any time


async def run(markets: list[str], symbols: list[str] | None) -> list[dict]:
    from config.loader import Settings
    from main import _make_client

    client = _make_client(Settings())
    await client.connect()
    rows: list[dict] = []
    try:
        active = await client.active_symbols()
        info = {(s.get("underlying_symbol") or s.get("symbol")): s for s in active}
        if symbols:
            chosen = symbols
        else:
            chosen = sorted(code for code, s in info.items()
                            if str(s.get("market", "")).lower() in markets)
        print(f"checking {len(chosen)} symbols...\n")
        for sym in chosen:
            meta = info.get(sym, {})
            try:
                resp = await client.contracts_for(sym)
            except Exception as exc:  # noqa: BLE001
                print(f"{sym}: contracts_for failed ({exc})")
                continue
            by_type: dict = defaultdict(lambda: {"min": None, "max": None, "expiry": set()})
            for c in resp.get("contracts_for", {}).get("available", []):
                ct = c.get("contract_type")
                if ct not in TYPE_NAMES:
                    continue
                name = TYPE_NAMES[ct]
                lo, hi = c.get("min_contract_duration"), c.get("max_contract_duration")
                e = by_type[name]
                if lo and (e["min"] is None or dur_seconds(lo) < dur_seconds(e["min"])):
                    e["min"] = lo
                if hi and (e["max"] is None or dur_seconds(hi) > dur_seconds(e["max"])):
                    e["max"] = hi
                if c.get("expiry_type"):
                    e["expiry"].add(c["expiry_type"])
            for name, e in sorted(by_type.items()):
                rows.append({"symbol": sym, "name": meta.get("display_name", ""),
                             "market": meta.get("market", ""), "submarket": meta.get("submarket", ""),
                             "open": bool(meta.get("exchange_is_open", True)), "contract": name,
                             "min": e["min"], "max": e["max"], "expiry_types": ",".join(sorted(e["expiry"]))})
    finally:
        await client.close()
    return rows


def print_table(rows: list[dict]) -> None:
    contracts = sorted({r["contract"] for r in rows})
    by_sym: dict = defaultdict(dict)
    meta = {}
    for r in rows:
        by_sym[r["symbol"]][r["contract"]] = f"{r['min']}-{r['max']}"
        meta[r["symbol"]] = r
    print(f"{'symbol':<14}{'name':<26}{'open':<6}" + "".join(f"{c:<18}" for c in contracts))
    for sym in sorted(by_sym, key=lambda s: (meta[s]["market"], s)):
        m = meta[sym]
        print(f"{sym:<14}{str(m['name'])[:25]:<26}{'yes' if m['open'] else 'no':<6}"
              + "".join(f"{by_sym[sym].get(c, '-'):<18}" for c in contracts))
    print("\n(min-max duration; t = ticks, s/m/h/d = seconds/minutes/hours/days; '-' = not offered)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="forex,indices",
                    help="comma list of Deriv market names (forex, indices, commodities, cryptocurrency...)")
    ap.add_argument("--symbols", default=None)
    a = ap.parse_args()
    markets = [m.strip().lower() for m in a.markets.split(",") if m.strip()]
    symbols = [s.strip() for s in a.symbols.split(",") if s.strip()] if a.symbols else None
    rows = asyncio.run(run(markets, symbols))
    if not rows:
        print("nothing found -- try --markets with another name, or --symbols")
        return
    print_table(rows)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "contract_discovery.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved data/contract_discovery.csv")


if __name__ == "__main__":
    main()
