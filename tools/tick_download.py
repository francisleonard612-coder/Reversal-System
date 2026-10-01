"""
Download the most recent raw ticks for Deriv symbols (Deriv keeps ~24h).

    python tools\\tick_download.py --symbols R_100,R_75,RDBULL,RDBEAR,1HZ10V --ticks 90000

Writes data/cache/<SYMBOL>_ticks.csv (epoch,price) -- send those files to
Claude. Resumable per symbol: delete a file to fetch it again.
Read-only: nothing is traded.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def fetch_ticks(client, symbol: str, total: int) -> list[tuple[float, float]]:
    out: dict[float, float] = {}
    end: int | str = "latest"
    while len(out) < total:
        for attempt in range(5):
            try:
                resp = await client._send({"ticks_history": symbol, "count": 5000, "end": end,
                                           "style": "ticks", "adjust_start_time": 1})
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"  {symbol}: page failed ({exc}), retrying ({attempt + 1}/4)")
                await asyncio.sleep(2 + 3 * attempt)
                try:
                    await client.ensure_connected()
                except Exception:  # noqa: BLE001
                    pass
        h = resp.get("history", {})
        page = [(float(t), float(p)) for p, t in zip(h.get("prices", []), h.get("times", []))]
        oldest_have = min(out) if out else float("inf")
        older = [(t, p) for t, p in page if t < oldest_have]
        if not older:            # past Deriv's retention it wraps back to "latest" -- stop
            break
        for t, p in older:
            out[t] = p
        end = int(min(t for t, _ in older)) - 1
        print(f"  {symbol}: {len(out):,} ticks", end="\r", flush=True)
    print()
    return sorted(out.items())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="R_100,R_75,RDBULL,RDBEAR")
    ap.add_argument("--ticks", type=int, default=90000, help="max ticks per symbol")
    a = ap.parse_args()
    from config.loader import Settings
    from main import _make_client

    os.makedirs(os.path.join("data", "cache"), exist_ok=True)
    client = _make_client(Settings())
    await client.connect()
    try:
        for sym in [s.strip() for s in a.symbols.split(",") if s.strip()]:
            path = os.path.join("data", "cache", f"{sym}_ticks.csv")
            if os.path.exists(path):
                print(f"{sym}: already downloaded ({path})")
                continue
            print(f"{sym}: downloading ...")
            rows = await fetch_ticks(client, sym, a.ticks)
            if not rows:
                print(f"{sym}: no data -- check the symbol name")
                continue
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "price"])
                w.writerows(rows)
            hrs = (rows[-1][0] - rows[0][0]) / 3600
            print(f"{sym}: {len(rows):,} ticks over {hrs:.1f} hours -> {path}")
    finally:
        await client.close()
    print("\nDone. Send the *_ticks.csv files from data\\cache to Claude.")


if __name__ == "__main__":
    asyncio.run(main())
