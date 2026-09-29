"""
Rise/Fall research for the forex + stock-index bot. One run collects every
number the bot's design depends on.

    python tools/rise_fall_research.py --selftest          # offline check (1 min)
    python tools/rise_fall_research.py                     # full run, 2 years
    python tools/rise_fall_research.py --days 180 --no-payouts

WHAT IT DOES
1. Downloads 5-minute candles for the 37 Rise/Fall symbols (25 forex pairs,
   12 stock indices) plus cryBTCUSD as a risk gauge. Cached in
   data/research/ -- a re-run only fetches what's missing. Read-only: no
   database, nothing is bought.
2. Generates events for eight strategy families (design doc "Symbol
   playbooks"), using only data available at decision time:
     F1 session momentum     F2 overshoot reversion   F3 trend persistence
     F4 currency strength    F5 risk tone (BTC)       F6 opening drive
     F7 gap fade             F8 lead-lag across regions
3. Scores each event at expiries 15m, 30m, 1h, 2h, 4h exactly as Rise/Fall
   settles: exit strictly beyond entry in the called direction.
4. Two-stage, so the verdict is honest:
     DISCOVERY  = first 60% of history. Per symbol and family, pick the
                  direction (as designed, or reversed) and the best expiry.
     VALIDATION = last 40%, untouched. Test ONLY those picks, against the
                  break-even implied by the live payout, with a Bonferroni
                  correction for the number of picks.
5. Samples live payouts for every symbol and expiry (proposals only).

OUTPUT (data/research/): summary.txt, results.csv, family_pooled.csv,
payouts.csv, events.csv, data_coverage.csv. Send summary.txt plus the CSVs.

Approximations, stated: entry/exit are 5-minute bar closes (Deriv settles on
ticks); payouts are one snapshot. Both are refined by the bot's own shadow
mode later.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join("data", "research")
BAR = 300                                  # 5-minute bars
EXPIRIES = [15, 30, 60, 120, 240]          # minutes
DISCOVERY_FRACTION = 0.6
MIN_DISC_EVENTS = 40
MIN_DISC_TILT = 0.03                       # |win rate - 0.5| needed to become a pick
DISC_P = 0.05                              # ...and significant in discovery (one-sided vs 50%)
DEFAULT_PAYOUT = 1.85                      # used only when a live payout is missing (flagged)
ALPHA = 0.05

FOREX = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD", "frxUSDCHF",
         "frxEURGBP", "frxEURJPY", "frxGBPJPY", "frxAUDJPY", "frxEURAUD", "frxEURCAD",
         "frxEURCHF", "frxGBPAUD", "frxAUDCAD", "frxAUDCHF", "frxAUDNZD", "frxEURNZD",
         "frxGBPCAD", "frxGBPCHF", "frxGBPNZD", "frxNZDJPY", "frxNZDUSD", "frxUSDMXN",
         "frxUSDPLN"]
# symbol -> (timezone, cash open, cash close), local exchange times
INDICES = {
    "OTC_SPC": ("America/New_York", (9, 30), (16, 0)),
    "OTC_NDX": ("America/New_York", (9, 30), (16, 0)),
    "OTC_DJI": ("America/New_York", (9, 30), (16, 0)),
    "OTC_GDAXI": ("Europe/Berlin", (9, 0), (17, 30)),
    "OTC_SX5E": ("Europe/Berlin", (9, 0), (17, 30)),
    "OTC_FCHI": ("Europe/Paris", (9, 0), (17, 30)),
    "OTC_AEX": ("Europe/Amsterdam", (9, 0), (17, 30)),
    "OTC_SSMI": ("Europe/Zurich", (9, 0), (17, 30)),
    "OTC_FTSE": ("Europe/London", (8, 0), (16, 30)),
    "OTC_N225": ("Asia/Tokyo", (9, 0), (15, 30)),
    "OTC_HSI": ("Asia/Hong_Kong", (9, 30), (16, 0)),
    "OTC_AS51": ("Australia/Sydney", (10, 0), (16, 0)),
}
RISK = "cryBTCUSD"
FX_SESSIONS = {"london": ("Europe/London", (8, 0)), "newyork": ("America/New_York", (8, 0))}
LEADERS = {  # follower -> (leader, leader window description)
    "OTC_N225": "us_close", "OTC_HSI": "us_close", "OTC_AS51": "us_close",
    "OTC_SPC": "eu_morning", "OTC_NDX": "eu_morning", "OTC_DJI": "eu_morning",
}
FAMILIES = {
    "F1L": "London momentum", "F1N": "New York momentum", "F2": "overshoot reversion", "F3": "trend persistence",
    "F4": "currency strength", "F5": "risk tone (BTC)", "F6": "opening drive",
    "F7": "gap fade", "F8": "lead-lag",
}


# =================================================================== data

@dataclass
class Series:
    sym: str
    t: np.ndarray          # bar open epochs
    c: np.ndarray          # closes
    idx: dict              # epoch -> position
    r: np.ndarray          # 5m log return ending at bar i (nan across gaps)
    rv12: np.ndarray       # rms 5m return over last 12 bars (1h)
    rv288: np.ndarray      # rms 5m return over last 288 bars (1 day)

    def at(self, epoch: int):
        return self.idx.get(int(epoch))


def _rolling_rms(r: np.ndarray, n: int) -> np.ndarray:
    ok = ~np.isnan(r)
    r2 = np.where(ok, r * r, 0.0)
    cs, cn = np.concatenate([[0], np.cumsum(r2)]), np.concatenate([[0], np.cumsum(ok)])
    s = cs[n:] - cs[:-n]
    k = cn[n:] - cn[:-n]
    out = np.full(len(r), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1:] = np.sqrt(np.where(k > n // 2, s / np.maximum(k, 1), np.nan))
    return out


def make_series(sym: str, t: np.ndarray, c: np.ndarray) -> Series:
    order = np.argsort(t)
    t, c = t[order].astype(np.int64), c[order].astype(float)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, c = t[keep], c[keep]
    r = np.full(len(c), np.nan)
    good = np.diff(t) == BAR
    r[1:][good] = np.diff(np.log(c))[good]
    return Series(sym, t, c, {int(e): i for i, e in enumerate(t)}, r,
                  _rolling_rms(r, 12), _rolling_rms(r, 288))


def cache_file(sym: str) -> str:
    return os.path.join(OUT, "candles", f"{sym}_5m.csv")


def load_cache(sym: str):
    p = cache_file(sym)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.reader(f))[1:]
    if not rows:
        return None
    return np.array([int(r[0]) for r in rows]), np.array([float(r[1]) for r in rows])


def save_cache(sym: str, t, c) -> None:
    os.makedirs(os.path.dirname(cache_file(sym)), exist_ok=True)
    with open(cache_file(sym), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "close"])
        w.writerows(zip(t.tolist(), c.tolist()))


async def fetch_all(symbols: list[str], days: float) -> dict:
    from config.loader import Settings
    from main import _make_client

    want = int(days * 288)
    out, need = {}, []
    for s in symbols:
        cached = load_cache(s)
        if cached is not None and (time.time() - cached[0][-1]) < 3 * 86400 \
                and (cached[0][-1] - cached[0][0]) >= (days - 3) * 86400:
            out[s] = cached
        else:
            need.append(s)
    if need:
        client = _make_client(Settings())
        await client.connect()
        try:
            for i, s in enumerate(need, 1):
                try:
                    raw = await client.candle_history_paged(s, want, granularity=BAR)
                except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                    print(f"  [{i}/{len(need)}] {s}: FAILED ({exc}) -- re-run to retry")
                    continue
                if not raw:
                    print(f"  [{i}/{len(need)}] {s}: no data")
                    continue
                t = np.array([x.epoch for x in raw]); c = np.array([x.close for x in raw])
                save_cache(s, t, c)
                out[s] = (t, c)
                print(f"  [{i}/{len(need)}] {s}: {len(t):,} bars, "
                      f"{(t[-1] - t[0]) / 86400:.0f} days")
        finally:
            await client.close()
    print(f"data ready for {len(out)}/{len(symbols)} symbols")
    return out


async def sample_payouts(symbols: list[str]) -> dict:
    """{(sym, minutes): median payout multiple} from live proposals."""
    from config.loader import Settings
    from main import _make_client

    settings = Settings()
    client = _make_client(settings)
    await client.connect()
    res, rows = {}, []
    try:
        for s in symbols:
            for m in EXPIRIES:
                mults = []
                for ct in ("CALL", "PUT"):
                    try:
                        p = await client.proposal(symbol=s, contract_type=ct, amount=10,
                                                  currency=settings.currency, duration=m,
                                                  duration_unit="m")
                        if p.get("payout"):
                            mults.append(float(p["payout"]) / 10.0)
                    except Exception:  # noqa: BLE001 - closed market etc.
                        pass
                if mults:
                    res[(s, m)] = float(np.median(mults))
                rows.append({"symbol": s, "minutes": m,
                             "payout_multiple": round(res[(s, m)], 4) if (s, m) in res else "",
                             "break_even": round(1 / res[(s, m)], 4) if (s, m) in res else "",
                             "sampled_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M")})
            got = sum(1 for m in EXPIRIES if (s, m) in res)
            print(f"  payouts {s}: {got}/{len(EXPIRIES)} expiries quoted")
    finally:
        await client.close()
    _write_csv("payouts.csv", rows)
    return res


# ============================================================ helpers

def pair_ccys(sym: str):
    return (sym[3:6], sym[6:9]) if sym.startswith("frx") and len(sym) == 9 else (None, None)


def local_epoch(tz: str, d: date, hm) -> int:
    return int(datetime(d.year, d.month, d.day, hm[0], hm[1], tzinfo=ZoneInfo(tz)).timestamp())


def days_between(t0: int, t1: int):
    d = datetime.utcfromtimestamp(t0).date() - timedelta(days=1)
    end = datetime.utcfromtimestamp(t1).date() + timedelta(days=1)
    while d <= end:
        yield d
        d += timedelta(days=1)


def z_move(s: Series, i: int, j: int, nbars: int):
    """Move from bar j close to bar i close in units of the typical move over nbars."""
    sig = s.rv288[i]
    if j is None or i is None or not np.isfinite(sig) or sig <= 0:
        return None
    return math.log(s.c[i] / s.c[j]) / (sig * math.sqrt(nbars))


def last_bar_at_or_before(s: Series, epoch: int, max_back: int = 12):
    for k in range(max_back + 1):
        i = s.at(epoch - k * BAR)
        if i is not None:
            return i
    return None


def cooldown_filter(events, bars: int):
    out, last = [], {}
    for e in sorted(events, key=lambda e: e[1]):
        key = (e[0], e[2])
        if key in last and e[1] - last[key] < bars * BAR:
            continue
        last[key] = e[1]
        out.append(e)
    return out


# ============================================================ families
# An event = (symbol, decision_epoch, family, direction +1/-1).
# decision_epoch = the bar-open epoch whose CLOSE is the entry price.

def f1_session_momentum(S: dict):
    ev = []
    for sym in FOREX:
        s = S.get(sym)
        if s is None:
            continue
        for d in days_between(int(s.t[0]), int(s.t[-1])):
            for sess, (tz, hm) in FX_SESSIONS.items():
                fam = "F1L" if sess == "london" else "F1N"
                o = local_epoch(tz, d, hm)
                if datetime.fromtimestamp(o, ZoneInfo(tz)).weekday() >= 5:
                    continue
                i, j = s.at(o + 55 * 60), s.at(o - BAR)       # first hour of the session
                z = z_move(s, i, j, 12)
                if z is not None and abs(z) > 1.0:
                    ev.append((sym, int(s.t[i]), fam, 1 if z > 0 else -1))
    return ev


def f2_overshoot(S: dict):
    ev = []
    for sym in FOREX + list(INDICES):
        s = S.get(sym)
        if s is None or len(s.c) < 400:
            continue
        cs = np.concatenate([[0], np.cumsum(s.c)])
        for i in range(300, len(s.c)):
            if s.t[i] - s.t[i - 48] != 48 * BAR:
                continue
            twap = (cs[i + 1] - cs[i - 47]) / 48
            sig = s.rv288[i]
            if not np.isfinite(sig) or sig <= 0 or not (s.rv12[i] < s.rv288[i]):
                continue
            dev = math.log(s.c[i] / twap) / (sig * math.sqrt(48))
            if abs(dev) > 2.0:
                ev.append((sym, int(s.t[i]), "F2", -1 if dev > 0 else 1))
    return cooldown_filter(ev, 12)


def f3_trend(S: dict):
    ev = []
    for sym in FOREX + list(INDICES):
        s = S.get(sym)
        if s is None:
            continue
        for i in range(300, len(s.c)):
            if (s.t[i] // BAR) % 12 != 11:                   # decide on the hour
                continue
            j = s.at(int(s.t[i]) - 48 * BAR)
            z = z_move(s, i, j, 48)
            if z is None or not (s.rv12[i] < s.rv288[i]):
                continue
            if abs(z) > 1.5:
                ev.append((sym, int(s.t[i]), "F3", 1 if z > 0 else -1))
    return cooldown_filter(ev, 12)


def f4_strength(S: dict):
    """Pair should catch up with the strength gap between its two currencies,
    measured on the OTHER pairs only (hourly, 1h returns)."""
    hourly = defaultdict(dict)                 # epoch -> {sym: 1h log return}
    for sym in FOREX:
        s = S.get(sym)
        if s is None:
            continue
        for i in range(12, len(s.c)):
            if (s.t[i] // BAR) % 12 != 11:
                continue
            j = s.at(int(s.t[i]) - 12 * BAR)
            if j is not None:
                hourly[int(s.t[i])][sym] = math.log(s.c[i] / s.c[j])
    ev = []
    for e, rets in hourly.items():
        if len(rets) < 8:
            continue
        for sym, _ in rets.items():
            b, q = pair_ccys(sym)
            def strength(ccy):
                vals = []
                for other, r in rets.items():
                    if other == sym:
                        continue
                    ob, oq = pair_ccys(other)
                    if ob == ccy:
                        vals.append(r)
                    elif oq == ccy:
                        vals.append(-r)
                return np.mean(vals) if len(vals) >= 2 else None
            sb, sq = strength(b), strength(q)
            s = S[sym]
            i = s.at(e)
            if sb is None or sq is None or i is None or not np.isfinite(s.rv288[i]):
                continue
            z = (sb - sq) / (s.rv288[i] * math.sqrt(12))
            if abs(z) > 1.0:
                ev.append((sym, e, "F4", 1 if z > 0 else -1))
    return cooldown_filter(ev, 12)


def _risk_sign(sym: str) -> int:
    if sym in INDICES:
        return 1
    b, q = pair_ccys(sym)
    sgn = 0
    sgn += {"AUD": 1, "NZD": 1, "JPY": -1, "CHF": -1}.get(b, 0)
    sgn -= {"AUD": 1, "NZD": 1, "JPY": -1, "CHF": -1}.get(q, 0)
    return int(np.sign(sgn))


def f5_risk_tone(S: dict):
    btc = S.get(RISK)
    if btc is None:
        return []
    ev = []
    for i in range(300, len(btc.c)):
        if (btc.t[i] // BAR) % 12 != 11:
            continue
        j = btc.at(int(btc.t[i]) - 48 * BAR)
        z = z_move(btc, i, j, 48)
        if z is None or abs(z) <= 1.5:
            continue
        for sym in FOREX + list(INDICES):
            sg = _risk_sign(sym)
            s = S.get(sym)
            if sg == 0 or s is None or s.at(int(btc.t[i])) is None:
                continue
            ev.append((sym, int(btc.t[i]), "F5", sg * (1 if z > 0 else -1)))
    return cooldown_filter(ev, 48)


def f6_opening_drive(S: dict):
    ev = []
    for sym, (tz, op, _) in INDICES.items():
        s = S.get(sym)
        if s is None:
            continue
        for d in days_between(int(s.t[0]), int(s.t[-1])):
            o = local_epoch(tz, d, op)
            if datetime.fromtimestamp(o, ZoneInfo(tz)).weekday() >= 5:
                continue
            i, j = s.at(o + 25 * 60), s.at(o - BAR)          # first 30 minutes
            z = z_move(s, i, j, 6)
            if z is not None and abs(z) > 1.0:
                ev.append((sym, int(s.t[i]), "F6", 1 if z > 0 else -1))
    return ev


def f7_gap_fade(S: dict):
    ev = []
    for sym, (tz, op, cl) in INDICES.items():
        s = S.get(sym)
        if s is None:
            continue
        prev_close = None
        for d in days_between(int(s.t[0]), int(s.t[-1])):
            if datetime(d.year, d.month, d.day).weekday() >= 5:
                continue
            o, c_ep = local_epoch(tz, d, op), local_epoch(tz, d, cl)
            i = s.at(o)                                      # decide at the first bar's close
            if prev_close is not None and i is not None:
                z = z_move(s, i, prev_close, 78)             # vs a typical full-session move
                if z is not None and abs(z) > 1.0:
                    ev.append((sym, int(s.t[i]), "F7", -1 if z > 0 else 1))
            prev_close = last_bar_at_or_before(s, c_ep - BAR)
    return ev


def f8_lead_lag(S: dict):
    ev = []
    spc, sx5e = S.get("OTC_SPC"), S.get("OTC_SX5E")
    for sym, kind in LEADERS.items():
        s = S.get(sym)
        leader = spc if kind == "us_close" else sx5e
        if s is None or leader is None:
            continue
        tz, op, _ = INDICES[sym]
        for d in days_between(int(s.t[0]), int(s.t[-1])):
            o = local_epoch(tz, d, op)
            if datetime.fromtimestamp(o, ZoneInfo(tz)).weekday() >= 5:
                continue
            if kind == "us_close":                           # last 2h of the previous US session
                ud = datetime.fromtimestamp(o, ZoneInfo("America/New_York")).date() - timedelta(days=1)
                while ud.weekday() >= 5:
                    ud -= timedelta(days=1)
                end = last_bar_at_or_before(leader, local_epoch("America/New_York", ud, (16, 0)) - BAR)
                start = leader.at(local_epoch("America/New_York", ud, (14, 0)) - BAR)
                nb = 24
            else:                                             # Europe from its open to the US open
                end = last_bar_at_or_before(leader, o - BAR)
                start = leader.at(local_epoch("Europe/Berlin", d, (9, 0)) - BAR)
                nb = 54
            z = z_move(leader, end, start, nb) if end is not None else None
            i = s.at(o)
            if z is not None and i is not None and abs(z) > 1.0:
                ev.append((sym, int(s.t[i]), "F8", 1 if z > 0 else -1))
    return ev


FAMILY_FUNCS = [f1_session_momentum, f2_overshoot, f3_trend, f4_strength,
                f5_risk_tone, f6_opening_drive, f7_gap_fade, f8_lead_lag]


# ============================================================ evaluation

def outcomes(S: dict, events):
    """Adds win/loss at each expiry (None when the exit bar is missing)."""
    rows = []
    for sym, e, fam, d in events:
        s = S[sym]
        i = s.at(e)
        rec = {"symbol": sym, "epoch": e, "family": fam, "direction": d}
        for m in EXPIRIES:
            j = s.at(e + m * 60)
            if i is None or j is None:
                rec[m] = None
            else:
                move = s.c[j] - s.c[i]
                rec[m] = (move > 0) if d > 0 else (move < 0)
        rows.append(rec)
    return rows


def binom_sf(w: int, n: int, p: float) -> float:
    """P(X >= w) for X ~ Binomial(n, p)."""
    from scipy.stats import binom
    return float(binom.sf(w - 1, n, p))


def beta_lower(w: int, n: int, q: float) -> float:
    from scipy.stats import beta
    return float(beta.ppf(q, w + 1, n - w + 1))


def evaluate(rows, cut_epoch: int, payouts: dict):
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["symbol"], r["family"])].append(r)
    picks = []
    for (sym, fam), rs in by_key.items():
        disc = [r for r in rs if r["epoch"] < cut_epoch]
        best = None
        for m in EXPIRIES:
            res = [r[m] for r in disc if r[m] is not None]
            if len(res) < MIN_DISC_EVENTS:
                continue
            wr = sum(res) / len(res)
            tilt = abs(wr - 0.5)
            w_dir = sum(res) if wr > 0.5 else len(res) - sum(res)
            if binom_sf(w_dir, len(res), 0.5) > DISC_P:
                continue
            if tilt >= MIN_DISC_TILT and (best is None or tilt > best[2]):
                best = (m, 1 if wr > 0.5 else -1, tilt, wr, len(res))
        if best:
            picks.append((sym, fam) + best)
    K = max(1, len(picks))
    results = []
    for sym, fam, m, flip, tilt, dwr, dn in picks:
        val = [r[m] for r in by_key[(sym, fam)] if r["epoch"] >= cut_epoch and r[m] is not None]
        if flip < 0:
            val = [not x for x in val]
        n, w = len(val), sum(val)
        mult = payouts.get((sym, m))
        be = 1 / (mult if mult else DEFAULT_PAYOUT)
        p = binom_sf(w, n, be) if n else 1.0
        wr = w / n if n else float("nan")
        lb = beta_lower(w, n, ALPHA / K) if n else 0.0
        results.append({
            "symbol": sym, "family": fam, "family_name": FAMILIES[fam],
            "direction": "as designed" if flip > 0 else "REVERSED", "expiry_min": m,
            "disc_n": dn, "disc_win": round(dwr if flip > 0 else 1 - dwr, 4),
            "val_n": n, "val_win": round(wr, 4) if n else "",
            "payout": round(mult, 3) if mult else f"{DEFAULT_PAYOUT} (assumed)",
            "break_even": round(be, 4), "edge_pts": round((wr - be) * 100, 2) if n else "",
            "p_value": f"{p:.2e}", "lower_bound": round(lb, 4),
            "ev_worst_case": round(lb / be - 1, 4) if n else "",
            "PASS": "YES" if (n >= 30 and p < ALPHA / K) else ("promising" if (n >= 30 and p < ALPHA) else ""),
        })
    results.sort(key=lambda r: (r["PASS"] != "YES", r["PASS"] != "promising",
                                -(r["edge_pts"] if r["edge_pts"] != "" else -99)))
    return results, K


def pooled_by_family(rows, cut_epoch, results):
    """Validation win rate per family, pooled over its picks (same direction/expiry as
    picked), tested against the average break-even of those picks, Bonferroni over families."""
    pick = {(r["symbol"], r["family"]): (r["expiry_min"], r["direction"]) for r in results}
    be = defaultdict(list)
    for r in results:
        be[r["family"]].append(r["break_even"])
    agg = defaultdict(lambda: [0, 0, set()])
    for r in rows:
        k = (r["symbol"], r["family"])
        if k not in pick or r["epoch"] < cut_epoch:
            continue
        m, dirn = pick[k]
        v = r[m]
        if v is None:
            continue
        if dirn == "REVERSED":
            v = not v
        a = agg[r["family"]]
        a[0] += int(v); a[1] += 1; a[2].add(r["symbol"])
    out, F = [], max(1, len(agg))
    for fam, (w, n, syms) in sorted(agg.items()):
        b = float(np.mean(be[fam])) if be[fam] else 1 / DEFAULT_PAYOUT
        p = binom_sf(w, n, b) if n else 1.0
        out.append({"family": fam, "family_name": FAMILIES[fam], "symbols": len(syms),
                    "val_n": n, "val_win": round(w / n, 4) if n else "",
                    "break_even": round(b, 4), "p_vs_break_even": f"{p:.2e}",
                    "PASS": "YES" if p < ALPHA / F else ""})
    return out


def _write_csv(name, rows):
    if not rows:
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ============================================================ report

def run_analysis(raw: dict, payouts: dict, label: str) -> str:
    S = {s: make_series(s, *tc) for s, tc in raw.items()}
    cov = [{"symbol": s, "bars": len(x.t), "first_utc": datetime.utcfromtimestamp(int(x.t[0])).strftime("%Y-%m-%d"),
            "last_utc": datetime.utcfromtimestamp(int(x.t[-1])).strftime("%Y-%m-%d"),
            "days": round((x.t[-1] - x.t[0]) / 86400, 1)} for s, x in sorted(S.items())]
    _write_csv("data_coverage.csv", cov)

    events = []
    for f in FAMILY_FUNCS:
        t0 = time.time()
        ev = f(S)
        events += ev
        print(f"  {f.__name__:<22} {len(ev):>7,} events  ({time.time() - t0:.0f}s)")
    rows = outcomes(S, events)
    _write_csv("events.csv", [{**{k: v for k, v in r.items() if not isinstance(k, int)},
                               **{f"win_{m}m": ("" if r[m] is None else int(r[m])) for m in EXPIRIES}}
                              for r in rows])
    t_all = np.concatenate([x.t for x in S.values()])
    cut = int(t_all.min() + DISCOVERY_FRACTION * (t_all.max() - t_all.min()))
    results, K = evaluate(rows, cut, payouts)
    _write_csv("results.csv", results)
    pooled = pooled_by_family(rows, cut, results)
    _write_csv("family_pooled.csv", pooled)

    L = [f"RISE/FALL RESEARCH -- {label}",
         f"generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
         f"symbols with data: {len(S)}   events: {len(events):,}",
         f"discovery before {datetime.utcfromtimestamp(cut):%Y-%m-%d}, validation after "
         f"(picks tested: {K}, Bonferroni alpha {ALPHA / K:.1e})",
         f"live payouts sampled: {len(payouts)} symbol-expiry pairs"
         + ("" if payouts else f" -- NONE, break-even assumes {DEFAULT_PAYOUT}x"), ""]
    passes = [r for r in results if r["PASS"] == "YES"]
    prom = [r for r in results if r["PASS"] == "promising"]
    L.append(f"VALIDATED (survive the multiple-testing correction): {len(passes)}")
    L.append(f"PROMISING (beat break-even at 5%, not after correction): {len(prom)}\n")
    hdr = f"{'symbol':<11}{'family':<22}{'dir':<12}{'exp':>5}{'val n':>7}{'win':>7}{'b/e':>7}{'edge':>7}{'lower':>7}  result"
    L.append(hdr)
    for r in (passes + prom)[:40]:
        L.append(f"{r['symbol']:<11}{r['family_name']:<22}{r['direction']:<12}{r['expiry_min']:>4}m"
                 f"{r['val_n']:>7}{r['val_win']:>7}{r['break_even']:>7}{r['edge_pts']:>6}p"
                 f"{r['lower_bound']:>7}  {r['PASS']}")
    if not passes and not prom:
        L.append("  (none)")
    L.append("\nFAMILY TOTALS in validation (picks pooled, vs their break-even; "
             "PASS = beats it after correction over families)")
    for p in pooled:
        L.append(f"  {p['family']:<4}{p['family_name']:<22} {p['symbols']:>3} symbols "
                 f"{p['val_n']:>7} trades  win {p['val_win']}  b/e {p['break_even']}  "
                 f"p {p['p_vs_break_even']}  {p['PASS']}")
    L.append("\nData: 5-minute bar closes approximate Deriv's tick settlement; payouts are "
             "one snapshot.")
    text = "\n".join(L)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ============================================================ self-test

def synthetic(days=730, planted=True, seed=3):
    """Random walks for a few symbols; if `planted`, EURUSD's London first hour
    really does continue for the next hour (58%-ish). A correct pipeline
    validates that and nothing else."""
    rng = np.random.default_rng(seed)
    start = int(datetime(2025, 1, 6, tzinfo=ZoneInfo("UTC")).timestamp())
    t = np.arange(start, start + days * 86400, BAR)
    t = t[[datetime.utcfromtimestamp(int(x)).weekday() < 5 for x in t]]
    syms = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "OTC_SPC", "OTC_GDAXI",
            "OTC_N225", "OTC_SX5E", RISK]
    out = {}
    lon_open = {}
    for d in days_between(int(t[0]), int(t[-1])):
        lon_open[local_epoch("Europe/London", d, (8, 0))] = True
    for s in syms:
        r = rng.normal(0, 4e-4, len(t))
        if planted and s == "frxEURUSD":
            for k, e in enumerate(t):
                if int(e) - 3600 in lon_open and k >= 12:           # an hour after the open
                    drift = np.sign(r[k - 12:k].sum()) * 1.6e-4
                    r[k:k + 12] += drift
        out[s] = (t.copy(), 100 * np.exp(np.cumsum(r)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--days", type=float, default=730)
    ap.add_argument("--no-payouts", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    global OUT
    if a.selftest:
        OUT = os.path.join("data", "research_selftest")
        print("SELF-TEST (offline): planted edge on EURUSD session momentum only.")
        print(run_analysis(synthetic(), {}, "SELF-TEST synthetic"))
        print("\nExpected: London momentum on frxEURUSD validated; nothing else passes.")
        return
    symbols = FOREX + list(INDICES) + [RISK]
    print(f"1/3 downloading {len(symbols)} symbols, {a.days:g} days of 5-minute bars "
          f"(cached in {OUT})")
    raw = asyncio.run(fetch_all(symbols, a.days))
    payouts = {}
    if not a.no_payouts:
        print("2/3 sampling live payouts (proposals only, nothing bought)")
        payouts = asyncio.run(sample_payouts([s for s in symbols if s != RISK]))
    print("3/3 generating and scoring events")
    print("\n" + run_analysis(raw, payouts, f"{a.days:g} days"))
    print(f"\nFiles in {OUT}: summary.txt, results.csv, family_pooled.csv, payouts.csv, "
          f"events.csv, data_coverage.csv")


if __name__ == "__main__":
    main()
