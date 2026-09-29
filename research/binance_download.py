"""Download Binance USD(S)-M perpetual futures history for research.

What it fetches, from Binance's public archive (data.binance.vision, no
account or API key needed):
  * hourly candles (open, high, low, close, quote volume) for EVERY USDT
    perpetual, including delisted ones -- leaving dead coins out makes any
    strategy look better than it really was
  * funding-rate history for the same symbols

Standard library only (no pip installs). Resumable: rerun the same command
after a dropped connection and it continues where it stopped.

Usage (from the folder this file is in):
    python binance_download.py                 # everything, 1h candles
    python binance_download.py --interval 4h   # ~4x smaller if space/data is tight
    python binance_download.py --symbols BTCUSDT ETHUSDT   # quick trial

Output: data/binance/<interval>/  and a single upload file
        data/binance_<interval>.zip  -> send that file to Claude.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import lzma
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

ARCHIVE = os.environ.get("BINANCE_ARCHIVE", "https://data.binance.vision")
LISTING = os.environ.get("BINANCE_LISTING", "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision")
FIRST_MONTH = (2019, 9)          # USD-M futures launched September 2019
UA = {"User-Agent": "research-downloader/1.0"}


# ---------------------------------------------------------------- http
def fetch(url: str, tries: int = 5) -> bytes | None:
    """Bytes, or None on 404. Retries network errors with backoff."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == tries - 1:
                raise
        except Exception:
            if k == tries - 1:
                raise
        time.sleep(2 ** k)
    return None


def s3_list(prefix: str) -> tuple[list[str], list[str]]:
    """(sub-folders, file keys) under prefix, following pagination."""
    dirs, keys, marker = [], [], ""
    while True:
        url = f"{LISTING}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
        body = fetch(url)
        if body is None:
            break
        root = ET.fromstring(body)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        dirs += [e.text for e in root.iter(f"{ns}Prefix") if e.text and e.text != prefix]
        keys += [e.text for e in root.iter(f"{ns}Key")]
        if (root.findtext(f"{ns}IsTruncated") or "false").lower() != "true":
            break
        marker = root.findtext(f"{ns}NextMarker") or (keys[-1] if keys else (dirs[-1] if dirs else ""))
        if not marker:
            break
    return dirs, keys


# ------------------------------------------------------------- symbols
def all_symbols() -> list[str]:
    dirs, _ = s3_list("data/futures/um/monthly/klines/")
    syms = sorted({d.rstrip("/").split("/")[-1] for d in dirs})
    # USDT-margined perpetuals only: skip USDC/BUSD pairs and dated quarterlies (BTCUSDT_240628)
    syms = [s for s in syms if s.endswith("USDT") and "_" not in s]
    if syms:
        return syms
    # archive listing unreachable -> current symbols from the live API (misses delisted coins)
    body = fetch("https://fapi.binance.com/fapi/v1/exchangeInfo")
    if not body:
        return []
    print("WARNING: archive listing failed; using live symbols only (delisted coins missing).", flush=True)
    info = json.loads(body)
    return sorted(x["symbol"] for x in info.get("symbols", [])
                  if x.get("contractType") == "PERPETUAL" and x.get("quoteAsset") == "USDT")


def months_available(kind: str, sym: str, interval: str | None) -> list[str]:
    """YYYY-MM strings that exist in the archive for this symbol."""
    sub = f"{sym}/{interval}/" if interval else f"{sym}/"
    _, keys = s3_list(f"data/futures/um/monthly/{kind}/{sub}")
    months = sorted({m.group(1) for k in keys if (m := re.search(r"-(\d{4}-\d{2})\.zip$", k))})
    if months:
        return months
    # listing unavailable -> try every month since launch (404s are skipped)
    now = datetime.now(timezone.utc)
    y, m, out = *FIRST_MONTH, []
    while (y, m) < (now.year, now.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ------------------------------------------------------------- parsing
def unzip_csv(blob: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        text = z.read(name).decode()
    return [r for r in csv.reader(io.StringIO(text)) if r]


def kline_rows(blob: bytes):
    """-> (open_time_seconds, open, high, low, close, quote_volume). Newer files
    have a header row, older ones do not; open_time may be ms or us."""
    for r in unzip_csv(blob):
        if not r[0].strip().isdigit():
            continue                                   # header
        t = int(r[0])
        t = t // 1_000_000 if t > 10 ** 14 else t // 1000
        yield t, r[1], r[2], r[3], r[4], r[7]


def funding_rows(blob: bytes):
    """-> (time_seconds, funding_rate). Header: calc_time,funding_interval_hours,last_funding_rate."""
    rows = unzip_csv(blob)
    col_t, col_r = 0, -1
    if rows and not rows[0][0].strip().isdigit():
        h = [c.strip().lower() for c in rows[0]]
        col_t = next((i for i, c in enumerate(h) if "time" in c), 0)
        col_r = next((i for i, c in enumerate(h) if "rate" in c), len(h) - 1)
        rows = rows[1:]
    for r in rows:
        if r[col_t].strip().isdigit():
            t = int(r[col_t])
            yield (t // 1000 if t > 10 ** 11 else t), r[col_r]


# ------------------------------------------------------------ download
def download_symbol(sym: str, interval: str, out_dir: str) -> dict:
    kpath = os.path.join(out_dir, "klines", f"{sym}.csv.xz")
    fpath = os.path.join(out_dir, "funding", f"{sym}.csv.xz")
    if os.path.exists(kpath) and os.path.exists(fpath):
        return {"sym": sym, "skipped": True}

    rows, missing = {}, 0
    for mo in months_available("klines", sym, interval):
        blob = fetch(f"{ARCHIVE}/data/futures/um/monthly/klines/{sym}/{interval}/{sym}-{interval}-{mo}.zip")
        if blob is None:
            missing += 1
            continue
        for row in kline_rows(blob):
            rows[row[0]] = row
    fund = {}
    for mo in months_available("fundingRate", sym, None):
        blob = fetch(f"{ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{mo}.zip")
        if blob is None:
            continue
        for t, rate in funding_rows(blob):
            fund[t] = rate

    # write atomically so an interrupted run never leaves a half file that looks complete
    for path, header, data in ((kpath, "t,open,high,low,close,quote_volume", [rows[k] for k in sorted(rows)]),
                               (fpath, "t,funding_rate", [(k, fund[k]) for k in sorted(fund)])):
        tmp = path + ".part"
        with lzma.open(tmp, "wt") as f:
            f.write(header + "\n")
            for r in data:
                f.write(",".join(map(str, r)) + "\n")
        os.replace(tmp, path)
    first = datetime.fromtimestamp(min(rows), timezone.utc).date().isoformat() if rows else None
    last = datetime.fromtimestamp(max(rows), timezone.utc).date().isoformat() if rows else None
    return {"sym": sym, "bars": len(rows), "funding": len(fund), "first": first, "last": last}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", default="1h", choices=["1h", "4h"])
    ap.add_argument("--symbols", nargs="*", help="only these symbols (default: all USDT perpetuals incl. delisted)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join("data", "binance"))
    a = ap.parse_args()

    out_dir = os.path.join(a.out, a.interval)
    for sub in ("klines", "funding"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    print("Listing symbols from the Binance archive ...", flush=True)
    syms = a.symbols or all_symbols()
    if not syms:
        sys.exit("Could not list symbols -- check your internet connection and try again.")
    print(f"{len(syms)} symbols, {a.interval} candles. Safe to stop and rerun at any time.\n", flush=True)

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    failed, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(download_symbol, s, a.interval, out_dir): s for s in syms}
        for i, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                info = fut.result()
                if not info.get("skipped"):
                    manifest[s] = info
                    json.dump(manifest, open(manifest_path, "w"), indent=1)
                msg = "already done" if info.get("skipped") else f"{info['bars']} bars {info['first']} -> {info['last']}"
            except Exception as e:  # noqa: BLE001
                failed.append(s)
                msg = f"FAILED ({type(e).__name__}: {e}) -- rerun to retry"
            eta = (time.time() - t0) / i * (len(syms) - i) / 60
            print(f"[{i:>4}/{len(syms)}] {s:<18} {msg}   (~{eta:.0f} min left)", flush=True)

    if failed:
        print(f"\n{len(failed)} symbols failed: {' '.join(failed)}\nRun the same command again to retry them.")
        sys.exit(1)

    zpath = os.path.normpath(os.path.join(a.out, "..", f"binance_{a.interval}.zip"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:   # files are already xz-compressed
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn.endswith(".part"):
                    continue
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, a.out))
    mb = os.path.getsize(zpath) / 1e6
    print(f"\nDone in {(time.time() - t0) / 60:.0f} min. Upload this file to Claude:\n  {os.path.abspath(zpath)}  ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
