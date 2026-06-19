"""Rebuild a clean, survivorship-safe adjusted NSE close from:
  - RAW bhavcopy prices (ground truth, incl. delisted names), and
  - a corporate-action calendar cross-validated across NSE official, yfinance
    .splits, and the legacy local file, with the actual raw price drop as the
    final arbiter of every factor.

Logic (per symbol):
  PASS A — source-driven. For each corporate action any source lists, look for a
    confirming raw price drop near the ex-date. Apply it only if the price
    actually dropped (spurious source entries — the legacy file is ~54% of them —
    are rejected because the price is flat). The factor is the clean source ratio
    when the price confirms it, else the price drop itself. Cross-source
    agreement (NSE & yfinance) is recorded as confidence.
  PASS B — price-driven catch-all. Any EXTREME clean persistent drop (>= ~55%,
    physically impossible as a real one-day move on a circuit-banded NSE stock)
    not already covered is a big split/bonus the calendars missed — apply it.
  Medium drops (15-55%) that NO source confirms are left UNADJUSTED — they are
    real crashes, not corporate actions (this is the bonus-vs-crash disambiguation
    the single-source heuristic could not do).

Isolated bad-print spikes are repaired first; the series is back-adjusted so the
last bar == last raw close. Writes a separate cleaned dataset + audit with
per-action provenance. Originals are never modified.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

MAJOR = np.array(sorted({0.909, 0.889, 0.875, 0.857, 0.833, 0.8, 0.75, 0.714, 0.667,
                         0.6, 0.571, 0.5, 0.444, 0.4, 0.375, 0.333, 0.286, 0.25, 0.2,
                         0.167, 0.143, 0.125, 0.1, 0.0667, 0.05}))
EXTREME_DROP = 0.45     # >= ~55%: physically a corporate action even if unlisted
DROP_MAX = 0.85         # a price-drop candidate must be <= this
SRC_WIN = 15            # trading days around a NSE/yfinance ex-date to find the drop
                        # (yfinance/NSE ex-dates can sit a couple of weeks off the
                        # actual bhavcopy price-drop day)
REVERT = 0.93


def _snap(r):
    j = int(np.argmin(np.abs(np.log(MAJOR / r))))
    return MAJOR[j], abs(r / MAJOR[j] - 1.0)


def find_systematic_breaks(files, min_names=6):
    """Detect dates where >=min_names unrelated stocks share a PERSISTENT clean
    price-level step (~x0.5, x2, etc.) — a systematic bhavcopy data error (a
    price-basis break), NOT a corporate action. Returns {symbol: [date, ...]} so
    each affected name's step can be back-adjusted away. The proven case:
    2024-11-04, where 22 names' pre-date history is doubled vs the real price."""
    from collections import defaultdict
    date_hits = defaultdict(list)
    for p in files:
        if "INDX" in os.path.basename(p):
            continue
        try:
            d = pd.read_parquet(p, columns=["date", "close"])
        except Exception:
            continue
        d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date").reset_index(drop=True)
        c = d["close"].to_numpy(dtype=float); n = len(c)
        if n < 12:
            continue
        sym = os.path.basename(p)[:-8].replace("_", ".")
        for t in range(5, n - 5):
            if c[t - 1] <= 0:
                continue
            r = c[t] / c[t - 1]
            if 0.3 < r < 0.62 or 1.6 < r < 3.4:
                pre = np.median(c[t - 5:t]); post = np.median(c[t:t + 5])
                if pre > 0 and 0.85 < (post / pre) / r < 1.15:    # the step persists
                    date_hits[d["date"].iloc[t].date()].append(sym)
    out = defaultdict(list)
    for dt, syms in date_hits.items():
        if len(syms) >= min_names:
            for sym in syms:
                out[sym].append(dt)
    return out


def _trim_leading_stubs(df):
    """Drop leading 'stub' bars — a tiny isolated cluster of old bars separated
    from the real series by a multi-year gap (a bhavcopy artifact that otherwise
    shows up as an 80x one-day jump across the gap)."""
    dates = df["date"].to_numpy()
    n = len(df)
    start = 0
    for t in range(1, min(80, n)):
        gap = (pd.Timestamp(dates[t]) - pd.Timestamp(dates[t - 1])).days
        if gap > 730 or (gap > 150 and (t - start) < 25):   # multi-year gap, or small stub then a gap
            start = t
    return df.iloc[start:].reset_index(drop=True) if start > 0 else df


def _repair_isolated_spikes(px, passes=5):
    px = px.astype(float).copy(); n_fixed = 0
    for _ in range(passes):
        if len(px) < 3:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            up_prev = px[1:-1] / px[:-2]; up_next = px[1:-1] / px[2:]
        bad = ((up_prev > 2) & (up_next > 2)) | ((up_prev < 0.5) & (up_next < 0.5))
        bad &= np.isfinite(up_prev) & np.isfinite(up_next)
        if not bad.any():
            break
        idx = np.where(bad)[0] + 1
        px[idx] = np.sqrt(px[idx - 1] * px[idx + 1]); n_fixed += int(bad.sum())
    return px, n_fixed


def _load_calendar(nse_csv, yf_csv, legacy_csv):
    """ticker -> list of dict(ex_date(date), factor(float|None), source, win)."""
    cal = defaultdict(list)
    if nse_csv and os.path.exists(nse_csv):
        for row in csv.DictReader(open(nse_csv)):
            if not row.get("ex_date"):
                continue
            try:
                d = datetime.strptime(row["ex_date"], "%Y-%m-%d").date()
            except Exception:
                continue
            f = float(row["factor"]) if row.get("factor") else None
            cal[row["symbol"]].append({"ex": d, "f": f, "src": "nse", "win": SRC_WIN})
    if yf_csv and os.path.exists(yf_csv):
        for r in csv.reader(open(yf_csv)):
            if len(r) < 3 or not r[1] or r[1] in ("ERR",):
                continue
            try:
                d = datetime.strptime(r[1], "%Y-%m-%d").date(); ratio = float(r[2])
            except Exception:
                continue
            if ratio > 0:
                cal[r[0]].append({"ex": d, "f": 1.0 / ratio, "src": "yf", "win": SRC_WIN})
    if legacy_csv and os.path.exists(legacy_csv):
        df = pd.read_csv(legacy_csv)
        for _, row in df.iterrows():
            try:
                d = pd.to_datetime(row["ex_date"]).date()
            except Exception:
                continue
            cal[row["ticker"]].append({"ex": d, "f": None, "src": "legacy", "win": LEGACY_WIN})
    return cal


def _confirm_drop(dates, close, ex_date, win):
    """Largest real drop within `win` trading days of ex_date -> (idx, ratio) or None."""
    i0 = int(np.searchsorted(dates, np.datetime64(ex_date), side="left"))
    best = None
    for t in range(max(1, i0 - win), min(len(close), i0 + win + 1)):
        if close[t - 1] <= 0:
            continue
        r = close[t] / close[t - 1]
        if 0 < r < (best[1] if best else 1.0):
            best = (t, r)
    return best if best and best[1] < 0.94 else None


def detect(sym, dates, close, cal):
    """Return {row_idx: (factor, provenance_str)}."""
    det = {}
    events = sorted(cal.get(sym, []), key=lambda e: e["ex"])
    # cluster source events that are within ~10 days (same action listed by many)
    used_drops = {}
    for ev in events:
        c = _confirm_drop(dates, close, ev["ex"], ev["win"])
        if not c:
            continue                                   # source lists a CA but price is flat -> spurious
        t, r = c
        srcs = used_drops.setdefault(t, {"r": r, "src": set(), "ratios": []})
        srcs["src"].add(ev["src"])
        if ev["f"]:
            srcs["ratios"].append(ev["f"])
    for t, info in used_drops.items():
        r = info["r"]
        ratios = [f for f in info["ratios"] if abs(np.log(f / r)) < 0.12]   # source ratio that matches the price
        if ratios:
            factor = float(np.median(ratios))           # clean, price-confirmed source ratio
        else:
            R, err = _snap(r)
            factor = R if err < 0.06 else r              # snap to standard ratio else raw drop
        det[t] = (factor, "+".join(sorted(info["src"])))
    # PASS B: extreme unlisted drops
    n = len(close)
    for t in range(1, n):
        if t in det or close[t - 1] <= 0:
            continue
        r = close[t] / close[t - 1]
        if not (0 < r <= EXTREME_DROP):
            continue
        R, err = _snap(r)
        if err > 0.05:
            continue
        fwd = close[t + 1:t + 6] / close[t - 1] if t + 1 < n else np.array([1.0])
        if fwd.size and np.nanmax(fwd) > REVERT:
            continue
        det[t] = (r, "price-extreme")
    return det


def clean_symbol(path, cal):
    df = pd.read_parquet(path)
    if df.empty:
        return None, None
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    df = _trim_leading_stubs(df)
    sym = os.path.basename(path)[:-8].replace("_", ".")
    dates = df["date"].to_numpy()
    orig = df["close"].to_numpy(dtype=float)
    close, n_spikes = _repair_isolated_spikes(orig)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(orig > 0, close / orig, 1.0)
    corr[~np.isfinite(corr)] = 1.0
    for col in ("open", "high", "low"):
        df[col] = df[col].to_numpy(dtype=float) * corr
    df["close"] = close

    det = detect(sym, dates, close, cal)
    cf = np.ones(len(df))
    for idx, (factor, _) in det.items():
        cf[:idx] *= factor
    df["adjClose"] = close * cf

    raw_mult = close[-1] / close[0] if close[0] > 0 else np.nan
    clean_mult = df["adjClose"].iloc[-1] / df["adjClose"].iloc[0] if df["adjClose"].iloc[0] > 0 else np.nan
    share_mult = clean_mult / raw_mult if raw_mult and np.isfinite(raw_mult) else np.nan
    adjv = df["adjClose"].to_numpy(dtype=float); valid = np.isfinite(adjv) & (adjv > 0)
    av = adjv[valid]; resid = 0.0
    if len(av) > 1:
        ca_rows = set(det); ret = np.abs(np.log(av[1:] / av[:-1])); oi = np.where(valid)[0]
        for k in range(len(ret)):
            if oi[k + 1] not in ca_rows:
                resid = max(resid, float(ret[k]))
    cas = sorted((pd.Timestamp(dates[i]).date().isoformat(), round(f, 4), p) for i, (f, p) in det.items())
    audit = {
        "symbol": sym, "n_bars": len(df), "n_cas": len(det), "n_spikes_repaired": int(n_spikes),
        "raw_mult": round(float(raw_mult), 2) if np.isfinite(raw_mult) else None,
        "clean_mult": round(float(clean_mult), 2) if np.isfinite(clean_mult) else None,
        "share_mult": round(float(share_mult), 2) if np.isfinite(share_mult) else None,
        "max_residual_1d_move_pct": round((np.exp(resid) - 1) * 100, 0),
        "review_flag": bool((np.isfinite(share_mult) and share_mult > 60) or (np.exp(resid) - 1) * 100 > 100),
        "sources_used": "+".join(sorted({p for _, _, p in cas})) if cas else "",
        "corporate_actions": "; ".join(f"{d}x{f}[{p}]" for d, f, p in cas),
    }
    return df[["date", "open", "high", "low", "close", "volume", "adjClose"]], audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nse", default="/tmp/nse_ca.csv")
    ap.add_argument("--yf", default="/tmp/yf_splits.csv")
    ap.add_argument("--legacy", default="")   # legacy file is ~54% wrong — excluded by default
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cal = _load_calendar(args.nse, args.yf, args.legacy)
    print(f"calendar: {len(cal)} tickers with candidate CAs", flush=True)
    files = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    sysbreaks = find_systematic_breaks(files)
    for sym, dts in sysbreaks.items():
        for dt in dts:
            cal[sym].append({"ex": dt, "f": None, "src": "systematic", "win": 3})
    n_sb = sum(len(v) for v in sysbreaks.values())
    print(f"systematic bhavcopy data-breaks: {n_sb} across {len(sysbreaks)} names "
          f"(e.g. {sorted(sysbreaks)[:5]})", flush=True)
    rows = []
    for k, p in enumerate(files):
        try:
            out_df, audit = clean_symbol(p, cal)
        except Exception as exc:
            print(f"  ERROR {os.path.basename(p)}: {exc}", flush=True); continue
        if out_df is None:
            continue
        out_df.to_parquet(os.path.join(args.out, os.path.basename(p)), index=False)
        rows.append(audit)
        if (k + 1) % 300 == 0:
            print(f"  {k + 1}/{len(files)} ...", flush=True)
    au = pd.DataFrame(rows).sort_values("share_mult", ascending=False, na_position="last")
    au.to_csv(os.path.join(args.out, "_cleaning_audit.csv"), index=False)
    print(f"\nDONE: {len(rows)} symbols | adjusted {int((au.n_cas>0).sum())} | flagged {int(au.review_flag.sum())}")
    print(f"  CA provenance: " + au["sources_used"].value_counts().head(8).to_dict().__repr__())


if __name__ == "__main__":
    main()
