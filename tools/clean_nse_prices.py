"""Strictly rebuild NSE adjusted-close from the raw bhavcopy price + confirmed
corporate actions, and write a separate cleaned dataset (originals untouched).

WHY: the vendor `adjClose` in the NSE panels is badly over-adjusted — it applies
a corporate-actions file in which ~54% of the bonus/split entries are spurious
(no real price drop) and many factors/dates are wrong. That fabricates multi-
hundred-x phantom multibaggers (CUB 12667x, SUPREMEIND 4830x) that corrupt
momentum signals.

METHOD (conservative, raw-price-anchored):
  - The raw bhavcopy close is the ground truth (actual traded price).
  - A corporate action is APPLIED only when ALL hold:
      * a large one-day raw drop (>= ~15%, factor <= 0.85),
      * that snaps to a standard split/bonus ratio (within 6%),
      * within a forward-biased window of a corp-action listed for that ticker
        (the file is used only as a soft prior for WHICH ticker/period to look),
      * and the drop PERSISTS (not a bad print that recovers near the pre-drop
        level within 5 days).
    The factor used is the actual raw drop (absorbs the day's market move).
  - Adjusted close is back-adjusted so the LAST bar equals the last raw close.
  - Everything unconfirmed is left unadjusted (conservative: under-adjust rather
    than fabricate). Small bonuses (<15%) are indistinguishable from market noise
    on daily closes and are intentionally NOT applied; some real CAs missing from
    the file (e.g. a split listed nowhere) are therefore missed — see the audit.

OUTPUT: per-symbol parquets (same schema, only adjClose rebuilt) in
  <out_dir>, plus _cleaning_audit.csv. Point your loader's cacheDir there to use
  it; the original pricesNse directory is never modified.

Usage:
  python tools/clean_nse_prices.py \
    --src /Users/ekanshgowda/Documents/Code/clenowMomentum/data/raw/pricesNse \
    --ca  /Users/ekanshgowda/Documents/Code/Kite_API/nifty500_momentum/data/corporate_actions/nifty500_corp_actions.csv \
    --out /Users/ekanshgowda/Documents/Code/clenowMomentum/data/raw/pricesNseClean
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Standard split/bonus price-multipliers (a real CA drops the raw price to one of
# these). Only "large" ones (<= 0.85) are used — smaller CAs can't be told apart
# from ordinary daily moves on close-only data.
MAJOR = np.array(sorted({0.833, 0.8, 0.75, 0.667, 0.6, 0.5, 0.4, 0.333, 0.286,
                         0.25, 0.2, 0.167, 0.143, 0.125, 0.1, 0.0667, 0.05}))
DROP_MAX = 0.85        # a CA day's raw ratio must be <= this (>= ~15% drop)
EXTREME_DROP = 0.45    # >= ~55% drop: never a real one-day move on a banded NSE stock
SNAP_TOL = 0.06        # raw drop must be within 6% of a standard CA ratio
REVERT = 0.93          # if price recovers above this x pre-drop within 5d -> bad print
WIN_BACK, WIN_FWD = 10, 75   # search window (trading days) around a file ex-date
FLAG_SHARE_MULT = 40   # audit flag for manual review


def _snap(r: float):
    j = int(np.argmin(np.abs(np.log(MAJOR / r))))
    return MAJOR[j], abs(r / MAJOR[j] - 1.0)


def _repair_isolated_spikes(px: np.ndarray, passes: int = 5):
    """Replace a bad single-day print (>2x BOTH neighbours, or <0.5x both) with
    the geometric mean of its neighbours. A real split/bonus is NOT isolated (it
    persists vs the next day), so it is preserved. Returns (repaired, n_fixed)."""
    px = px.astype(float).copy()
    n_fixed = 0
    for _ in range(passes):
        if len(px) < 3:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            up_prev = px[1:-1] / px[:-2]
            up_next = px[1:-1] / px[2:]
        bad = ((up_prev > 2.0) & (up_next > 2.0)) | ((up_prev < 0.5) & (up_next < 0.5))
        bad &= np.isfinite(up_prev) & np.isfinite(up_next)
        if not bad.any():
            break
        idx = np.where(bad)[0] + 1
        px[idx] = np.sqrt(px[idx - 1] * px[idx + 1])
        n_fixed += int(bad.sum())
    return px, n_fixed


def _strip_phantom_calendar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-trading-day artifacts carried in verbatim from the per-ticker
    cache: weekend (Sat/Sun) rows that merely DUPLICATE the prior bar (the 2024
    yfinance gap-fill duplicated each prior Friday onto ~8 Sundays across ~940
    names; genuine special weekend sessions like the Sat 2025-02-01 / Sun
    2026-02-01 Budget sessions carry distinct prices and are KEPT), and zero /
    non-positive OHLC fields with a valid close -> clamp into the close and
    re-derive high/low so the bar's true range is sane (no fabricated ATR spike).
    df must already be date-sorted. Returns the cleaned, re-indexed frame."""
    d = df
    dow = d["date"].dt.dayofweek
    ohlcv = [c for c in ("open", "high", "low", "close", "volume") if c in d.columns]
    dup_prev = d[ohlcv].eq(d[ohlcv].shift()).all(axis=1) if ohlcv else False
    d = d[~(dow.isin([5, 6]) & dup_prev)].reset_index(drop=True)
    if {"open", "high", "low", "close"}.issubset(d.columns):
        bad = (d["close"] > 0) & ((d["open"] <= 0) | (d["high"] <= 0) | (d["low"] <= 0))
        if bad.any():
            for col in ("open", "high", "low"):
                m = bad & (d[col] <= 0)
                d.loc[m, col] = d.loc[m, "close"]
            d.loc[bad, "high"] = d.loc[bad, ["open", "high", "low", "close"]].max(axis=1)
            d.loc[bad, "low"] = d.loc[bad, ["open", "high", "low", "close"]].min(axis=1)
    return d


def load_ca(path: str) -> dict[str, list]:
    try:
        ca = pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return {}                                   # no/empty CA file -> adjust nothing
    if ca.empty or "ex_date" not in ca.columns:
        return {}
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    return {t: sorted(set(g["ex_date"])) for t, g in ca.groupby("ticker")}


def detect_cas(ex_dates: list, dates: np.ndarray, close: np.ndarray) -> dict[int, float]:
    """Return {row_index: factor} of confirmed corporate actions: for each
    corporate action the file lists for this ticker, the single best large clean
    persistent raw drop within a forward-biased window of it. Factor = the actual
    raw drop. Conservative — unconfirmed/small CAs are left unadjusted."""
    n = len(close)
    det: dict[int, float] = {}
    for ex in ex_dates:
        i0 = int(np.searchsorted(dates, np.datetime64(ex), side="left"))
        best = None
        for t in range(max(1, i0 - WIN_BACK), min(n, i0 + WIN_FWD)):
            if close[t - 1] <= 0:
                continue
            r = close[t] / close[t - 1]
            if not (0 < r <= DROP_MAX):
                continue
            _, err = _snap(r)
            if err > SNAP_TOL:
                continue
            fwd = close[t + 1:t + 6] / close[t - 1] if t + 1 < n else np.array([1.0])
            if fwd.size and np.nanmax(fwd) > REVERT:
                continue                       # recovered near pre-drop -> bad print
            if best is None or err < best[1]:
                best = (t, err, r)
        if best:
            det[best[0]] = best[2]             # factor = actual raw drop

    # Extreme standalone: a clean, persistent one-day drop beyond ~55%
    # (<= EXTREME_DROP) is physically impossible as a real move on a circuit-
    # banded NSE stock — it is a large split/bonus even if the file omits it.
    # This is well below the 1:1-bonus/crash ambiguity zone (0.5), so it cannot
    # mis-adjust a real crash.
    for t in range(1, n):
        if t in det or close[t - 1] <= 0:
            continue
        r = close[t] / close[t - 1]
        if not (0 < r <= EXTREME_DROP):
            continue
        _, err = _snap(r)
        if err > 0.03:
            continue
        fwd = close[t + 1:t + 6] / close[t - 1] if t + 1 < n else np.array([1.0])
        if fwd.size and np.nanmax(fwd) > REVERT:
            continue
        det[t] = r
    return det


def clean_symbol(path: str, ca_map: dict) -> tuple[pd.DataFrame | None, dict]:
    df = pd.read_parquet(path)
    if df.empty:
        return None, {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    df = _strip_phantom_calendar_rows(df)
    sym = os.path.basename(path)[:-8].replace("_", ".")
    dates = df["date"].to_numpy()

    # 1) repair isolated raw bad-prints (fat-finger spikes); scale OHL with the
    #    same correction so candles stay consistent. Real CAs are not isolated.
    orig_close = df["close"].to_numpy(dtype=float)
    close, n_spikes = _repair_isolated_spikes(orig_close)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(orig_close > 0, close / orig_close, 1.0)
    corr[~np.isfinite(corr)] = 1.0
    for col in ("open", "high", "low"):
        df[col] = df[col].to_numpy(dtype=float) * corr
    df["close"] = close

    # 2) confirmed corporate-action back-adjustment
    det = detect_cas(ca_map.get(sym, []), dates, close)
    cf = np.ones(len(df))
    for idx, factor in det.items():
        cf[:idx] *= factor                     # back-adjust pre-ex prices
    df["adjClose"] = close * cf                 # last bar == last raw close (cf[last]=1)

    raw_mult = close[-1] / close[0] if close[0] > 0 else np.nan
    clean_mult = df["adjClose"].iloc[-1] / df["adjClose"].iloc[0] if df["adjClose"].iloc[0] > 0 else np.nan
    share_mult = clean_mult / raw_mult if raw_mult and np.isfinite(raw_mult) else np.nan
    # largest residual single-day move NOT explained by a detected CA (raw-data
    # artifact or an undetected split) — surfaced so it can be reviewed.
    adjv = df["adjClose"].to_numpy(dtype=float)
    valid = np.isfinite(adjv) & (adjv > 0)
    av = adjv[valid]
    resid = 0.0
    if len(av) > 1:
        ca_rows = set(det.keys())
        ret = np.abs(np.log(av[1:] / av[:-1]))
        orig_idx = np.where(valid)[0]
        for k in range(len(ret)):
            if orig_idx[k + 1] not in ca_rows:
                resid = max(resid, float(ret[k]))
    resid_pct = round((np.exp(resid) - 1) * 100, 0)
    cas = sorted((pd.Timestamp(dates[i]).date().isoformat(), round(float(f), 3)) for i, f in det.items())
    audit = {
        "symbol": sym, "n_bars": len(df), "last_date": pd.Timestamp(dates[-1]).date().isoformat(),
        "n_cas": len(det), "n_spikes_repaired": int(n_spikes),
        "raw_mult": round(float(raw_mult), 2) if np.isfinite(raw_mult) else None,
        "clean_mult": round(float(clean_mult), 2) if np.isfinite(clean_mult) else None,
        "share_mult": round(float(share_mult), 2) if np.isfinite(share_mult) else None,
        "max_residual_1d_move_pct": resid_pct,
        "review_flag": bool((np.isfinite(share_mult) and share_mult > FLAG_SHARE_MULT) or resid_pct > 100),
        "corporate_actions": "; ".join(f"{d}x{f}" for d, f in cas),
    }
    return df[["date", "open", "high", "low", "close", "volume", "adjClose"]], audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ca", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ca_map = load_ca(args.ca)
    files = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    rows = []
    for k, p in enumerate(files):
        try:
            out_df, audit = clean_symbol(p, ca_map)
        except Exception as exc:                # never let one bad file stop the run
            print(f"  ERROR {os.path.basename(p)}: {exc}", flush=True)
            continue
        if out_df is None:
            continue
        out_df.to_parquet(os.path.join(args.out, os.path.basename(p)), index=False)
        rows.append(audit)
        if (k + 1) % 300 == 0:
            print(f"  {k + 1}/{len(files)} ...", flush=True)
    audit_df = pd.DataFrame(rows).sort_values("share_mult", ascending=False, na_position="last")
    audit_df.to_csv(os.path.join(args.out, "_cleaning_audit.csv"), index=False)
    # Coverage-cliff guard: >=30 names sharing one pre-cutoff last-date = a refresh
    # cliff (the 2024-11-01 case), distinct from scattered genuine delistings.
    # Sanitizing can't recreate missing bars — surface + list them for a re-fetch.
    ld = pd.to_datetime(audit_df["last_date"]); gmax = ld.max()
    recent = ld[ld < (gmax - pd.Timedelta(days=30))].dt.date.value_counts()
    clustered = recent[recent >= 30]            # >=30 names sharing one exact date = a refresh cliff
    if len(clustered):
        cliff_dates = set(clustered.index)
        stranded = audit_df[pd.to_datetime(audit_df["last_date"]).dt.date.isin(cliff_dates)]
        stranded.to_csv(os.path.join(args.out, "_coverage_stranded.csv"), index=False)
        print(f"  !! COVERAGE CLIFF: {len(stranded)} live names truncated on a shared date "
              f"{dict(clustered)} (latest bar = {gmax.date()}) — incomplete upstream refresh / "
              f"renamed symbols, NOT delistings. Re-fetch _coverage_stranded.csv.", flush=True)
    n_adj = int((audit_df["n_cas"] > 0).sum())
    n_flag = int(audit_df["review_flag"].sum())
    print(f"\nDONE: wrote {len(rows)} cleaned symbols to {args.out}")
    print(f"  adjusted for >=1 corporate action: {n_adj}; flagged for review: {n_flag}")
    print(f"  audit: {os.path.join(args.out, '_cleaning_audit.csv')}")


if __name__ == "__main__":
    main()
