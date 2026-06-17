"""Price panel: survivorship-aware adjusted OHLCV matrices.

Reads one parquet per symbol (columns: date, open, high, low, close, volume,
adjClose) from a configurable directory, aligns every symbol onto the regime
benchmark's trading calendar, and derives fully adjusted OHLC the same way the
Clenow project does:

    factor   = adjClose / close          # back-adjusts splits AND dividends
    adjOpen  = open * factor
    adjHigh  = high * factor
    adjLow   = low  * factor

so high/low breakout signals are computed on dividend/split-adjusted candles.
Missing days are NaN (dense rectangle, never absent rows); only a 3-day
forward-filled close is kept for mark-to-market, matching the Clenow loader.

The panel is wide numpy arrays of shape (nDates, nSymbols) — no per-symbol
lookahead is possible because the engine only ever reads rows <= the signal row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from universe import Universe

FFILL_LIMIT_DAYS = 3


def _safe_name(symbol: str) -> str:
    return symbol.replace(".", "_").replace("/", "_")


def _read_symbol(data_dir: str, symbol: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, _safe_name(symbol) + ".parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df


@dataclass
class Panel:
    dates: np.ndarray            # datetime64[D], ascending trading calendar
    symbols: list[str]
    symbol_index: dict[str, int]
    adj_open: np.ndarray         # (T, N) NaN where no bar
    adj_high: np.ndarray
    adj_low: np.ndarray
    adj_close: np.ndarray
    close_raw: np.ndarray
    volume: np.ndarray
    adj_close_ffill: np.ndarray  # ffill(limit=3) — mark-to-market only
    regime_close: np.ndarray     # (T,)
    report_close: np.ndarray     # (T,) ffilled (no limit)
    start_row: int               # first row at/after config start_date (trading begins)
    py_dates: list[date]         # python date objects, len T (convenience)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def row_for(self, d) -> int:
        """Last trading row with date <= d (-1 if before the calendar)."""
        d64 = np.datetime64(pd.Timestamp(d).to_datetime64(), "D")
        return int(np.searchsorted(self.dates, d64, side="right")) - 1

    def last_valid_row_upto(self, j: int, row: int) -> int | None:
        """Most recent row <= `row` where symbol column j has a real bar."""
        col = self.adj_close[: row + 1, j]
        valid = np.where(np.isfinite(col))[0]
        if valid.size == 0:
            return None
        return int(valid[-1])


WILD_RATIO_HI = 2.0     # >100% up in a day on adjusted close
WILD_RATIO_LO = 0.5     # <-50% down in a day
RAW_TAME_HI = 1.5       # raw close "didn't really move" band — flags spurious adj-only jumps
RAW_TAME_LO = 0.66
MAX_WILD_RATIOS = 3     # quarantine if STILL this wild after causal repair (genuinely broken series)


def _repair_adjusted_close(close: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Causal repair of corrupt corporate-action adjustments.

    The corrupt fingerprint is the *adjustment factor* (adjClose/close) jumping
    in a way no split or dividend would: the adjusted return is wild
    (outside [0.5, 2.0]) AND inconsistent with the raw return — i.e. the raw
    close stayed tame (inside [0.66, 1.5]) AND the implied factor change
    (adj_ratio / raw_ratio) is itself wild. Only then do we substitute the raw
    return for that day. This keeps:
      - legit splits  (raw moves a lot -> raw NOT tame -> untouched),
      - legit dividends (small factor change -> factor NOT wild -> untouched),
      - genuine large news moves where adj and raw move together (factor ~1),
    while fixing spikes like adjClose +300% on a day the raw close fell 20%.
    The cleaned series is rebuilt by compounding repaired ratios, re-anchoring
    at the start of every contiguous run of valid bars (no carry across gaps).
    """
    n = len(adj)
    valid = np.isfinite(close) & np.isfinite(adj) & (close > 0) & (adj > 0)
    if valid.sum() == 0:
        return np.full(n, np.nan)

    def _is_corrupt(ra, rq):
        if not (np.isfinite(ra) and np.isfinite(rq) and rq > 0):
            return False
        wild = (ra > WILD_RATIO_HI) or (ra < WILD_RATIO_LO)
        tame = RAW_TAME_LO < rq < RAW_TAME_HI
        factor = ra / rq
        factor_wild = (factor > WILD_RATIO_HI) or (factor < WILD_RATIO_LO)
        return wild and tame and factor_wild

    # Fast path: no interior gaps (the common case) -> vectorized cumprod.
    if valid.all():
        with np.errstate(divide="ignore", invalid="ignore"):
            ra = adj[1:] / adj[:-1]
            rq = close[1:] / close[:-1]
            factor = np.where(rq > 0, ra / rq, np.inf)
        wild = (ra > WILD_RATIO_HI) | (ra < WILD_RATIO_LO)
        tame = (rq > RAW_TAME_LO) & (rq < RAW_TAME_HI)
        factor_wild = (factor > WILD_RATIO_HI) | (factor < WILD_RATIO_LO)
        repair = wild & tame & factor_wild & np.isfinite(rq)
        r_use = np.where(repair, rq, ra)
        clean = np.empty(n)
        clean[0] = adj[0]
        clean[1:] = adj[0] * np.cumprod(r_use)
        return clean

    # Gapped fallback: forward walk, re-anchoring at each contiguous valid run.
    clean = np.full(n, np.nan)
    prev = None
    for t in range(n):
        if not valid[t]:
            prev = None
            continue
        if prev is None:
            clean[t] = adj[t]               # re-seed a fresh segment at the real value
        else:
            ra = adj[t] / adj[prev]
            rq = close[t] / close[prev]
            r = rq if _is_corrupt(ra, rq) else ra
            clean[t] = clean[prev] * r
        prev = t
    return clean


def _wild_ratio_count(adj_close: np.ndarray) -> int:
    v = adj_close[np.isfinite(adj_close) & (adj_close > 0)]
    if v.size < 2:
        return 0
    r = v[1:] / v[:-1]
    return int(np.sum((r > WILD_RATIO_HI) | (r < WILD_RATIO_LO)))


# ----------------------------------------------------------------------------
# "Clenow" sophisticated cleaning — a faithful port of the cleaning in
# clenowMomentum/dataLoader.py (repairPrices + capRatios + countWildRatios +
# cleanAdjustedClose). Works on the adjusted-close series ALONE (it never reads
# the raw close), so it can quarantine shape-cascade corruption that a
# spike-by-spike repair cannot. Selectable via config `cleaning: clenow` so the
# Unholy Grails panel can match the Clenow baseline's exact prices bit-for-bit.
# ----------------------------------------------------------------------------
CLENOW_RATIO_LO = 1.0 / 3
CLENOW_RATIO_HI = 3.0
CLENOW_MAX_WILD = 3


def _count_wild_ratios_clenow(adj: np.ndarray) -> int:
    finite = np.where(np.isfinite(adj) & (adj > 0))[0]
    if len(finite) < 2:
        return 0
    r = adj[finite[1:]] / adj[finite[:-1]]
    return int(((r > CLENOW_RATIO_HI) | (r < CLENOW_RATIO_LO)).sum())


def _repair_prices_clenow(px: np.ndarray, passes: int = 5) -> np.ndarray:
    """Erase ISOLATED single-day spikes that revert: a day >2x both neighbours
    (or <0.5x both) -> geometric mean of its neighbours. Persistent level
    shifts are kept. Non-causal (reads the following day), as in the backtest."""
    px = px.astype(float).copy()
    px[~np.isfinite(px) | (px <= 0)] = np.nan
    if np.isnan(px).any():
        px = pd.Series(px).ffill().bfill().to_numpy(copy=True)
    for _ in range(passes):
        if len(px) < 3:
            break
        up_prev = px[1:-1] / px[:-2]
        up_next = px[1:-1] / px[2:]
        bad = ((up_prev > 2.0) & (up_next > 2.0)) | ((up_prev < 0.5) & (up_next < 0.5))
        if not bad.any():
            break
        idx = np.where(bad)[0] + 1
        px[idx] = np.sqrt(px[idx - 1] * px[idx + 1])
    return px


def _cap_ratios_clenow(px: np.ndarray, lo: float = CLENOW_RATIO_LO, hi: float = CLENOW_RATIO_HI) -> np.ndarray:
    """Rebuild the path with daily ratios clipped to [lo, hi]."""
    ratios = np.ones(len(px))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios[1:] = px[1:] / px[:-1]
    ratios[~np.isfinite(ratios)] = 1.0
    ratios = np.clip(ratios, lo, hi)
    base = px[0] if (np.isfinite(px[0]) and px[0] > 0) else 1.0
    return base * np.cumprod(ratios)


def _clean_adjusted_close_clenow(adj: np.ndarray) -> np.ndarray:
    """Quarantine (>3 wild daily ratios -> all-NaN), else repair isolated spikes
    and cap residual daily ratios. Mirrors Clenow's cleanAdjustedClose."""
    finite = np.isfinite(adj) & (adj > 0)
    if finite.sum() < 3:
        return np.where(finite, adj, np.nan)
    if _count_wild_ratios_clenow(adj) > CLENOW_MAX_WILD:
        return np.full_like(adj, np.nan, dtype=float)   # shape-corrupted -> refuse
    repaired = _cap_ratios_clenow(_repair_prices_clenow(adj))
    return np.where(finite, repaired, np.nan)


def _derive_adjusted(df: pd.DataFrame, cleaning: str = "factor_repair") -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=float)
    adj_raw = df["adjClose"].to_numpy(dtype=float)
    if cleaning == "clenow":
        adj = _clean_adjusted_close_clenow(adj_raw)
    else:  # "factor_repair" — raw-anchored causal repair (this repo's default)
        adj = _repair_adjusted_close(close, adj_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where((close > 0) & np.isfinite(close), adj / close, np.nan)
    out = pd.DataFrame({"date": df["date"].to_numpy()})
    out["adjOpen"] = df["open"].to_numpy(dtype=float) * factor
    out["adjHigh"] = df["high"].to_numpy(dtype=float) * factor
    out["adjLow"] = df["low"].to_numpy(dtype=float) * factor
    out["adjClose"] = adj
    out["closeRaw"] = close
    out["volume"] = df["volume"].to_numpy(dtype=float)
    # Guard against non-positive adjusted prices (bad vendor rows).
    for c in ("adjOpen", "adjHigh", "adjLow", "adjClose"):
        vals = np.array(out[c], dtype=float)   # writable copy (pandas 3 CoW)
        vals[~(vals > 0)] = np.nan
        out[c] = vals
    return out


def load_panel(config: dict, universe: Universe) -> Panel:
    data_dir = config["data_dir"]
    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    # 1) Trading calendar = regime benchmark dates, up to end_date (full history
    #    before start_date is kept for indicator warmup).
    regime_df = _read_symbol(data_dir, config["regime_symbol"])
    if regime_df is None:
        raise RuntimeError(f"regime benchmark {config['regime_symbol']} not found in {data_dir}")
    regime_df = regime_df[regime_df["date"] <= end]
    calendar = pd.DatetimeIndex(regime_df["date"].to_numpy())
    dates64 = calendar.values.astype("datetime64[D]")
    n = len(calendar)

    # 2) Symbols = union of all PIT members that actually have a parquet file.
    cleaning = config.get("cleaning", "factor_repair")
    candidate = sorted(universe.all_tickers_ever())
    symbols: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    quarantined: list[str] = []
    for sym in candidate:
        df = _read_symbol(data_dir, sym)
        if df is None:
            continue
        adj = _derive_adjusted(df, cleaning=cleaning)
        adj_close = adj["adjClose"].to_numpy(dtype=float)
        if not np.isfinite(adj_close).any():
            quarantined.append(sym)        # clenow cleaning refused a shape-corrupted series
            continue
        if cleaning != "clenow" and _wild_ratio_count(adj_close) > MAX_WILD_RATIOS:
            quarantined.append(sym)        # bad vendor series — exclude from all selection/valuation
            continue
        symbols.append(sym)
        frames[sym] = adj
    if quarantined:
        print(f"  cleaning={cleaning}: quarantined {len(quarantined)} symbols")
    symbol_index = {s: i for i, s in enumerate(symbols)}
    m = len(symbols)

    # 3) Allocate matrices and fill by positional alignment onto the calendar.
    def empty():
        return np.full((n, m), np.nan, dtype=float)

    adj_open, adj_high, adj_low, adj_close = empty(), empty(), empty(), empty()
    close_raw, volume = empty(), empty()
    for sym in symbols:
        j = symbol_index[sym]
        df = frames[sym]
        pos = calendar.get_indexer(pd.DatetimeIndex(df["date"].to_numpy()))
        ok = pos >= 0
        pj = pos[ok]
        adj_open[pj, j] = df["adjOpen"].to_numpy()[ok]
        adj_high[pj, j] = df["adjHigh"].to_numpy()[ok]
        adj_low[pj, j] = df["adjLow"].to_numpy()[ok]
        adj_close[pj, j] = df["adjClose"].to_numpy()[ok]
        close_raw[pj, j] = df["closeRaw"].to_numpy()[ok]
        volume[pj, j] = df["volume"].to_numpy()[ok]

    adj_close_ffill = (
        pd.DataFrame(adj_close).ffill(limit=FFILL_LIMIT_DAYS).to_numpy()
    )

    # 4) Benchmark series (regime + report), aligned to the calendar.
    regime_close = regime_df.set_index("date")["close"].reindex(calendar).to_numpy(dtype=float)
    report_df = _read_symbol(data_dir, config["report_symbol"])
    if report_df is not None:
        rc = report_df.set_index("date")["adjClose"].reindex(calendar)
        # cap ffill at 5 days so a stale/halted benchmark tail goes NaN (and is
        # dropped from scoring) rather than flat-lining the benchmark curve
        report_close = rc.ffill(limit=5).to_numpy(dtype=float)
    else:
        report_close = np.full(n, np.nan)

    start_row = int(np.searchsorted(dates64, np.datetime64(start.to_datetime64(), "D"), side="left"))
    start_row = min(start_row, n - 1)

    py_dates = [pd.Timestamp(d).date() for d in calendar]

    return Panel(
        dates=dates64, symbols=symbols, symbol_index=symbol_index,
        adj_open=adj_open, adj_high=adj_high, adj_low=adj_low, adj_close=adj_close,
        close_raw=close_raw, volume=volume, adj_close_ffill=adj_close_ffill,
        regime_close=regime_close, report_close=report_close,
        start_row=start_row, py_dates=py_dates,
    )
