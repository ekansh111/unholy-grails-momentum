"""Vectorized rolling indicators over (T, N) panel arrays.

All use pandas rolling with ``min_periods == window`` so any symbol without a
full window of real bars yields NaN there (ineligible), matching the Clenow
convention. Donchian channels are shifted by one bar so a *breakout* compares
today's close against the PRIOR window's extreme (no same-bar lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _df(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(a)


def rolling_max(a: np.ndarray, window: int, shift: int = 0) -> np.ndarray:
    r = _df(a).rolling(window, min_periods=window).max()
    if shift:
        r = r.shift(shift)
    return r.to_numpy()


def rolling_min(a: np.ndarray, window: int, shift: int = 0) -> np.ndarray:
    r = _df(a).rolling(window, min_periods=window).min()
    if shift:
        r = r.shift(shift)
    return r.to_numpy()


def sma(a: np.ndarray, window: int) -> np.ndarray:
    return _df(a).rolling(window, min_periods=window).mean().to_numpy()


def ema(a: np.ndarray, window: int) -> np.ndarray:
    # NaN-aware EMA: ignore_na=False keeps the decay anchored to real spacing.
    return _df(a).ewm(span=window, min_periods=window, adjust=False).mean().to_numpy()


def rolling_std(a: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    return _df(a).rolling(window, min_periods=window).std(ddof=ddof).to_numpy()


def avg_dollar_turnover(close_raw: np.ndarray, volume: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling mean of raw close x raw volume (unadjusted shares)."""
    dollar = close_raw * volume
    return _df(dollar).rolling(window, min_periods=window).mean().to_numpy()


def sma_series(a: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a).rolling(window, min_periods=window).mean().to_numpy()


def true_range(adj_high: np.ndarray, adj_low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    hl = adj_high - adj_low
    hc = np.abs(adj_high - prev_close)
    lc = np.abs(adj_low - prev_close)
    return np.fmax(hl, np.fmax(hc, lc))


def atr(adj_high: np.ndarray, adj_low: np.ndarray, adj_close: np.ndarray, window: int = 20) -> np.ndarray:
    prev_close = _df(adj_close).shift(1).to_numpy()
    tr = true_range(adj_high, adj_low, prev_close)
    return _df(tr).rolling(window, min_periods=window).mean().to_numpy()
