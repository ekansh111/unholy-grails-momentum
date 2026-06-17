"""The eight Unholy Grails strategies as signal generators.

Each strategy maps a price panel -> StrategySignals: two (T, N) boolean
matrices (`entry`, `raw_exit`) plus optional position-relative stop parameters
the engine applies day by day. A signal forms on a *close*; the engine fills at
the NEXT day's open. See docs/STRATEGY_SPEC.md for the book rules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import indicators as ind


@dataclass
class StrategySignals:
    name: str
    entry: np.ndarray                       # (T, N) bool — entry breakout closes here
    raw_exit: np.ndarray                    # (T, N) bool — raw exit closes here
    initial_stop_pct: float | None = None   # exit if close < entry_price*(1-x)
    trailing_stop_pct: float | None = None  # exit if close < high-since-entry*(1-x)
    trail_ema_lows: int | None = None       # exit if close < EMA(adj_low, n)
    # The defensive action the book applies to THIS strategy when the index
    # filter turns bearish (engine overrides via --filter-mode if desired).
    default_filter_mode: str = "no_new_entries"


def _consec_true(cond: np.ndarray, k: int) -> np.ndarray:
    """True on the bar where `cond` has been true for k consecutive bars."""
    f = pd.DataFrame(cond.astype(float))
    run = f.rolling(k, min_periods=k).sum().to_numpy()
    return run >= k


# ----------------------------------------------------------------- 1. Yearly High
def yearly_high(panel, lookback: int = 250) -> StrategySignals:
    prior_high = ind.rolling_max(panel.adj_high, lookback, shift=1)
    prior_low = ind.rolling_min(panel.adj_low, lookback, shift=1)
    entry = panel.adj_close > prior_high
    raw_exit = panel.adj_close < prior_low
    return StrategySignals("yearly_high", entry, raw_exit,
                           default_filter_mode="trailing_10pct")


# ----------------------------------------------------------------- 2. 100-day High
def hundred_day_high(panel, lookback: int = 100) -> StrategySignals:
    prior_high = ind.rolling_max(panel.adj_high, lookback, shift=1)
    prior_low = ind.rolling_min(panel.adj_low, lookback, shift=1)
    entry = panel.adj_close > prior_high
    raw_exit = panel.adj_close < prior_low
    return StrategySignals("hundred_day_high", entry, raw_exit,
                           default_filter_mode="trailing_10pct")


# ----------------------------------------------------------------- 3. TrendPilot
def trendpilot(panel, ma: int = 200, confirm: int = 5) -> StrategySignals:
    s = ind.sma(panel.adj_close, ma)
    above = panel.adj_close > s
    below = panel.adj_close < s
    entry = _consec_true(above, confirm)
    raw_exit = _consec_true(below, confirm)
    return StrategySignals("trendpilot", entry, raw_exit,
                           default_filter_mode="no_new_entries")


# ----------------------------------------------------------------- 4. Golden Cross
def golden_cross(panel, fast: int = 50, slow: int = 200) -> StrategySignals:
    s_fast = ind.sma(panel.adj_close, fast)
    s_slow = ind.sma(panel.adj_close, slow)
    prev_fast = pd.DataFrame(s_fast).shift(1).to_numpy()
    prev_slow = pd.DataFrame(s_slow).shift(1).to_numpy()
    entry = (s_fast > s_slow) & (prev_fast <= prev_slow)
    raw_exit = (s_fast < s_slow) & (prev_fast >= prev_slow)
    return StrategySignals("golden_cross", entry, raw_exit,
                           default_filter_mode="exit_all_cash")


# ----------------------------------------------------------------- 5. MAC
def mac(panel, high_ma: int = 10, low_ma: int = 8, confirm: int = 5) -> StrategySignals:
    upper = ind.sma(panel.adj_high, high_ma)
    lower = ind.sma(panel.adj_low, low_ma)
    bar_above = panel.adj_low > upper     # entire bar above the channel top
    bar_below = panel.adj_high < lower    # entire bar below the channel bottom
    entry = _consec_true(bar_above, confirm)
    raw_exit = _consec_true(bar_below, confirm)
    return StrategySignals("mac", entry, raw_exit,
                           default_filter_mode="no_new_entries")


# ----------------------------------------------------------------- 6. TechTrader
def techtrader(panel, params: dict | None = None) -> StrategySignals:
    params = params or {}
    price_ceiling = params.get("price_ceiling")          # ASX '<$10' — default off
    turnover_floor = params.get("turnover_floor", 0.0)
    s40 = ind.sma(panel.adj_close, 40)
    above40 = panel.adj_close > s40
    up_day = panel.adj_close > panel.adj_open
    turnover = ind.avg_dollar_turnover(panel.close_raw, panel.volume, 21)
    liquid = turnover > turnover_floor
    high70 = panel.adj_close > ind.rolling_max(panel.adj_high, 70, shift=1)
    high10 = panel.adj_close > ind.rolling_max(panel.adj_high, 10, shift=1)
    entry = above40 & up_day & liquid & high70 & high10
    if price_ceiling is not None:
        entry = entry & (panel.adj_close < price_ceiling)
    raw_exit = np.zeros_like(entry, dtype=bool)           # exits via stops only
    return StrategySignals("techtrader", entry, raw_exit,
                           initial_stop_pct=0.10, trail_ema_lows=180,
                           default_filter_mode="no_new_entries")


# ----------------------------------------------------------------- 7. 20% Flipper
def _flipper_zigzag(adj_close, adj_high, adj_low, pct: float = 0.20):
    """20% reversal zigzag. Swing extremes track intraday high/low (the book's
    '20% below the entry day's HIGH' stop); the trigger is a CLOSE crossing the
    20% threshold off the swing low (buy) or swing high (sell)."""
    T, N = adj_close.shape
    entry = np.zeros((T, N), dtype=bool)
    raw_exit = np.zeros((T, N), dtype=bool)
    up = 1.0 + pct
    down = 1.0 - pct
    for j in range(N):
        started = False
        seeking_up = True
        low = high = 0.0
        for t in range(T):
            c = adj_close[t, j]
            hi = adj_high[t, j]
            lo = adj_low[t, j]
            if not (np.isfinite(c) and np.isfinite(hi) and np.isfinite(lo)):
                continue
            if not started:
                low, high = lo, hi
                started = True
                seeking_up = True
                continue
            if seeking_up:
                if lo < low:
                    low = lo
                if c >= low * up:
                    entry[t, j] = True
                    seeking_up = False
                    high = hi
            else:
                if hi > high:
                    high = hi
                if c <= high * down:
                    raw_exit[t, j] = True
                    seeking_up = True
                    low = lo
    return entry, raw_exit


def flipper(panel, pct: float = 0.20) -> StrategySignals:
    entry, raw_exit = _flipper_zigzag(panel.adj_close, panel.adj_high, panel.adj_low, pct)
    # raw_exit IS the always-on 20%-from-swing-high protective trail (the book's
    # stop); default_filter_mode tightens it to 10% when the index turns bearish.
    return StrategySignals("flipper", entry, raw_exit,
                           default_filter_mode="trailing_10pct")


# ----------------------------------------------------------------- 8. Bollinger
def bbo(panel, ma: int = 100, upper_sd: float = 3.0, lower_sd: float = 1.0) -> StrategySignals:
    mid = ind.sma(panel.adj_close, ma)
    sd = ind.rolling_std(panel.adj_close, ma, ddof=0)
    upper = mid + upper_sd * sd
    lower = mid - lower_sd * sd
    entry = panel.adj_close > upper
    raw_exit = panel.adj_close < lower
    return StrategySignals("bbo", entry, raw_exit,
                           default_filter_mode="no_new_entries")


# ----------------------------------------------------------------- registry
def build(name: str, panel, params: dict | None = None) -> StrategySignals:
    params = params or {}
    if name == "yearly_high":
        return yearly_high(panel)
    if name == "hundred_day_high":
        return hundred_day_high(panel)
    if name == "trendpilot":
        return trendpilot(panel)
    if name == "golden_cross":
        return golden_cross(panel)
    if name == "mac":
        return mac(panel)
    if name == "techtrader":
        return techtrader(panel, params.get("techtrader"))
    if name == "flipper":
        return flipper(panel)
    if name == "bbo":
        return bbo(panel)
    raise ValueError(f"unknown strategy {name}")


ALL_STRATEGIES = [
    "yearly_high", "hundred_day_high", "trendpilot", "golden_cross",
    "mac", "techtrader", "flipper", "bbo",
]

DISPLAY_NAMES = {
    "yearly_high": "52-Week High",
    "hundred_day_high": "100-Day High",
    "trendpilot": "TrendPilot",
    "golden_cross": "Golden Cross",
    "mac": "Moving Avg Channel",
    "techtrader": "TechTrader",
    "flipper": "20% Flipper",
    "bbo": "Bollinger Breakout",
}
