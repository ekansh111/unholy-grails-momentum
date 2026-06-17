"""Engine + accounting invariant tests on a synthetic panel.

Run: python -m pytest tests/ -q   (from the repo root, with src/ importable)
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import Panel  # noqa: E402
from engine import run_backtest  # noqa: E402
from universe import Universe  # noqa: E402
import strategies as S  # noqa: E402

CONFIG = {
    "slippage_bps": 5, "commission_per_trade": 1.0, "max_positions": 3,
    "liquidity_floor": 0.0, "initial_capital": 100000,
}


def _synthetic_panel(n_days=400, n_sym=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.array([np.datetime64("2010-01-04") + np.timedelta64(i, "D") for i in range(n_days)],
                     dtype="datetime64[D]")
    symbols = [f"S{i}" for i in range(n_sym)]
    sidx = {s: i for i, s in enumerate(symbols)}
    # symbol 0 = clean uptrend (will trigger breakouts); rest = noisy flat
    close = np.zeros((n_days, n_sym))
    for j in range(n_sym):
        if j == 0:
            # steady uptrend: daily growth (0.5%) must exceed the intrabar high
            # premium (0.2%) so each close prints a genuine new high (breakout)
            close[:, j] = 100 * (1.005 ** np.arange(n_days))
        else:
            close[:, j] = 50 + np.cumsum(rng.normal(0, 0.3, n_days))
    close = np.maximum(close, 1.0)
    adj_open = close * (1 + rng.normal(0, 0.0005, close.shape))
    adj_high = np.maximum(close, adj_open) * 1.002
    adj_low = np.minimum(close, adj_open) * 0.998
    volume = np.full((n_days, n_sym), 1e6)
    py_dates = [pd.Timestamp(d).date() for d in dates]
    regime = pd.Series(close[:, 0]).rolling(1).mean().to_numpy()  # rising -> bullish
    return Panel(
        dates=dates, symbols=symbols, symbol_index=sidx,
        adj_open=adj_open, adj_high=adj_high, adj_low=adj_low, adj_close=close,
        close_raw=close, volume=volume,
        adj_close_ffill=pd.DataFrame(close).ffill().to_numpy(),
        regime_close=close[:, 0], report_close=close[:, 0],
        start_row=260, py_dates=py_dates,
    )


def _universe(symbols, first):
    return Universe(dates=[first], member_sets=[frozenset(symbols)])


def test_runs_and_respects_position_cap():
    panel = _synthetic_panel()
    uni = _universe(panel.symbols, panel.py_dates[0])
    sig = S.yearly_high(panel, lookback=250)
    res = run_backtest(panel, uni, sig, CONFIG, filter_mode="none")
    assert (res.equity_curve["n_positions"] <= CONFIG["max_positions"]).all()
    assert res.equity_curve["equity"].iloc[-1] > 0


def test_cash_never_negative_and_uptrend_is_bought():
    panel = _synthetic_panel()
    uni = _universe(panel.symbols, panel.py_dates[0])
    sig = S.yearly_high(panel, lookback=250)
    res = run_backtest(panel, uni, sig, CONFIG, filter_mode="none")
    # the steady uptrend (S0) must be entered at some point
    assert (res.trades["symbol"] == "S0").any()
    # equity is cash + holdings, both non-negative by construction
    assert (res.equity_curve["cash"] >= -1e-6).all()


def test_episode_pnl_identity():
    panel = _synthetic_panel()
    uni = _universe(panel.symbols, panel.py_dates[0])
    sig = S.yearly_high(panel, lookback=250)
    res = run_backtest(panel, uni, sig, CONFIG, filter_mode="none")
    ep = res.episodes
    if not ep.empty:
        recomputed = (ep["exit_price"] - ep["entry_price"]) * ep["shares"]
        # net_pnl = gross - commissions; gross within a cent of recomputed gross
        assert ((recomputed - (ep["net_pnl"] + 2 * CONFIG["commission_per_trade"])).abs()
                < ep["shares"].abs() * 0.01 + 5).all()


def test_no_lookahead_entry_fills_next_open():
    """A buy logged on day t must use open[t], driven by a signal on day t-1's close."""
    panel = _synthetic_panel()
    uni = _universe(panel.symbols, panel.py_dates[0])
    sig = S.yearly_high(panel, lookback=250)
    res = run_backtest(panel, uni, sig, CONFIG, filter_mode="none")
    buys = res.trades[(res.trades["action"] == "BUY") & (res.trades["symbol"] == "S0")]
    assert len(buys) >= 1
    d0 = buys.iloc[0]["date"]
    row = panel.py_dates.index(d0)
    j = panel.symbol_index["S0"]
    expected = panel.adj_open[row, j] * (1 + CONFIG["slippage_bps"] / 10000.0)
    assert abs(buys.iloc[0]["price"] - expected) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
