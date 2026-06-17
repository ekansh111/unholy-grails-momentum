"""Performance metrics — defined to match the Clenow reports and the book.

CAGR over 365.25-day years; vol = daily std x sqrt(252); Sharpe uses a CAGR
numerator with rf = 0 (Clenow convention, with the book's caveat that upside
volatility is penalised); MaxDD from the running-peak drawdown; MAR =
CAGR/|MaxDD|. Win rate / payoff / expectancy are on closed round-trip episodes
(the book's basis), which the Clenow reports omit — added here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _cagr(equity: pd.Series) -> float:
    first, last = equity.index[0], equity.index[-1]
    years = (pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def equity_metrics(equity: pd.Series, drawdown: pd.Series | None = None) -> dict:
    """Core return/risk metrics from an equity series (shared by UG + Clenow)."""
    equity = equity.dropna()
    daily_ret = equity.pct_change().dropna()
    cagr = _cagr(equity)
    ann_vol = float(daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(daily_ret) > 1 else float("nan")
    if drawdown is None:
        drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    mar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    sharpe = cagr / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else float("nan")
    return {"CAGR": cagr, "MaxDD": max_dd, "MAR": mar, "ann_vol": ann_vol, "Sharpe": sharpe}


def summarize(result, panel, config: dict) -> dict:
    eq = result.equity_curve
    equity = eq["equity"]
    daily_ret = equity.pct_change().dropna()
    years = (pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).days / 365.25

    cagr = _cagr(equity)
    ann_vol = float(daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(daily_ret) > 1 else float("nan")
    max_dd = float(eq["drawdown"].min())
    mar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    sharpe = cagr / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else float("nan")
    exposure = float(eq["exposure"].mean())
    avg_positions = float(eq["n_positions"].mean())

    # closed-episode stats (exclude the forced end-of-test liquidation)
    ep = result.episodes
    closed = ep[ep["reason"] != "endOfTest"] if not ep.empty else ep
    n_trades = int(len(closed))
    if n_trades:
        # losers are strictly < 0; exact-zero scratch trades count toward the
        # win-rate denominator but contaminate neither avg_win nor avg_loss.
        wins = closed[closed["net_pnl"] > 0]
        losses = closed[closed["net_pnl"] < 0]
        win_rate = len(wins) / n_trades
        avg_win = float(wins["return_pct"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["return_pct"].mean()) if len(losses) else 0.0
        payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("nan")
        expectancy = float(closed["return_pct"].mean())
        avg_hold = float(closed["hold_days"].mean())
    else:
        win_rate = avg_win = avg_loss = payoff = expectancy = avg_hold = float("nan")

    trades_per_year = n_trades / years if years > 0 else float("nan")

    out = {
        "strategy": result.summary["strategy"],
        "filter_mode": result.summary["filter_mode"],
        "start": equity.index[0],
        "end": equity.index[-1],
        "years": round(years, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "CAGR": cagr,
        "MaxDD": max_dd,
        "MAR": mar,
        "ann_vol": ann_vol,
        "Sharpe": sharpe,
        "exposure": exposure,
        "avg_positions": round(avg_positions, 2),
        "win_rate": win_rate,
        "payoff": payoff,
        "expectancy": expectancy,
        "avg_hold_days": avg_hold,
        "n_trades": n_trades,
        "trades_per_year": round(trades_per_year, 1),
        "total_commission": round(result.summary["total_commission"], 2),
    }
    # benchmark (Buy & Hold of the report index) over the same window
    bench = pd.Series(panel.report_close, index=panel.py_dates)
    bench = bench.loc[(bench.index >= equity.index[0]) & (bench.index <= equity.index[-1])].dropna()
    if len(bench) > 1:
        b_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (365.25 / (pd.Timestamp(bench.index[-1]) - pd.Timestamp(bench.index[0])).days) - 1.0
        b_peak = bench.cummax()
        b_dd = float((bench / b_peak - 1.0).min())
        out["bench_CAGR"] = b_cagr
        out["bench_MaxDD"] = b_dd
        out["bench_MAR"] = b_cagr / abs(b_dd) if b_dd < 0 else float("nan")
        out["bench_start"] = bench.index[0]
    else:
        out["bench_CAGR"] = out["bench_MaxDD"] = out["bench_MAR"] = float("nan")
        out["bench_start"] = None
    return out


def format_pct(x) -> str:
    return f"{x*100:.1f}%" if x is not None and np.isfinite(x) else "n/a"
