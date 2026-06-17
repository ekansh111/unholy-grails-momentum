"""Daily event-driven backtest engine for the Unholy Grails strategies.

Accounting conventions are copied from the Clenow project's `backtestEngine`
so the two systems are directly comparable:

  - signal on day t's close -> order fills at day t+1's OPEN
  - SELL fill = open*(1 - slipBps/1e4); BUY fill = open*(1 + slipBps/1e4)
  - flat commission per ticket, waived down to gross on scrap
  - sells settle before buys; cash can never go negative (buys trimmed to cash)
  - a held name with no bar for >5 trading days is liquidated at its last
    adjusted close (postdictive delisting exit, per the book)
  - equity is marked to market DAILY at the close (honest drawdowns)

Portfolio: up to `max_positions` equal-weight single-lot positions (no
pyramiding / no resizing — the book buys one lot at entry). When more entry
signals appear than free slots, candidates are ranked by 21-day average
turnover (most tradable first). The 75-day index filter gates entries and adds
the book's defensive exit when the regime is bearish.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
from strategies import StrategySignals

DELIST_GRACE_DAYS = 5
MAX_FILL_RETRY_DAYS = 5
REGIME_MA_DAYS = 75
TURNOVER_WINDOW = 21


@dataclass
class Position:
    shares: float
    entry_price: float          # fill price incl. slippage
    entry_row: int
    high_since_entry: float     # highest adjusted high since entry (for trailing stop)
    buy_commission: float


@dataclass
class PendingOrder:
    symbol: str
    side: str                   # BUY | SELL
    shares: float
    reason: str
    rank: float
    retries: int = 0


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    episodes: pd.DataFrame
    summary: dict


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def run_backtest(panel, universe, signals: StrategySignals, config: dict,
                 filter_mode: str = "none", rng=None) -> BacktestResult:
    """filter_mode: none | no_new_entries | trailing_10pct | exit_all_cash.

    rng: if a numpy Generator is passed, same-day entry candidates are chosen at
    RANDOM among eligible signals (position-variability / Monte Carlo). Default
    (None) ranks deterministically by 21-day turnover (most tradable first).
    """
    slip = config["slippage_bps"] / 10000.0
    commission = float(config["commission_per_trade"])
    max_positions = int(config["max_positions"])
    liquidity_floor = float(config.get("liquidity_floor", 0.0))
    initial_capital = float(config["initial_capital"])

    adj_open = panel.adj_open
    adj_close = panel.adj_close
    adj_high = panel.adj_high
    adj_close_ffill = panel.adj_close_ffill

    # regime (index filter)
    regime_sma = ind.sma_series(panel.regime_close, REGIME_MA_DAYS)
    regime_bullish = np.isfinite(regime_sma) & (panel.regime_close > regime_sma)

    turnover = ind.avg_dollar_turnover(panel.close_raw, panel.volume, TURNOVER_WINDOW)
    ema_lows = (ind.ema(panel.adj_low, signals.trail_ema_lows)
                if signals.trail_ema_lows else None)

    entry = signals.entry
    raw_exit = signals.raw_exit
    sym = panel.symbols
    sidx = panel.symbol_index
    T = panel.n_dates
    start = panel.start_row

    cash = initial_capital
    positions: dict[str, Position] = {}
    pending: list[PendingOrder] = []

    trades: list[dict] = []
    episodes: list[dict] = []
    eq_rows: list[dict] = []

    total_commission = 0.0
    total_slippage = 0.0

    def member_mask(row: int) -> set:
        return universe.members_on(panel.py_dates[row])

    def close_episode(symbol: str, pos: Position, sell_price: float,
                      sell_commission: float, row: int, reason: str):
        gross_pnl = (sell_price - pos.entry_price) * pos.shares
        net_pnl = gross_pnl - pos.buy_commission - sell_commission
        invested = pos.entry_price * pos.shares
        episodes.append({
            "symbol": symbol,
            "entry_date": panel.py_dates[pos.entry_row],
            "exit_date": panel.py_dates[row],
            "hold_days": row - pos.entry_row,
            "entry_price": pos.entry_price,
            "exit_price": sell_price,
            "shares": pos.shares,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / invested if invested > 0 else 0.0,
            "reason": reason,
        })

    for t in range(start, T):
        # ---- 1) fill pending orders (from t-1) at today's open ----
        still_pending: list[PendingOrder] = []
        # sells first, then buys; within a side lower rank fills first
        pending.sort(key=lambda o: (o.side != "SELL",
                                    o.rank if np.isfinite(o.rank) else math.inf))
        for o in pending:
            j = sidx[o.symbol]
            op = adj_open[t, j]
            if not _finite(op) or op <= 0:
                o.retries += 1
                if o.retries <= MAX_FILL_RETRY_DAYS:
                    still_pending.append(o)
                else:
                    trades.append({"date": panel.py_dates[t], "symbol": o.symbol,
                                   "action": "CANCELLED", "shares": o.shares,
                                   "price": float("nan"), "reason": f"unfilled:{o.reason}"})
                continue
            if o.side == "SELL":
                price = op * (1.0 - slip)
                pos = positions.get(o.symbol)
                if pos is None:
                    continue
                shares = pos.shares
                gross = shares * price
                charged = min(commission, gross)
                cash += gross - charged
                total_commission += charged
                total_slippage += shares * op * slip
                trades.append({"date": panel.py_dates[t], "symbol": o.symbol,
                               "action": "SELL", "shares": shares, "price": price,
                               "reason": o.reason})
                close_episode(o.symbol, pos, price, charged, t, o.reason)
                del positions[o.symbol]
            else:  # BUY
                # Hard cap: sells are processed before buys this same loop, so a
                # held name whose SELL filled has already vacated its slot. If the
                # book is still full (e.g. an exit's SELL failed to fill on a
                # missing open), cancel the buy — never hold > max_positions.
                if len(positions) >= max_positions:
                    trades.append({"date": panel.py_dates[t], "symbol": o.symbol,
                                   "action": "CANCELLED", "shares": 0, "price": op,
                                   "reason": f"capFull:{o.reason}"})
                    continue
                price = op * (1.0 + slip)
                shares = o.shares
                cost = shares * price + commission
                if cost > cash:
                    affordable = math.floor((cash - commission) / price) if price > 0 else 0
                    shares = max(0, affordable)
                if shares <= 0:
                    trades.append({"date": panel.py_dates[t], "symbol": o.symbol,
                                   "action": "CANCELLED", "shares": 0, "price": price,
                                   "reason": f"insufficientCash:{o.reason}"})
                    continue
                cost = shares * price + commission
                cash -= cost
                total_commission += commission
                total_slippage += shares * op * slip
                positions[o.symbol] = Position(
                    shares=shares, entry_price=price, entry_row=t,
                    high_since_entry=(adj_high[t, j] if _finite(adj_high[t, j]) else price),
                    buy_commission=commission)
                trades.append({"date": panel.py_dates[t], "symbol": o.symbol,
                               "action": "BUY", "shares": shares, "price": price,
                               "reason": o.reason})
            assert cash > -1e-6, f"cash went negative: {cash}"
        pending = still_pending

        # ---- 2) delisting liquidation (held names with stale data) ----
        for symbol in list(positions.keys()):
            j = sidx[symbol]
            last_valid = panel.last_valid_row_upto(j, t)
            if last_valid is not None and (t - last_valid) > DELIST_GRACE_DAYS:
                pos = positions[symbol]
                price = adj_close[last_valid, j]
                gross = pos.shares * price
                charged = min(commission, gross)
                cash += gross - charged
                total_commission += charged
                trades.append({"date": panel.py_dates[t], "symbol": symbol,
                               "action": "SELL", "shares": pos.shares, "price": price,
                               "reason": "delisted"})
                close_episode(symbol, pos, price, charged, t, "delisted")
                del positions[symbol]

        # ---- 3) update trailing highs for held positions ----
        for symbol, pos in positions.items():
            j = sidx[symbol]
            h = adj_high[t, j]
            if _finite(h) and h > pos.high_since_entry:
                pos.high_since_entry = h

        bullish = bool(regime_bullish[t])

        # ---- 4) exits at today's close -> queue sells for t+1 ----
        exits_this_turn: set = set()
        already_pending_sell = {o.symbol for o in pending if o.side == "SELL"}
        for symbol, pos in positions.items():
            if symbol in already_pending_sell:
                continue  # an exit order is still in flight (failed to fill) — don't double-queue
            j = sidx[symbol]
            c = adj_close[t, j]
            price_for_rules = c if _finite(c) else adj_close_ffill[t, j]
            if not _finite(price_for_rules):
                continue
            reason = None
            if bool(raw_exit[t, j]):
                reason = "rawExit"
            elif signals.initial_stop_pct and price_for_rules < pos.entry_price * (1 - signals.initial_stop_pct):
                reason = "initialStop"
            elif signals.trailing_stop_pct and price_for_rules < pos.high_since_entry * (1 - signals.trailing_stop_pct):
                reason = "trailStop"
            elif ema_lows is not None and _finite(ema_lows[t, j]) and price_for_rules < ema_lows[t, j]:
                reason = "emaTrail"
            elif not bullish and filter_mode == "exit_all_cash":
                reason = "regimeCash"
            elif not bullish and filter_mode == "trailing_10pct" and price_for_rules < pos.high_since_entry * 0.90:
                reason = "regimeTrail"
            if reason:
                exits_this_turn.add(symbol)
                pending.append(PendingOrder(symbol, "SELL", pos.shares, reason, rank=math.nan))

        # ---- 5) mark to market at close & record equity ----
        holdings_value = 0.0
        n_pos = 0
        for symbol, pos in positions.items():
            j = sidx[symbol]
            px = adj_close_ffill[t, j]
            if not _finite(px):
                lv = panel.last_valid_row_upto(j, t)
                px = adj_close[lv, j] if lv is not None else pos.entry_price
            holdings_value += pos.shares * px
            n_pos += 1
        equity = cash + holdings_value
        eq_rows.append({
            "date": panel.py_dates[t], "equity": equity, "cash": cash,
            "holdings": holdings_value, "n_positions": n_pos,
            "exposure": holdings_value / equity if equity > 0 else 0.0,
            "regime_bullish": bullish,
        })

        # ---- 6) entries at today's close -> queue buys for t+1 ----
        entries_allowed = bullish or filter_mode == "none"
        remaining_after_exits = len(positions) - len(exits_this_turn)
        pending_buys = sum(1 for o in pending if o.side == "BUY")
        free_slots = max_positions - remaining_after_exits - pending_buys
        if entries_allowed and free_slots > 0 and equity > 0:
            members = member_mask(t)
            held_or_pending = set(positions.keys()) | {o.symbol for o in pending}
            target_value = equity / max_positions
            # gather eligible candidates
            cands = []
            for symbol in members:
                j = sidx.get(symbol)
                if j is None or symbol in held_or_pending:
                    continue
                if not bool(entry[t, j]):
                    continue
                c = adj_close[t, j]
                if not _finite(c) or c <= 0:
                    continue
                tv = turnover[t, j]
                if not _finite(tv) or tv < liquidity_floor:
                    continue
                cands.append((tv, symbol, j, c))
            # rank by turnover desc (most tradable first), or randomise for MC
            if rng is not None:
                rng.shuffle(cands)
            else:
                cands.sort(key=lambda x: -x[0])
            for k, (tv, symbol, j, c) in enumerate(cands[:free_slots]):
                shares = math.floor(target_value / c)
                if shares <= 0:
                    continue
                pending.append(PendingOrder(symbol, "BUY", shares,
                                            reason="entry", rank=float(k)))

    # liquidate anything still open at the final close (for clean episode stats)
    last = T - 1
    for symbol in list(positions.keys()):
        j = sidx[symbol]
        lv = panel.last_valid_row_upto(j, last)
        price = adj_close[lv, j] if lv is not None else positions[symbol].entry_price
        pos = positions[symbol]
        close_episode(symbol, pos, price, 0.0, last, "endOfTest")
        del positions[symbol]

    equity_curve = pd.DataFrame(eq_rows).set_index("date")
    peak = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = equity_curve["equity"] / peak - 1.0

    trades_df = pd.DataFrame(trades)
    episodes_df = pd.DataFrame(episodes)

    closed = episodes_df[episodes_df["reason"] != "endOfTest"] if not episodes_df.empty else episodes_df
    summary = {
        "strategy": signals.name,
        "filter_mode": filter_mode,
        "final_equity": float(equity_curve["equity"].iloc[-1]),
        "initial_capital": initial_capital,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "n_fills": int((trades_df["action"].isin(["BUY", "SELL"])).sum()) if not trades_df.empty else 0,
        "n_closed_episodes": int(len(closed)),
    }
    return BacktestResult(equity_curve, trades_df, episodes_df, summary)
