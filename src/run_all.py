"""Run all eight strategies x {raw, index-filtered} on a market and save outputs.

Usage:
    python src/run_all.py --market us
    python src/run_all.py --market india
    python src/run_all.py --market both
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import strategies as S
from cfg import load_config, output_dir
from data import load_panel
from engine import run_backtest
from metrics import summarize
from universe import Universe

MARKETS = {"us": "config/us.yaml", "india": "config/india.yaml"}


def run_market(market: str) -> pd.DataFrame:
    cfg = load_config(MARKETS[market])
    uni = Universe.from_file(cfg["constituents_file"], cfg["drop_sentinels"])
    print(f"[{market}] loading panel ...", flush=True)
    panel = load_panel(cfg, uni)
    print(f"[{market}] panel: {panel.py_dates[panel.start_row]} -> {panel.py_dates[-1]}, "
          f"{panel.n_symbols} symbols", flush=True)

    eq_dir = output_dir(market, "equity")
    rows = []
    for name in S.ALL_STRATEGIES:
        sig = S.build(name, panel)
        variants = [("raw", "none"), ("filtered", sig.default_filter_mode)]
        for variant, mode in variants:
            res = run_backtest(panel, uni, sig, cfg, filter_mode=mode)
            m = summarize(res, panel, cfg)
            m["market"] = market
            m["variant"] = variant
            m["display"] = S.DISPLAY_NAMES[name]
            rows.append(m)
            res.equity_curve[["equity", "drawdown", "exposure", "n_positions"]].to_csv(
                os.path.join(eq_dir, f"{name}_{variant}.csv"))
            print(f"  {S.DISPLAY_NAMES[name]:20} {variant:8} "
                  f"CAGR {m['CAGR']*100:5.1f}%  MaxDD {m['MaxDD']*100:6.1f}%  "
                  f"MAR {m['MAR']:.2f}  trades {m['n_trades']}", flush=True)

    # benchmark series for plotting (normalised to initial capital).
    # report index = total-return where available (US SP500TR; India Nifty500/CRSLDX
    # which only starts 2005). regime index = the index-filter benchmark, which has
    # FULL history (US GSPC, India Sensex) — the apples-to-apples full-window B&H.
    start_date = panel.py_dates[panel.start_row]
    for series, name in [(panel.report_close, "benchmark_equity.csv"),
                         (panel.regime_close, "regime_benchmark_equity.csv")]:
        b = pd.Series(series, index=panel.py_dates).dropna()
        b = b.loc[b.index >= start_date]
        if len(b):
            (b / b.iloc[0] * cfg["initial_capital"]).to_csv(
                os.path.join(output_dir(market), name), header=["equity"])

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir(market), "summary.csv"), index=False)
    print(f"[{market}] wrote summary.csv ({len(rows)} runs)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "india", "both"], default="both")
    args = ap.parse_args()
    markets = ["us", "india"] if args.market == "both" else [args.market]
    for mk in markets:
        run_market(mk)


if __name__ == "__main__":
    main()
