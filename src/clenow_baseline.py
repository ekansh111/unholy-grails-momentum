"""Run the Clenow 'Stocks on the Move' momentum baseline IN-PROCESS on the
identical data window/costs, and dump its equity curve for comparison.

This imports the (separate) clenowMomentum project directly. It must run with
the clenow directory first on sys.path, so it deliberately does NOT import this
repo's own modules (which would shadow Clenow's `universe`/`indicators`). The
saved equity CSV is later scored by compare.py with the SAME metrics function
used for the Unholy Grails strategies.

Usage:
    python src/clenow_baseline.py            # both markets
    python src/clenow_baseline.py --market us
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CLENOW_DIR = os.environ.get(
    "CLENOW_DIR", "/Users/ekanshgowda/Documents/Code/clenowMomentum")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "clenow")

# Match the Unholy Grails comparison windows exactly.
MARKETS = {
    "us": {"config": "config.yaml", "start": "1996-01-02", "end": "2026-06-16"},
    "india": {"config": "configNiftyBook.yaml", "start": "1998-09-01", "end": "2026-04-30"},
}


def _run_one(loadPanel, computeIndicators, runBacktest, parseDate, config, label, tag, market):
    print(f"[clenow:{market}:{label}] loading panel ...", flush=True)
    panel, db = loadPanel(config, endDate=parseDate(config["backtest"]["endDate"]))
    indicators = computeIndicators(panel, config)
    print(f"[clenow:{market}:{label}] running backtest ...", flush=True)
    result = runBacktest(config, panel, indicators, db)
    eq = result.equityCurve.copy()
    eq = eq.rename(columns={"totalEquity": "equity", "nPositions": "n_positions"})
    eq["exposure"] = eq["exposurePct"] / 100.0
    eq = eq.set_index("date")[["equity", "drawdown", "exposure", "n_positions"]]
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    eq.to_csv(os.path.join(OUT_DIR, f"{market}{suffix}_equity.csv"))
    s = result.summary
    tl = result.tradeLog
    n_fills = int(tl["action"].isin(["BUY", "SELL"]).sum()) if (tl is not None and "action" in tl) else None
    meta = {
        "market": market, "variant": label,
        "start": config["backtest"]["startDate"], "end": config["backtest"]["endDate"],
        "final_equity": float(eq["equity"].iloc[-1]),
        "avg_positions": float(eq["n_positions"].mean()),
        "max_positions_held": int(eq["n_positions"].max()),
        "closed_episodes": s.get("closedEpisodes"),
        "win_rate": (s.get("winningEpisodes") / s.get("closedEpisodes")
                     if s.get("closedEpisodes") else None),
        "n_fills": n_fills, "total_commission": s.get("totalCommission"),
    }
    with open(os.path.join(OUT_DIR, f"{market}{suffix}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[clenow:{market}:{label}] final {meta['final_equity']:,.0f}  "
          f"avgPos {meta['avg_positions']:.1f}  win {meta['win_rate']*100:.1f}%  "
          f"-> {market}{suffix}_equity.csv", flush=True)


def run(market: str):
    spec = MARKETS[market]
    cwd = os.getcwd()
    sys.path.insert(0, CLENOW_DIR)
    os.chdir(CLENOW_DIR)
    try:
        from utils import loadConfig, parseDate
        from dataLoader import loadPanel
        from strategyEngine import computeIndicators
        from backtestEngine import runBacktest

        base = loadConfig(spec["config"])
        base["backtest"]["startDate"] = spec["start"]
        base["backtest"]["endDate"] = spec["end"]
        # 1) Clenow as the book defines it (uncapped, buys down the ranking).
        _run_one(loadPanel, computeIndicators, runBacktest, parseDate, base,
                 "book (uncapped)", "", market)
        # 2) Clenow capped at 20 positions — like-for-like vs the 20-name Unholy
        #    Grails portfolios, to isolate signal quality from diversification.
        capped = loadConfig(spec["config"])
        capped["backtest"]["startDate"] = spec["start"]
        capped["backtest"]["endDate"] = spec["end"]
        capped.setdefault("portfolio", {})["maxPositions"] = 20
        _run_one(loadPanel, computeIndicators, runBacktest, parseDate, capped,
                 "20-position cap", "cap20", market)
    finally:
        os.chdir(cwd)
        sys.path.remove(CLENOW_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "india", "both"], default="both")
    args = ap.parse_args()
    markets = ["us", "india"] if args.market == "both" else [args.market]
    for mk in markets:
        run(mk)


if __name__ == "__main__":
    main()
