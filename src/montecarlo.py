"""Monte Carlo over position variability (the book's robustness check).

When a strategy fires more entry signals than there are free slots, which names
you actually take is path-dependent. We re-run the full backtest N times,
choosing same-day candidates at RANDOM each run, and look at the distribution
of CAGR / MaxDD / terminal wealth. A tight cluster => robust (low dependence on
which signals you happened to take); a broad cluster => fragile.

Usage:
    python src/montecarlo.py --market us --strategy hundred_day_high --variant filtered --runs 60
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import strategies as S
from cfg import load_config, output_dir
from data import load_panel
from engine import run_backtest
from metrics import summarize
from universe import Universe

MARKETS = {"us": "config/us.yaml", "india": "config/india.yaml"}


def run_mc(market: str, strategy: str, variant: str, runs: int):
    cfg = load_config(MARKETS[market])
    uni = Universe.from_file(cfg["constituents_file"], cfg["drop_sentinels"])
    panel = load_panel(cfg, uni)
    sig = S.build(strategy, panel)
    mode = "none" if variant == "raw" else sig.default_filter_mode

    # deterministic (turnover-ranked) reference
    det = summarize(run_backtest(panel, uni, sig, cfg, mode), panel, cfg)

    rows = []
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        res = run_backtest(panel, uni, sig, cfg, mode, rng=rng)
        m = summarize(res, panel, cfg)
        rows.append({"seed": seed, "CAGR": m["CAGR"], "MaxDD": m["MaxDD"],
                     "MAR": m["MAR"], "final_equity": m["final_equity"]})
        if (seed + 1) % 10 == 0:
            print(f"  {seed+1}/{runs}", flush=True)
    mc = pd.DataFrame(rows)
    out = output_dir(market)
    mc.to_csv(os.path.join(out, f"montecarlo_{strategy}_{variant}.csv"), index=False)

    def pct(s):
        return f"P5 {np.percentile(s,5)*100:.1f}%  median {np.median(s)*100:.1f}%  P95 {np.percentile(s,95)*100:.1f}%"
    print(f"\n{S.DISPLAY_NAMES[strategy]} ({variant}) on {market.upper()} — {runs} Monte Carlo runs")
    print(f"  deterministic: CAGR {det['CAGR']*100:.1f}%  MaxDD {det['MaxDD']*100:.1f}%  MAR {det['MAR']:.2f}")
    print(f"  CAGR   {pct(mc['CAGR'])}")
    print(f"  MaxDD  {pct(mc['MaxDD'])}")
    print(f"  MAR    P5 {np.percentile(mc['MAR'],5):.2f}  median {np.median(mc['MAR']):.2f}  P95 {np.percentile(mc['MAR'],95):.2f}")

    # scatter: terminal return vs maxDD (book style)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(mc["MaxDD"].abs() * 100, (mc["final_equity"] / cfg["initial_capital"]),
               alpha=0.6, s=40, color="#1f77b4", label="MC runs (random selection)")
    ax.scatter(abs(det["MaxDD"]) * 100, det["final_equity"] / cfg["initial_capital"],
               marker="*", s=400, color="#d62728", label="deterministic (turnover-ranked)", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("Maximum Drawdown (%)")
    ax.set_ylabel("Terminal wealth multiple (log)")
    ax.set_title(f"{market.upper()}: {S.DISPLAY_NAMES[strategy]} ({variant}) — position-variability Monte Carlo ({runs} runs)")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    p = os.path.join(out, f"montecarlo_{strategy}_{variant}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "india"], required=True)
    ap.add_argument("--strategy", default="hundred_day_high")
    ap.add_argument("--variant", choices=["raw", "filtered"], default="filtered")
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()
    run_mc(args.market, args.strategy, args.variant, args.runs)


if __name__ == "__main__":
    main()
