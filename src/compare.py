"""Build the cross-strategy + Clenow + Buy&Hold comparison: tables, charts, report.

Reads the per-market summary.csv (Unholy Grails), the Clenow baseline equity
CSVs (book-uncapped and 20-position-capped), and the benchmark equity CSVs
(report index, and the full-history regime index); scores every external curve
with the SAME metrics function used for the strategies; then writes
comparison.csv, the book's signature CAGR-vs-MaxDD scatter, an equity overlay, a
MAR bar chart, and a markdown RESULTS report.

Fairness notes baked into the table: per-row scored window (the India report
benchmark only starts 2005), average positions held (Clenow book is uncapped),
and an explicit 20-capped Clenow row for a like-for-like vs the 20-name UG books.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfg import REPO_ROOT, load_config, output_dir
from metrics import equity_metrics
from strategies import DISPLAY_NAMES

MARKETS = {"us": "config/us.yaml", "india": "config/india.yaml"}
CLENOW_OUT = os.path.join(REPO_ROOT, "outputs", "clenow")
ABBR = {
    "52-Week High": "52WkHi", "100-Day High": "100DHi", "TrendPilot": "TrPilot",
    "Golden Cross": "GoldX", "Moving Avg Channel": "MAC", "TechTrader": "TechT",
    "20% Flipper": "Flip20", "Bollinger Breakout": "BBO",
}


def _score_equity_csv(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    eq = df["equity"]
    dd = df["drawdown"] if "drawdown" in df else None
    m = equity_metrics(eq, dd)
    m["exposure"] = float(df["exposure"].mean()) if "exposure" in df else np.nan
    m["avg_positions"] = float(df["n_positions"].mean()) if "n_positions" in df else np.nan
    m["final_equity"] = float(eq.iloc[-1])
    m["start"] = eq.index[0].date()
    m["end"] = eq.index[-1].date()
    return m


def _clenow_row(market: str, tag: str, label: str, kind: str) -> dict | None:
    suffix = f"_{tag}" if tag else ""
    row = _score_equity_csv(os.path.join(CLENOW_OUT, f"{market}{suffix}_equity.csv"))
    if not row:
        return None
    meta_path = os.path.join(CLENOW_OUT, f"{market}{suffix}_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    row.update({
        "label": label, "display": "Clenow", "variant": tag or "book", "kind": kind,
        "strategy": "clenow", "win_rate": meta.get("win_rate"),
        "n_trades": meta.get("closed_episodes"),
        "payoff": np.nan, "expectancy": np.nan, "avg_hold_days": np.nan,
        "total_commission": meta.get("total_commission"),
    })
    return row


def load_comparison(market: str) -> pd.DataFrame:
    mkt_out = output_dir(market)
    cfg = load_config(MARKETS[market])
    ug = pd.read_csv(os.path.join(mkt_out, "summary.csv"))
    ug["label"] = ug["display"] + " (" + ug["variant"] + ")"
    ug["kind"] = "UG-" + ug["variant"]
    rows = ug.to_dict("records")

    cb = _clenow_row(market, "", "Clenow — book (uncapped)", "Clenow")
    if cb:
        rows.append(cb)
    cc = _clenow_row(market, "cap20", "Clenow — 20-position cap", "Clenow-cap20")
    if cc:
        rows.append(cc)

    # Buy & Hold — report index (TR where available).
    bh = _score_equity_csv(os.path.join(mkt_out, "benchmark_equity.csv"))
    if bh:
        bh.update({
            "label": f"Buy & Hold — {cfg['benchmark_name']}", "display": "Buy & Hold",
            "variant": "benchmark", "kind": "BuyHold", "strategy": "buyhold",
            "win_rate": np.nan, "n_trades": 0, "payoff": np.nan,
            "expectancy": np.nan, "avg_hold_days": np.nan, "avg_positions": np.nan,
        })
        rows.append(bh)
    # Buy & Hold — regime index (full history; only added when it starts meaningfully
    # earlier than the report benchmark, i.e. India Sensex 1998 vs Nifty500 2005).
    rb = _score_equity_csv(os.path.join(mkt_out, "regime_benchmark_equity.csv"))
    if rb and bh and (bh["start"] - rb["start"]).days > 180:
        rb.update({
            "label": f"Buy & Hold — {cfg['regime_name']} (full window, price-only)",
            "display": "Buy & Hold", "variant": "benchmark-full", "kind": "BuyHold-full",
            "strategy": "buyhold", "win_rate": np.nan, "n_trades": 0, "payoff": np.nan,
            "expectancy": np.nan, "avg_hold_days": np.nan, "avg_positions": np.nan,
        })
        rows.append(rb)

    df = pd.DataFrame(rows)
    df["market"] = market
    df.to_csv(os.path.join(mkt_out, "comparison.csv"), index=False)
    return df


# ----------------------------------------------------------------- charts
def scatter_chart(df: pd.DataFrame, market: str):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    styles = {
        "UG-raw": dict(marker="o", facecolor="none", edgecolor="#1f77b4", s=90, label="UG raw"),
        "UG-filtered": dict(marker="o", color="#1f77b4", s=90, label="UG + index filter"),
        "Clenow": dict(marker="*", color="#d62728", s=440, label="Clenow (uncapped)", zorder=5),
        "Clenow-cap20": dict(marker="P", color="#d62728", s=200, label="Clenow (20-cap)", zorder=5),
        "BuyHold": dict(marker="s", color="#2ca02c", s=150, label="Buy & Hold", zorder=5),
        "BuyHold-full": dict(marker="s", facecolor="none", edgecolor="#2ca02c", s=150, label="Buy & Hold (full-window)", zorder=5),
    }
    seen = set()
    for _, r in df.iterrows():
        st = dict(styles[r["kind"]])
        lab = st.pop("label")
        lab = None if lab in seen else (seen.add(lab) or lab)
        ax.scatter(abs(r["MaxDD"]) * 100, r["CAGR"] * 100, label=lab, **st)
        if r["kind"].startswith("UG"):
            ax.annotate(ABBR.get(r["display"], r["display"]),
                        (abs(r["MaxDD"]) * 100, r["CAGR"] * 100),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points",
                        color="#1f77b4" if r["variant"] == "filtered" else "#7f7f7f")
    xmax = df["MaxDD"].abs().max() * 100 * 1.05
    for mar in (0.25, 0.5, 1.0):
        x = np.linspace(1, xmax, 50)
        ax.plot(x, mar * x, "--", color="#cccccc", lw=0.8, zorder=0)
        ax.annotate(f"MAR {mar}", (x[-1], mar * x[-1]), fontsize=7, color="#999999")
    ax.set_xlabel("Maximum Drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(f"{market.upper()}: Return vs Risk — Unholy Grails vs Clenow vs Buy & Hold\n"
                 f"(up-and-to-the-left is better; dashed = constant MAR)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(output_dir(market), "scatter.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def equity_overlay(df: pd.DataFrame, market: str):
    mkt_out = output_dir(market)
    ugf = df[df["kind"] == "UG-filtered"].sort_values("MAR", ascending=False).head(4)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for _, r in ugf.iterrows():
        path = os.path.join(mkt_out, "equity", f"{r['strategy']}_filtered.csv")
        if os.path.exists(path):
            eq = pd.read_csv(path, index_col=0, parse_dates=True)["equity"]
            ax.plot(eq.index, eq.values, lw=1.3, label=f"{r['display']} (filtered)")
    cl = os.path.join(CLENOW_OUT, f"{market}_equity.csv")
    if os.path.exists(cl):
        eq = pd.read_csv(cl, index_col=0, parse_dates=True)["equity"]
        ax.plot(eq.index, eq.values, lw=2.2, color="#d62728", label="Clenow (uncapped)")
    for fname, color, ls, lab in [("benchmark_equity.csv", "#2ca02c", "--", "Buy & Hold")]:
        bh = os.path.join(mkt_out, fname)
        if os.path.exists(bh):
            eq = pd.read_csv(bh, index_col=0, parse_dates=True)["equity"]
            ax.plot(eq.index, eq.values, lw=2.0, color=color, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log scale)")
    ax.set_title(f"{market.upper()}: Equity — top index-filtered UG strategies vs Clenow vs Buy & Hold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    p = os.path.join(mkt_out, "equity.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def mar_bars(df: pd.DataFrame, market: str):
    ug = df[df["kind"].str.startswith("UG")].copy()
    piv = ug.pivot_table(index="display", columns="variant", values="MAR")
    order = piv.mean(axis=1).sort_values(ascending=False).index
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv.get("raw", pd.Series(index=piv.index)).values, 0.4, label="raw", color="#9ecae1")
    ax.bar(x + 0.2, piv.get("filtered", pd.Series(index=piv.index)).values, 0.4, label="+ index filter", color="#1f77b4")
    cl = df[df["kind"] == "Clenow"]
    if len(cl):
        ax.axhline(cl.iloc[0]["MAR"], color="#d62728", ls="--", lw=1.5, label=f"Clenow uncapped ({cl.iloc[0]['MAR']:.2f})")
    cc = df[df["kind"] == "Clenow-cap20"]
    if len(cc):
        ax.axhline(cc.iloc[0]["MAR"], color="#d62728", ls=":", lw=1.5, label=f"Clenow 20-cap ({cc.iloc[0]['MAR']:.2f})")
    bh = df[df["kind"] == "BuyHold"]
    if len(bh):
        ax.axhline(bh.iloc[0]["MAR"], color="#2ca02c", ls=":", lw=1.5, label=f"Buy & Hold ({bh.iloc[0]['MAR']:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAR (CAGR / |MaxDD|)")
    ax.set_title(f"{market.upper()}: Risk-adjusted return (MAR) by strategy — raw vs index-filtered")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = os.path.join(output_dir(market), "mar_bars.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ----------------------------------------------------------------- report
def _pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) and np.isfinite(x) else "—"


def _num(x, d=2):
    return f"{x:.{d}f}" if pd.notna(x) and np.isfinite(x) else "—"


def md_table(df: pd.DataFrame) -> str:
    head = ("| Strategy | From | CAGR | MaxDD | MAR | Sharpe | Exposure | AvgPos | Win% | Payoff | Trades |\n"
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    lines = []
    for _, r in df.iterrows():
        frm = pd.Timestamp(r["start"]).year if pd.notna(r.get("start")) else "—"
        ap = _num(r.get("avg_positions"), 1)
        tr = int(r["n_trades"]) if pd.notna(r["n_trades"]) else "—"
        lines.append(f"| {r['label']} | {frm} | {_pct(r['CAGR'])} | {_pct(r['MaxDD'])} | {_num(r['MAR'])} | "
                     f"{_num(r['Sharpe'])} | {_pct(r['exposure'])} | {ap} | {_pct(r['win_rate'])} | "
                     f"{_num(r['payoff'])} | {tr} |")
    return head + "\n".join(lines)


DISCLOSURE = """\
> **What is held constant vs not.** The comparison fixes the *data* (same
> survivorship-aware adjusted-OHLC panels and causal-repair), *costs*, *point-in-time
> universe* and *window* for every system. It does **not** equalise portfolio
> construction: Clenow's book profile is **uncapped** (it buys down the ranking until
> cash runs out — avg ~22 names US / ~31 India), ATR-volatility-sized, resized
> bi-weekly and fully compounding, whereas the Unholy Grails systems hold **≤20
> equal-weight single lots, no resizing** (the book's rules). The **Clenow — 20-position
> cap** row isolates signal quality from that diversification advantage. Unholy Grails
> also applies the book's liquidity floor (Clenow's book profile does not). Delisted
> names are liquidated at last traded price for **both** systems (the book's and Clenow's
> shared convention — an optimistic, no-haircut assumption).
"""


def build():
    sections = []
    for market in ["us", "india"]:
        df = load_comparison(market)
        cfg = load_config(MARKETS[market])
        scatter_chart(df, market)
        equity_overlay(df, market)
        mar_bars(df, market)
        ranked = df.sort_values("MAR", ascending=False)
        ug = df[df["kind"].str.startswith("UG")]
        span = f"{ug['start'].min()} → {ug['end'].max()}"
        sections.append(
            f"## {market.upper()} — {cfg['benchmark_name']} universe, strategies {span}\n\n"
            f"Sorted by MAR (risk-adjusted return = CAGR / |MaxDD|). 'From' = the year each\n"
            f"row's scored series begins (note the India report benchmark only starts 2005).\n\n"
            + md_table(ranked) + "\n\n" + DISCLOSURE
            + f"\n\n![scatter](outputs/{market}/scatter.png)\n"
            + f"![equity](outputs/{market}/equity.png)\n"
            + f"![mar](outputs/{market}/mar_bars.png)\n")
    report = ("# Results: Unholy Grails vs Clenow\n\n"
              "Auto-generated by `src/compare.py`. See [README.md](README.md) for the narrative.\n\n"
              + "\n\n".join(sections))
    with open(os.path.join(REPO_ROOT, "RESULTS.md"), "w") as fh:
        fh.write(report)
    print("wrote RESULTS.md + per-market comparison.csv and charts")


if __name__ == "__main__":
    build()
