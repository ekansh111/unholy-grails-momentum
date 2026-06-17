# Unholy Grails — Strategy Specification

Faithful encoding of the eight momentum strategies in Nick Radge's *Unholy Grails*
(2012), plus the Index Filter overlay and the portfolio-construction rules. Each
strategy is a daily, event-driven **breakout / trail** system: a signal forms on a
*close*, and the order fills at the **next day's open** (the book's VWAP-proxy
convention). Positions are equal-weighted; the portfolio holds up to **20** names.

> Convention shared by all systems: "buy" = enter next open after a long signal
> closes; "sell"/"exit" = exit next open after an exit signal closes. No shorting
> (the book is long/cash only — short-selling was banned on the ASX in 2008, so the
> book treats "revert to cash" as the defensive action).

---

## The eight strategies

### 1. New Yearly Highs (52-week / 250-day High)
- **Entry:** close at a new **250-day high** → buy next open.
- **Exit:** close below the **250-day low** → sell next open.
- Wide exit (250-day low can sit far below price) → slow to defend in 2008; this is
  the book's motivation for the Index Filter.

### 2. 100-Day High
- **Entry:** close above the highest high of the last **100 days** → buy next open.
- **Exit:** close below the lowest low of the last **100 days** → sell next open.
- More aggressive entry / earlier exit than #1 → faster drawdown recovery.

### 3. TrendPilot
- Per-instrument **200-day SMA** with a 5-day confirmation to kill whipsaw.
- **Entry:** close **above** the 200-day SMA for **5 consecutive days** → buy next open.
- **Exit:** close **below** the 200-day SMA for **5 consecutive days** → sell next open.
- (Book's primary form trades the index itself; we run the 20-stock portfolio variant
  — each stock against its own 200-day SMA — for comparability with the others.)

### 4. Golden Cross
- **Entry:** **50-day SMA** crosses up through the **200-day SMA** → buy next open.
- **Exit (Death Cross):** 50-day SMA crosses down below the 200-day SMA → sell next open.

### 5. Moving Average Channel (MAC)
- Channel: **upper = 10-day SMA of highs**, **lower = 8-day SMA of lows** (Bernstein).
- **Entry:** **5+ consecutive bars completely above** the channel top (bar **low** >
  upper) → buy next open. A partial bar resets the count.
- **Exit:** **5+ consecutive bars completely below** the channel bottom (bar **high** <
  lower) → sell next open.
- Optional fail-safe: exit on an *n*% fall from the high (book recommends ≥25% if used;
  default OFF — tighter stops "destroy the strategy").

### 6. TechTrader (objective rules only; John Rowland)
1. Close above the **40-day SMA**.
2. Price below a price ceiling *(ASX-specific "<$10" — see ADAPTATION below)*.
3. Close > open on the trigger day.
4. **21-day average turnover** above a liquidity floor (ASX: $500k).
5. Trigger day is the **highest value over the last 70 days** (70-day high).
6. Trigger day also crosses the **highest high of the last 10 days**.
7. Initial protective stop **10% below entry**.
8. Trailing stop: **180-day EMA of the lows**.
- **Exit:** initial 10% stop hit, OR close below the 180-day EMA of lows.
- Subjective rules (obvious uptrend / not range-bound) are **not** coded (per the book).
- **ADAPTATION:** rules 2 & 4 are ASX/AUD artifacts. On a large-cap universe (S&P 500 /
  Nifty 500) almost nothing trades below $10, so the price ceiling is **configurable and
  default-disabled**; the turnover floor is scaled to local currency. Documented as a
  deviation in the comparison.

### 7. 20% Flipper (purest momentum)
- Track the running **low**. A close **20% above** the running low → buy next open; on
  entry place a protective stop **20% below the entry day's high**.
- Track the running **high**. A **20% fall from any high** → sell next open.

### 8. Bollinger Band Breakout (BBO)
- Central = **100-day SMA**. Upper band = **+3 SD**, lower band = **−1 SD** (book's
  loosened settings for longer-term momentum).
- **Entry:** close **above the upper (3 SD)** band → buy next open.
- **Exit:** close **below the lower (1 SD)** band → sell next open.

---

## Index Filter overlay (75-day SMA on the index)

A **75-day SMA** on the underlying index defines the regime.
- **Index > 75-day SMA (uptrend):** take new entries normally.
- **Index < 75-day SMA (downtrend):** defensive. The book uses one of three actions
  depending on the strategy — implemented as a configurable mode:
  - `no_new_entries` — stop taking new signals; manage existing per the raw exit (BBO).
  - `trailing_10pct` — additionally exit any held name that falls 10% from its high (or
    any subsequent higher high), instead of waiting for the wide raw exit (52-wk High,
    100-day High filtered).
  - `exit_all_cash` — sell everything to cash next open (Golden Cross filtered; "100-Day
    High Cash"; 20% Flipper filtered also tightens stops to 10% + no new entries).

Each strategy's filtered variant uses the defensive mode the book applied to it; modes
are configurable so all combinations are testable.

---

## Portfolio construction & position variability
- **20 equal-weight positions**, max **5%** of portfolio each; same position count
  regardless of capital (book's rule — keeps sample/position-variability bias constant).
- **More signals than free slots:** the book accepts "position variability" and studies
  it with Monte Carlo. For the deterministic single run we rank same-day candidates by
  **liquidity (21-day average dollar/₹ turnover, descending)** — the most tradable first,
  matching the interviewees' ranking habit. We ALSO run a **Monte Carlo** (random pick
  among eligible signals, N runs) to report the distribution, exactly as the book does.
- Universe is **point-in-time index membership** (survivorship-controlled) + a liquidity
  floor; delisted names are liquidated at last traded price (postdictive exit), as the
  book prescribes.

## Costs, fills, data (must match the Clenow baseline)
- **Fills:** next day's **open** (VWAP proxy). Sells settle before buys; cash never goes
  negative (buys trimmed to available cash).
- **Costs:** identical model to the Clenow baseline run on the same data — flat
  commission + slippage bps per side (US and India values taken from the Clenow configs),
  so the comparison is apples-to-apples. (The book's own convention is 0.25% or min
  ~$30/trade; we report that as a sensitivity but headline on the matched-cost run.)
- **Data:** the same survivorship-aware adjusted-OHLC panels and point-in-time
  constituent lists the Clenow project uses (US: ~1,188 symbols incl. delisted; NSE:
  ~1,389), so both systems see the identical universe and prices.

## Implementation notes & deliberate deviations
- **Donchian on highs/lows.** The 52-Week and 100-Day systems trigger when the *close*
  breaks the prior window's highest **high** / lowest **low** (Donchian, shifted one bar
  so there's no same-bar lookahead). This follows the book's explicit 100-Day wording —
  "a close above the highest **high point** of those 100 days … the lowest **low** of
  those 100 days" — and is applied to the 52-Week system for consistency.
- **20% Flipper swings** track the intraday **high**/**low** (so the protective stop sits
  "20% below the entry day's high" as the book specifies), while the buy/sell *trigger* is
  a **close** crossing the ±20% level off the swing extreme.
- **Delisting exit.** A held name with no bar for >5 trading days is liquidated at its
  **last traded (adjusted) price** with no bankruptcy haircut — the book's "exit at the
  last traded price" convention, which the Clenow baseline also uses, so the comparison is
  symmetric. This is an optimistic (upper-bound) recovery assumption in a survivorship-free
  backtest; both systems share it.
- **Causal adjustment repair.** NSE vendor data carries occasional corrupt single-day
  adjusted-close spikes (e.g. adjClose +300% on a day the raw close fell 20%). The loader
  repairs only days where the adjusted return is wild **and** the raw return is tame
  **and** the implied adjustment factor itself jumped — so genuine splits, dividends and
  real large moves are left untouched. See `src/data.py:_repair_adjusted_close`.

## Metrics (match the book + the Clenow reports)
CAGR, Max Drawdown, **MAR (CAGR/MaxDD)**, annualised return stdev, Sharpe (book's caveat
noted), **market exposure / time-in-market**, **win rate**, **win/loss (payoff) ratio**,
number of round-turn trades, expectancy, and the equity / underwater curves. Benchmark =
Buy & Hold of the universe's own index (total-return where available).
