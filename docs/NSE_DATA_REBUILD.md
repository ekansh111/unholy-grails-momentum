# Rebuilding a clean NSE adjusted-price series

The India results depend entirely on getting corporate-action adjustment right, and
the available NSE data made that hard. This documents the problem and the rebuild
(`tools/rebuild_nse_clean.py`).

## The problem
The vendor `adjClose` in the NSE panels is **catastrophically over-adjusted**:

| symbol | raw price | vendor adjClose | correct |
|---|--:|--:|--:|
| CUB | 10.5× | **12,667×** | 99–115× |
| SUPREMEIND | 12× | **4,830×** | 119× |
| GAIL | 1.7× | **480×** | 11× |

Root cause: the corporate-action factors come from a free-text NSE "subject" field that a
prior pipeline mis-parsed — **~54% of its bonus/split entries are spurious** (the file claims
a bonus on a day the price never moved; e.g. GODREJIND factor 0.0008). Applying those
fabricates phantom multibaggers that corrupt momentum signals and inflate the backtest.

A second, separate defect: a **systematic data break on 2024-11-04** where 22 unrelated
blue-chips (HDFCBANK, NESTLÉ, WIPRO, PIDILITE…) have their *entire pre-date history doubled*
versus the real price (verified against yfinance). A simultaneous −50% across unrelated names
is a bhavcopy basis-error, not a corporate action.

## The rebuild
The raw **bhavcopy close is ground truth** (actual traded prices, survivorship-safe). We throw
away the vendor `adjClose` and reconstruct it:

1. **Trim leading stub bars** (a few stray old bars before a multi-year gap) and repair
   isolated bad prints.
2. **Corporate-action calendar**, merged + cross-validated across two independent sources:
   - **NSE official** corporate-actions API (parsed with canonical factor math: bonus A:B →
     B/(A+B); split → newFV/oldFV).
   - **yfinance `.splits`** (covers older splits NSE's archive predates).
   They agree on the factor **96.8%** of the time (582 agree / 19 disagree); each catches some
   the other misses. Conflicts are resolved by the price.
3. **The raw price drop is the final arbiter.** A source CA is applied only where the price
   actually dropped (rejecting spurious entries) and the factor is taken from the real drop
   (handling combined bonus+split and parse errors — e.g. TITAN's 1:1 bonus + 10:1 split on
   one day resolves to ×0.0533 from the price). A clean drop **no source confirms** is left
   unadjusted unless it is beyond the circuit band (≥55%, physically a split).
4. **Systematic-break repair:** dates where ≥6 unrelated names share a persistent clean step
   are back-adjusted away (fixes the 2024-11-04 doubling).
5. **Anchor:** back-adjusted so the last bar = last raw close.

MoneyControl (consent-gated) and Chittorgarh (JS-rendered) could not be scraped from the build
environment; NSE-official + yfinance provided the two-source cross-validation instead.

## Result
527 names adjusted (339 cross-validated by both sources). Multiples now match reality —
PFC 10× (vendor said 40×), TITAN 715× (incl. its split), HDFCBANK 518× (continuous through
2024-11-04). Originals are never modified; output goes to a separate `pricesNseClean/` dir
with a per-symbol `_cleaning_audit.csv` (provenance for every action) and the two CA calendars.

**Why it matters:** rebuilding the data **reversed the India comparison**. The corruption had
inflated Clenow (its rank loaded up on the fake multibaggers) and injected fake crashes that
penalised the breakouts. On clean data the best filtered breakout (52-Wk High, MAR 0.64) beats
both Clenow variants (0.50 / 0.58).

## Residual tail (known limitation)
~88 names still show a >100% one-day residual: **mid-series multi-year gaps** (e.g. GOODYEAR's
18-year suspension — harmless, the engine won't trade across it) and **pre-2006 bonuses** the
NSE archive predates. All flagged (`review_flag`, `max_residual_1d_move_pct` in the audit). We
leave the ambiguous old ones unadjusted rather than guess — a clean −50% drop with no source
is indistinguishable from a real crash on close-only data. A reliable paid CA feed (EODHD
All-In-One, Capitaline/ACE) would close this.

## Regenerate
```bash
python tools/rebuild_nse_clean.py \
  --src .../pricesNse --out .../pricesNseClean \
  --nse .../pricesNseClean/_corporate_actions_nse.csv \
  --yf  .../pricesNseClean/_corporate_actions_yfinance.csv
```
The CA calendars are fetched from the NSE API and yfinance (persisted alongside the output).
