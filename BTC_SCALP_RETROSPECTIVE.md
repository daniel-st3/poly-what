# BTC Scalp Strategy — Retrospective

**Status: RETIRED**
Retired: 2026-03-29
Offline evaluation verdict: RETIRE (no profitable regime found)

---

## Overview

`btc_scalp` was a short-term intraday scalping strategy that traded binary
Polymarket BTC-updown markets with 0.5–6.5 minutes to expiry. It used a
log-normal price model driven by a Coinbase WebSocket price feed, momentum
signal, and realized volatility to estimate `p_up` and compute fee-aware
expected value before paper-trading a $2 per-contract position.

The strategy ran live (paper-trading mode) from approximately 2026-03-26 to
2026-03-29 and was retired after offline edge evaluation found no measurable
profitable regime across 332 closed trades.

---

## Original Hypothesis

Binary BTC-updown markets on Polymarket reprice slowly relative to the real-time
BTC spot price. A fast log-normal model with momentum adjustment could identify
mispriced contracts just before expiry and capture a positive EV edge by buying
the correct directional side slightly before resolution.

---

## What Was Tested

- Log-normal `p_up` model anchored to a `start_price_proxy` with 60s momentum
  adjustment and realized-vol z-score normalization.
- Fee-aware EV filter: only enter if `EV > btc_scalp_min_ev_per_contract`.
- Multiple guards added iteratively: spread cap, CLOB availability check,
  minimum drift floor (`btc_scalp_min_signal_drift`), and `p_up` floor
  (`btc_scalp_min_p_up_for_down`) to skip near-coin-flip Down entries.
- Paper-traded on Railway (continuous deployment, live Coinbase feed).
- Down-side entries only (`btc_scalp_disable_up_entries=true` was set early
  and never removed — Up-side was never meaningfully tested).

---

## Key Debugging Milestones

| Milestone | Outcome |
|-----------|---------|
| Export bottleneck fixed | CSV export was writing duplicate scan rows per trade; fixed to emit one entry-time row per trade_id |
| Entry-row audit | Earlier analysis misread scan-time rows as trade-entry rows, inflating apparent sample size; corrected |
| Mapping inversion investigation | Suspected Up/Down label swap in Polymarket data; could not be proven — no reliable inversion in the final losing regime |
| Formula reproduction | `p_up` model formula verified to match exactly against hand-computed reference values |
| `realized_vol` observation | `realized_vol` sat at the floor (0.001) for the overwhelming majority of scans, meaning the z-score was driven almost entirely by raw drift and momentum, not by true volatility normalization |
| Drift-floor guard | Added `btc_scalp_min_signal_drift` to skip entries when observed BTC drift was too small to justify a directional bet |
| `p_up` floor guard for Down | Added `btc_scalp_min_p_up_for_down`: skip Down entry if model `p_up` is so low it implies the Up side is actually cheap |
| Post-guard behavior | After both guards were active, the bot mostly stopped trading (`trade_candidates=0` in most scan cycles); no live edge ever materialized |

---

## Final Evaluation Results

Offline evaluation run on 2026-03-29 using `scripts/btc_scalp_edge_eval.py`
across 332 deduplicated closed trades (2026-03-26 to 2026-03-27).

**Overall**

| Metric | Value |
|--------|-------|
| Trades | 332 |
| Side | Down only (Up entries disabled throughout) |
| Win rate | 4.8% |
| Avg PnL | -0.8533 |
| Total PnL | -283.30 |

**No bucket in any dimension showed win rate > 55% and avg PnL > 0.01:**

| Dimension | Best bucket | Win % | Avg PnL |
|-----------|-------------|-------|---------|
| BTC drift from open | -0.05 to +0.05% | 5.6% | -0.86 |
| p_up model | 0.45–0.55 | 6.6% | -0.82 |
| minutes_left | 4m+ | 5.1% | -0.88 |
| entry price | 0.40–0.50 | 7.6% | -0.79 |

The p_up model ranged from 0.35–0.55 across all trades, meaning the model
never expressed strong directional conviction; all entries were near-coin-flip
bets dressed with a thin EV justification.

---

## Why the Strategy Is Retired

1. **No regime with positive expectation.** Across drift, p_up, time-to-expiry,
   and entry price — no bucket produced positive average PnL with sample size
   ≥ 5 trades.

2. **Only one side tested.** Up entries were disabled for the entire live period.
   The negative result applies specifically to the Down-only configuration.
   A symmetric test was never run.

3. **Model signal was too weak.** `realized_vol` at the floor meant the model
   was essentially computing drift / floor, producing inflated z-scores and
   overconfident `p_up` estimates that did not reflect true market uncertainty.

4. **Guards caused self-suppression.** Each guard added to limit bad entries
   also reduced the trade count. After both drift and `p_up` floor guards were
   active, the strategy produced near-zero trade candidates per session.
   A strategy that cannot trade cannot recover.

5. **Market structure not in our favor.** Polymarket BTC-updown markets in this
   period resolved Up the vast majority of the time while the bot consistently
   bet Down, suggesting systematic model bias rather than random noise.

---

## Lessons Learned

- **Validate the signal before building guards.** The first live trades should
  have been a forward-validation batch with minimal filters, not an iterative
  debugging session. Adding guards to a signal that has no edge just limits
  losses — it does not create edge.

- **`realized_vol` at the floor is a model smell.** When your vol normalizer
  is pinned at a constant, your model is no longer a volatility model — it is
  a drift-scaled heuristic. Detect this before going live.

- **Paper-trading on Railway with a live price feed is useful infrastructure.**
  The export pipeline, scan logs, and offline evaluation framework are all
  reusable for future strategies.

- **One-sided testing is not testing.** Disabling Up entries "for safety"
  before validating the signal meant the strategy operated with half its
  degrees of freedom from day one. Validate both sides on a neutral sample
  before disabling either.

- **Document negative results.** 332 trades across a small time window is a
  small but real negative result. It narrows the hypothesis space for the
  next attempt.

---

## Recommended Next Direction

If revisiting BTC intraday strategies on Polymarket:

1. **Establish a baseline resolution rate.** Before any model, measure how
   often BTC-updown markets resolve Up vs. Down by time-of-day and drift
   magnitude. If the market is structurally asymmetric (e.g., 70% Up during
   certain sessions), a directional prior is more valuable than a log-normal
   model.

2. **Use real volatility.** Replace the realized-vol floor heuristic with a
   rolling ATR or 5-minute realized vol from the Coinbase feed before computing
   z-scores.

3. **Test Up-side entries in isolation first.** Given the Down-only results,
   Up entries on the same signal family might outperform — but this requires
   a clean forward-validation run, not inference from a Down-only sample.

4. **Consider a different market structure entirely.** Event-driven markets
   (macro announcements, on-chain data releases) may offer more durable edge
   than mechanical price-vs-open comparisons with 5-minute windows.

---

## Manual Disable Note

`btc_scalp` can be disabled by setting `ENABLE_BTC_SCALP=false` in `.env`.
This flag is checked at the top of `watchdog run-btc-scalp` in `cli.py`
and is now set to `false` by default in `Settings`.

The Railway deployment (`railway.json`) still points to `run-btc-scalp` as
its start command. To fully stop Railway from running the strategy, either
update `railway.json` to point to a different command or shut down the
Railway service directly.

---

> **Scope note:** This retirement applies to the current `btc_scalp` signal
> family and implementation. It does not prove all BTC intraday market-making
> or event trading ideas are invalid.
