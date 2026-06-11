# Model robustness: does 150 years of data hurt?

A reader challenged the Oracle: "most of that data is irrelevant, use recent matches only." Tested empirically: the Elo engine was rebuilt on three training windows (since 1872, since 2010, since 2018) and all 10,000 simulations re-run on each (same seed).

| Team | Since 1872 | Since 2010 | Since 2018 |
|---|---|---|---|
| Spain | 28.4% | 24.1% | 24.8% |
| Argentina | 20.3% | 18.6% | 11.2% |
| France | 10.4% | 8.7% | 7.9% |
| Mexico | 3.4% | 6.0% | 7.1% |
| Morocco | 1.3% | 3.6% | 7.7% |
| Japan | 1.3% | 2.8% | 4.7% |

**Findings**

1. **1872 vs 2010: nearly identical.** Elo updates sequentially, so recent results have already overwritten the past; the old matches were effectively invisible. Long history is harmless and buys rating convergence.
2. **2018-only breaks via cold start, not relevance.** All teams start at the same baseline; elite sides sit hundreds of points above it, and eight years is not enough runway to climb back. Argentina fall from 20.3% to 11.2%, Brazil halve. Truncation makes the model forgetful, not current.
3. **Pre-registered prediction was wrong, kept in deliberately:** the expectation was that minnows would wobble and giants would hold. The opposite happened (Curaçao and Haiti stable in all windows; the elite moved).
4. **The genuine signal:** Morocco (1.3% to 7.7%) and Japan (1.3% to 4.7%) reflect real post-2022 form. The right upgrade is time-decay weighting inside the full record, not deleting the record.

Reproduce: `make_site.py` contains the Elo and simulator; pass an earlier start date to `build_elo` to replicate any window.
