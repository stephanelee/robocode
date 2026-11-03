# Aegis 0.2.0 development screens (unregistered, exploratory)

Work-in-progress follow-up to the rejected [Aegis v1 pilot](../../docs/improvements/aegis-shield-v1.md), aimed at raising the block rate directly. **Not registered, no strength claim.** These are local container dev games against the pinned Diamond Port; `v2c` also used the dev-only instrumented copy, which is never used at runtime or in registered games. The 0.2.0 source on this branch is WIP and not frozen.

Changes from 0.1.0, each targeting a measured failure class:

1. **Corrected opponent self-prediction model.** The shooter predicts its next position by continuing its *previous* speed change under classic rules (`Aegis.classicSpeed(prev, current)`, using legal scans at N−3 and N−2). The 0.1.0 rule used the next scan's speed and missed. In `self-prediction-rule-validation.txt`, over 2,945 logged aims, origin errors above 2 px fall from 17.9% to 0% (max 3.33 → 1.95 px). In v1 development data, ±3 px line errors had a 58% hit rate.
2. **Behind-side adjacent coverage.** When the shooter misses its centre candidate, it almost always picks the candidate one angle behind our motion at aim time (≈ range·tan(asin(8/v)/29)). The solver now adds expected-block weight on that side only.
3. **Replay label lead** of 2 px in our direction of motion at wave launch. Off-centre picks rose to 26% of aims after recent negative labels, against 5% when labels were balanced.
4. Shots stay targetable down to their last feasible fire turn (`contact ≥ t + 2`).

| Screen | Changes | Rounds won | Share | Blocked / inferred |
| --- | --- | --- | --- | --- |
| v1 dev reference (`aegis28`) | — | 0/10 | — | 74.1% |
| `v2a` | 1, 2 (both sides) | **2/10** | 7.75% | 79.2% |
| `v2b` | + aggressive reload chaining (0.3, reverted) | 0/10 | 0.19% | 72.6% |
| `v2c` | 1, 2 (behind side), 3, 4 | **2/10** | 8.82% | 78.5% |

Remaining failures in `v2c`:
- Off-centre aims ≈ 10% of shots. Per-round instrumented logs show 95–98% centre aims under the opponent's TripHammer gun but 68–89% under its Anti-Surfer gun, which it selects by virtual-gun rating.
- Low-coverage shields ≈ 3.5%.
- Unshielded dodge cases ≈ 3%.

Rounds 1–2 were won in both v2a and v2c; later rounds were lost.
