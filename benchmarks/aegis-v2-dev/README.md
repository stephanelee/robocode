# Aegis 0.2.0 development screens (unregistered, exploratory)

Work-in-progress follow-up to the rejected [Aegis v1 pilot](../../docs/improvements/aegis-shield-v1.md), aimed at raising the block rate directly. **Not registered, no strength claim.** These are local container dev games against the pinned Diamond Port; `v2c` also used the dev-only instrumented copy, which is never used at runtime or in registered games. The 0.2.0 changes are preserved as `aegis-0.2.0-wip.patch` against the registered 0.1.0 source (`git apply` from the repository root). They are not frozen and have no rebuilt JAR. The tracked `experiments/Aegis/` source and JAR remain the registered 0.1.0 bytes. The patch also changes `classicSpeed`'s semantics, so the 0.1.0 unit test of the old rule must be updated before any freeze.

Changes from 0.1.0, each targeting a measured failure class:

1. **Corrected opponent self-prediction model.** The shooter predicts its next position by continuing its *previous* speed change under classic rules (`Aegis.classicSpeed(prev, current)`, using legal scans at N−3 and N−2). The 0.1.0 rule used the next scan's speed and missed. In `self-prediction-rule-validation.txt`, over 2,945 logged aims, origin errors above 2 px fall from 17.9% to 0% (max 3.33 → 1.95 px). In v1 development data, ±3 px line errors had a 58% hit rate.
2. **Behind-side adjacent coverage.** When the shooter misses its centre candidate, it almost always picks the candidate one angle behind our motion at aim time (≈ range·tan(asin(8/v)/29)). The solver now adds expected-block weight on that side only.
3. **Replay label lead** of 2 px in our direction of motion at wave launch. Off-centre picks rose to 26% of aims after recent negative labels, against 5% when labels were balanced.
4. Shots stay targetable down to their last feasible fire turn (`contact ≥ t + 2`).

| Screen | Changes | Rounds won | Share | Blocked / inferred shots (`bulletsBlocked / enemyShotsDetected`) |
| --- | --- | --- | --- | --- |
| v1 dev reference (`aegis28`) | — | 0/10 | — | 74.1% (395/533 classified) |
| `v2a` | 1, 2 (both sides) | **2/10** | 7.75% | 79.20% (956/1,207) |
| `v2b` | + aggressive reload chaining (0.3, reverted) | 0/10 | 0.19% | 72.97% (926/1,269) |
| `v2c` | 1, 2 (behind side), 3, 4 | **2/10** | 8.82% | 78.69% (1,396/1,774) |
| `vg1` (6 rounds, rating instrumentation) | same as v2c | 0/6 | 0.32% | 72.23% (645/893) |
| `vg2` | + exact zero-label cost on real-shot waves (reverted) | 0/10 | 0.51% | 75.39% (1,529/2,028) |
| `vg3` | v2c + 5: adaptive aim-class tracker (**current WIP patch**) | **2/10** | 9.19% | **80.33%** (1,850/2,303) |
| `eg1` (8 rounds, energy logging) | vg3 | 2/8 | 12.44% | 78.97% (1,664/2,107) |
| `vg4` | vg3 + narrow empirical centre weights, three-line coverage (not kept) | 1/10 | 4.48% | 78.61% (1,650/2,099) |
| `vg5` | vg4 + acceptance by score / centre coverage (not kept) | 1/10 | 4.90% | 76.29% (1,960/2,569) |

5. **Adaptive aim-class tracker (gun-switching work).** The shooter picks TripHammer or Anti-Surfer by a battle-long virtual-bullet rating computed only on real-shot waves. In `vg1/virtual-gun-ratings.txt`, our real-shot break centres lean behind our motion (mean −0.0028 rad). Anti-Surfer won more of the disagreements, overtook TripHammer and was selected; under it, centre-candidate aims fall to 68–89%. A direct exact zero-label cost on real-shot waves (`vg2`) did not remove the bias and hurt geometry, so it was reverted. Off-centre aims are persistent: P(next off-centre) is 6% after 0 of the last 8 resolved shots and 32% after 3+. Without trying to prevent them, the tracker turns the last 8 resolved shot classes (centre / behind / ahead, from the actual bullet lines in legal hit/collision events) into adaptive mixture weights for the solver. In `vg3`, off-centre hits fell from about 8% to 5.4% of classified outcomes, and the centre-aiming gun won 351 of 602 rating disagreements.

Denominator note: block rates above use all inferred shots from `metrics.jsonl`. The retained `*/outcome-classes.txt` files (unchanged classifier outputs) use a different population, classified outcomes, which also includes unmatched hits. Their shares are therefore slightly lower and are not used in the table. The per-power block shares further below are over *resolved* (matched blocked or hit) shots and are labelled as such.

Remaining failures in `v2c` (before change 5):
- Off-centre aims ≈ 10% of shots. Per-round instrumented logs show 95–98% centre aims under the opponent's TripHammer gun but 68–89% under its Anti-Surfer gun, which it selects by virtual-gun rating.
- Low-coverage shields ≈ 3.5%.
- Unshielded dodge cases ≈ 3%.

Rounds 1–2 were won in both v2a and v2c; later rounds were lost.

**Endgame finding (`eg1/energy-trajectories.txt`, `block-rate-by-enemy-power.txt`).**
- Among resolved (matched blocked or hit) full-power (≥1.5) shots, the tracker variant blocks **91.8%**, which is above the ≈88% break-even. Against the shooter's low-energy, cubically powered-down shots it blocks only 80–83%, and those are 62% of all resolved shots (2,623 of 4,200 were below 0.2 power).
- Rounds the shooter never powers down in are won (round 1 of `eg1`: shooter 100→0 by turn 704).
- In lost long rounds Aegis typically *enters* the low-power phase far ahead (for example 77 vs 26 or 82 vs 37 energy) and bleeds that lead away over thousands of turns. At 0.1 power each block costs both sides 0.1, while each leak costs Aegis 0.4 and refunds the shooter 0.3.
- With the corrected model, real lines are 87.7% within 1 px of centre and 12% at exactly one adjacent-candidate spacing.
- Narrowing the solver's weights to that distribution (`vg4`/`vg5`) removed its robustness margin and was worse, so it was not kept.
- The power-down applies only beyond 325 px. The most promising next lever is therefore range control that keeps the shooter firing full power, where blocking already exceeds break-even, or a dedicated low-power-phase plan. Neither is implemented or registered.

Long rounds (5,000–7,000 turns) are still lost in the shooter's low-power endgame, where a fixed 0.1 interceptor costs about as much as its bullet. The instrumented opponent's rating logging is a dev-only patch to the private copy (`vg` events) and is never used at runtime or in registered games.

Decision state: not registered. My expectation is that a pilot of this WIP would likely produce at least one round win but fail the registered >20% share gate. The next development target is the opponent's gun-dependent off-centre aims. Any candidate needs its own frozen source/JAR, prospective registration, independent review and fresh games.

Scripts: `predrule.py` (offline rule validation over dev logs). The dev harness and classifier are in `../aegis-shield-v1/development/tools/`.

## Reproducibility and provenance

- `raw/<run>/trace.jsonl.gz` holds Aegis's own default-off diagnostic trace, and `raw/<run>/diamond.jsonl.gz` holds the opponent rows logged by a **private dev-only instrumented copy**. These are the exact inputs for every statistic quoted here, from runs `instr1-4`, `aegis8/9/10/18/20/22` (evolving 0.1.0 development sources), `v2a–c`, `vg1–5` and `eg1`.
- `dev-only-instrumented-opponent.patch` is the instrumentation, applied to the pinned `diamond_port/gun.py`. It adds logging of `aim` (selected gun and angle), `break` (wave label) and, from `vg1` onward, `vg` (virtual-gun scores) events. It changes no opponent decision. Runs before `vg1` contain no `vg` rows.
- `python3 -I benchmarks/aegis-v2-dev/analyses.py` reproduces every statistic from those retained files: the screen table, the 2,945-aim self-prediction comparison, label-bias conditioning, per-round gun and centre shares, virtual-gun disagreements, off-centre persistence, block share by power and the line-error distribution. It fails on any missing input. Its output is retained as `analyses-output.txt`. The script's docstring defines the joins, windows and timing map, including: the opponent aims at tick T for a bullet created at N = T + 2; the runtime rule uses legal scan speeds at N − 3 and N − 2 and the heading at N − 1, all received before the drop is detected at scan N; so the comparison has no look-ahead.
- `predrule.py` and `vgstat.py` are the original exploratory scripts. They read local scratch paths and silently skipped missing runs, so they are superseded by `analyses.py` and kept only for history.
