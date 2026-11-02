# Cycle 1: back-as-front steering

Issue: [#10](https://github.com/stephanelee/robocode/issues/10). Previous source: `86e4a2b7c112ffd054475a3b5fc1cf0c9243b0bb`.

## Preregistered hypothesis and decision

This plan is committed before candidate battle outcomes are collected. The initial Retcon integration baseline ran 20 training rounds plus five evaluation rounds for each of fixed minimum risk, cold DQN, and trained DQN. It exposed a stronger challenge: cold DQN won 0/5 rounds, scored 173 against Retcon's 844, and hit walls 23 times. Trained DQN won 0/5 with 24 wall collisions. These short, unpaired games establish a problem, not a precise strength estimate.

AISurvivor currently turns its body toward every requested travel heading and always drives forward. At speed 8 the maximum body turn is 4 degrees per turn, so a desired direction behind the bot can produce a long turning arc. The sole gameplay candidate in this cycle uses back-as-front steering: if the desired travel direction is more than 90 degrees from the body heading, target the opposite body heading and drive backward. The engine still applies normal acceleration and deceleration. Wall smoothing, targeting, power selection, DQN weights/configuration, and reward rules are unchanged.

Primary experiment: previous versus candidate JAR, each in cold frozen DQN evaluation against the same pinned Retcon source, using bot seeds **20261102, 20261103, 20261104**, **20 rounds per seed per version**, and no training or imported checkpoint. Total planned primary budget: **120 evaluation rounds**. Alternate previous/candidate launch order between seed blocks. Each game gets fresh isolated directories and cold models. Matching bot seeds initialize matching networks across versions; server starting positions and scheduling remain unpaired and unseeded.

The primary outcome is the equal-weight mean of the three game score shares, `AISurvivor score / (AISurvivor score + Retcon score)`. Scores are compared only at identical round budgets and opponent configuration. Secondary measurements are official wins, actual bullet hit rate, damage dealt/taken, wall collisions per 1,000 bot turns, and skipped turns.

Retain the gameplay candidate only if all primary checks hold:

- Mean score share improves by at least **5 percentage points**.
- Score share improves in at least **two of three** seed blocks.
- No seed block loses more than **5 percentage points** of score share.
- Candidate total wins are no more than **three below** the previous version.
- Candidate records **zero skipped turns**.
- Candidate wall collisions per 1,000 turns are at most **1.2 times** the previous version's rate; if previous has zero collisions, candidate must also have zero.

An early rejection is permitted after the first completed 20-round block per arm only if candidate score share drops by at least 15 percentage points and candidate has zero wins while previous has at least four. A process failure or invalid evidence stops the experiment; it is never counted as a zero score. Any corrected retry must retain and disclose the failed attempt separately.

If primary checks pass, run a secondary fixed minimum-risk comparison: the same three seed labels, **five rounds per version per block** (30 additional rounds). Fixed policy does not use bot RNG, so these are independent game replicates, not seed-induced strategy variants. Report this diagnostic alongside the primary decision; a material contradictory result requires further review before retaining the change.

Report every seed block, total counts, score-share deltas, and descriptive Wilson intervals for aggregate win proportions. The round observations share a game and an adaptive opponent, so binomial intervals are descriptive and may understate uncertainty. Three game blocks and unpaired starts do not justify strong causal or held-out generalization claims. This is adaptation to Retcon, not a claim of general bot strength. Do not change thresholds after seeing outcomes.

## Evidence requirements

Retain immutable snapshots and SHA256 hashes of both version JARs, source revisions, the checkpoint (if a later trained-policy comparison uses one), runtime version, Retcon source/API pin, runner asset hashes, seed/order/budget, raw official battle results, per-round bot telemetry, and runner logs. Trained comparisons must load byte-identical checkpoint copies in both arms and verify them unchanged afterward.

The comparison harness validates participant identities, policy/seed/round counters, finite measurements, and checkpoint status before computing any decision. The previous and candidate bot descriptors/launchers must match; changing ancillary bot configuration is outside this cycle.

## Results

**Decision: reject the gameplay candidate.** All 120 planned primary evaluation rounds completed. Mean score share fell from 11.545% to 10.262% (-1.283 percentage points), and one seed block lost 9.476 points. These fail the preregistered minimum gain and maximum block loss criteria. Thresholds were not changed after observing outcomes. The conditional fixed-policy diagnostic was therefore not run.

| Bot seed | Previous AIS / Retcon score | Candidate AIS / Retcon score | Previous share | Candidate share | Change | Wins previous / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261102 | 463 / 3141 | 464 / 2712 | 12.847% | 14.610% | +1.763 pp | 0 / 1 |
| 20261103 | 232 / 2940 | 332 / 2638 | 7.314% | 11.178% | +3.864 pp | 0 / 1 |
| 20261104 | 450 / 2659 | 144 / 2737 | 14.474% | 4.998% | -9.476 pp | 1 / 0 |

Each cell covers 20 rounds; the primary score-share mean weights the three blocks equally. Previous won 1/60 rounds, candidate 2/60. Descriptive 95% Wilson intervals are 0.295%–8.855% and 0.919%–11.364%, respectively. Their limitations are stated above; these short, unpaired games do not establish a precise causal effect.

The candidate reduced wall collisions from 2,208 in 27,353 bot turns to 575 in 36,074 turns: **80.72 → 15.94 collisions per 1,000 turns**. That is a useful movement diagnostic, but the primary score test still failed. Actual bullet hit rate fell from 134/1,412 (9.49%) to 126/1,700 (7.41%). Damage dealt fell from 931.08 to 800.06; damage taken fell from 4,252.00 to 3,954.58. Both versions recorded zero skipped turns. The lower collision rate and one additional win do not override the registered score criteria.

The final PR restores `AISurvivor.java` and the tracked JAR byte-for-byte to the accepted previous version, removes the experimental steering helper/tests and temporary comparison workflow, and retains the reusable harness and measured evidence. Issue #10 remains open as an unresolved improvement hypothesis. Cycle 2 starts from the original steering behavior.

## Provenance and reproduction

- [Successful measured run](https://github.com/stephanelee/robocode/actions/runs/41491118125), job `125434047764`, evaluated candidate `b10b9b88bfcf8c6646e4e0a35385e4b1399e1117` against previous `86e4a2b7c112ffd054475a3b5fc1cf0c9243b0bb`.
- [Registered plan commit](https://github.com/stephanelee/robocode/commit/c7b08e33927f2fd961d2e81d2076b4039255f5e3) predates candidate outcomes. The harness read that commit's exact Git blob; [the immutable plan copy](../../benchmarks/cycle-1/preregistered-plan.md) is retained with the data.
- [Canonical comparison JSON](../../benchmarks/cycle-1/comparison.json) preserves every block, launch order, decision check, both complete bundle hashes, opponent pin, and runtime. [Raw per-round telemetry, official battle JSON, runner logs, and candidate source patch](../../benchmarks/cycle-1/) are checked in for durable review.
- [Original artifact](https://github.com/stephanelee/robocode/actions/runs/41491118125/artifacts/13163668766) also contains both immutable complete JAR bundles. Artifacts may expire; source and recorded hashes allow rebuilding. Previous JAR SHA256: `0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2`. Candidate: `f67061d3cb1a82d381f80dc1b172b784d78d9e098f9ec368705dfbb34405dd47`.
- Bot build: Java 17. Match runtime: Temurin 25.0.5+8-LTS. Server/runner: Tank Royale 1.4.0. Opponent: unchanged Retcon 0.2.0 at source commit `28981c7a469ac064194b41001cd7183ec5ecf189`, using its own pinned bot API 1.3.1. Exact source/API/runner hashes are in the JSON.

To repeat the historical experiment, build isolated previous/candidate source checkouts at the revisions above and use the command in [version comparisons](../version-comparisons.md), supplying those bundle paths, source revisions, this plan path, and the registered commit. Fresh unpaired arena starts will produce different outcomes. The candidate patch is archived for inspection; it is not active gameplay code.
