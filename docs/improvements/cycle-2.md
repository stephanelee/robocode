# Cycle 2: complete wall-smoothing search

Issue: [#11](https://github.com/stephanelee/robocode/issues/11). Tracker: [#8](https://github.com/stephanelee/robocode/issues/8).

## Preregistered hypothesis and decision

This plan is committed before candidate battle outcomes are collected. Preparation starts from `2af272c`, but the previous arm will be the accepted master revision after cycle 1 is retained or rejected and independently reviewed. Record that exact revision and both JAR hashes before launching this experiment. No cycle 2 battles, including candidate smoke battles, will run before the cycle 1 decision and integration gate.

The sole gameplay candidate changes wall-smoothing heading selection. The current search checks at most 36 headings in five-degree increments, covering only half a revolution, and can return the next heading without checking its endpoint. A direction that reaches the desired inner wall envelope may therefore be missed. The candidate checks all 72 headings in the requested rotation direction, including the desired heading, before choosing a fallback. Endpoints on the margin boundary are allowed within floating-point tolerance. If no sampled heading reaches the inner envelope, explicitly choose and check the inward heading toward the arena centre instead of returning an unchecked search heading. This is a heading lookahead heuristic, not a guarantee that the engine's actual turn and acceleration trajectory avoids walls.

A pure helper and focused tests will establish the intended behavior before measurement: a bot at `(120,120)` in an `800x600` arena with margin/stick `80`, desired heading `125`, and positive rotation must find the safe heading near `330`, beyond the old half-turn search. At `(18,18)`, where no length-80 endpoint can reach both inner margins, the fallback must point inward toward the centre. Also test preservation of an already-safe heading, reverse search direction, and finite normalized output. No targeting, firepower, rewards, model configuration, or checkpoint semantics change. Cycle 1's accepted steering behavior remains the same in both arms.

Primary experiment: previous versus candidate JAR, each in cold frozen DQN evaluation against the same pinned **Retcon 0.2.0 Java bot in 1v1 only**, using bot seeds **20261102, 20261103, 20261104**, **20 rounds per seed per version**, and no training or imported checkpoint. Total planned primary budget: **120 evaluation rounds**. Alternate previous/candidate launch order between seed blocks. Each game gets fresh isolated directories and cold models. Matching seeds initialize matching networks; server starting positions and scheduling remain unpaired and unseeded.

Use the immutable-plan comparison harness from cycle 1's merged PR #13 after that PR is independently reviewed and merged. Its validated raw evidence and decision computation must be used; no ad hoc score-only comparison replaces it. Record the plan commit, exact previous/candidate source revisions, pinned opponent tree/API, runtime, and JAR hashes before starting either arm.

The primary outcome is the equal-weight mean of the three game score shares, `AISurvivor score / (AISurvivor score + Retcon score)`. Compare scores only at identical round budgets and opponent configuration. Secondary measurements are official wins, actual bullet hit rate, damage dealt/taken, wall collisions per 1,000 bot turns, and skipped turns.

Retain the gameplay candidate only if all primary checks hold, using the same thresholds as cycle 1:

- Mean score share improves by at least **5 percentage points**.
- Score share improves in at least **two of three** seed blocks.
- No seed block loses more than **5 percentage points** of score share.
- Candidate total wins are no more than **three below** the previous version.
- Candidate records **zero skipped turns**.
- Candidate wall collisions per 1,000 turns are at most **1.2 times** the previous version's rate; if previous has zero collisions, candidate must also have zero.

An early rejection is permitted after the first completed 20-round block per arm only if candidate score share drops by at least 15 percentage points and candidate has zero wins while previous has at least four. A process failure or invalid evidence stops the experiment; it is never counted as a zero score. Preserve and disclose failed attempts separately from any corrected retry.

If primary checks pass, run the secondary fixed minimum-risk diagnostic with the same three seed labels and **five rounds per version per block** (30 additional rounds). Fixed policy does not use bot RNG; these are independent game replicates rather than seed-induced strategy variants. Report the diagnostic beside the primary decision. A material contradictory result requires further review before retaining the change.

Report every seed block, total counts, score-share deltas, and descriptive Wilson intervals for aggregate win proportions. Rounds share a game and an adaptive opponent, so binomial intervals are descriptive and may understate uncertainty. Three unpaired game blocks do not justify strong causal or held-out generalization claims. This is adaptation to Retcon, not a claim of general strength. Do not change thresholds after seeing outcomes.

## Evidence and integration requirements

Preserve immutable snapshots and SHA256 hashes of both complete bot bundles, source revisions, runtime, Retcon source/API pins, runner asset hashes, seed/order/budget, raw official results, round telemetry, and runner logs. Both arms must use matching descriptors and launchers. The harness must validate identities, policy/seed/round counters, finite measurements, and cold-checkpoint status before computing a decision.

Build the tracked candidate JAR from source with Java 17 and run the existing tests plus wall-smoothing regressions. Actual build/JAR-parity, sample smoke, and Retcon smoke CI must pass before independent merge. Keep these functional checks distinct from the preregistered primary measurement. The running interactive game is outside this experiment and must remain untouched.

If the candidate is rejected, revert its gameplay and tracked JAR while preserving the plan, useful regression evidence, comparison harness, raw experiment artifacts, and rejection report. No self-merge: an independent reviewer must assess correctness, evidence, and the retain/reject decision before merging the final PR.

## Results

**Rejected after all 120 planned primary rounds.** Cycle 1's rejection was merged as `18f2659f6f55f10a7a30b1ec54460eb7159ce101`, preserving original steering. That accepted version was compared with candidate `4ead24ad85b6851ab5d2f336b1fe32be15f6ace8` in one complete, validated attempt. No early stop or failed battle occurred. The mean score share improved from **13.68485% to 15.07555% (+1.39070 percentage points)**, below the registered minimum gain of 5 percentage points. All six other primary checks passed. The conditional fixed-policy diagnostic did not run because the primary decision was reject.

| Bot seed | Previous AIS / Retcon score | Candidate AIS / Retcon score | Previous share | Candidate share | Change | Wins previous / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261102 | 507 / 3068 | 617 / 3177 | 14.182% | 16.263% | +2.081 pp | 0 / 0 |
| 20261103 | 315 / 2867 | 402 / 2665 | 9.899% | 13.107% | +3.208 pp | 0 / 0 |
| 20261104 | 572 / 2798 | 483 / 2563 | 16.973% | 15.857% | -1.116 pp | 0 / 1 |

Each cell covers 20 rounds. The mean score share weights the three games equally. Previous won 0/60 rounds and candidate 1/60; descriptive 95% Wilson intervals are 0%–6.017% and 0.295%–8.855%, respectively. Rounds share games and an adaptive opponent; these intervals and three unpaired game blocks do not establish statistical significance or general strength.

Walls fell from 1,821 collisions in 29,683 bot turns to 226 in 34,657: **61.348 → 6.521 collisions per 1,000 turns**. This supports the wall-avoidance diagnostic, but does not override the preregistered promotion rule. Actual bullet hit rate fell from 182/1,524 (11.942%) to 169/1,665 (10.150%). Damage dealt was 1,271.96 → 1,197.81 and damage taken 4,238.10 → 3,878.75. Both arms recorded zero skipped turns. One extra win and fewer collisions are insufficient for retention under this plan.

The final PR restores `AISurvivor.java` and the tracked JAR byte-for-byte to accepted `18f2659`, removes the candidate helper/tests and temporary measurement workflow, and retains the immutable plan, archived candidate source/test patch, raw evidence, and comparison validation tests. Issue #11 remains open as an unresolved improvement hypothesis. Cycle 3 starts from the original wall smoothing and steering.

## Provenance and reproduction

- [Successful measured run](https://github.com/stephanelee/robocode/actions/runs/41493197522), job `125441266315`, evaluated previous `18f2659f6f55f10a7a30b1ec54460eb7159ce101` against candidate `4ead24ad85b6851ab5d2f336b1fe32be15f6ace8`.
- [Registered plan commit](https://github.com/stephanelee/robocode/commit/f34cffbf5bbe8c79fb55d1b4acba5b581ac802d6) predates all candidate outcomes. The workflow verified its exact Git blob SHA256 `3e635c893281c938652506214ec708b2a464e0eedddcc870390eaf5d9c59cfc3` before starting either arm. The [immutable copy](../../benchmarks/cycle-2/preregistered-plan.md) is retained with the data.
- [Canonical comparison JSON](../../benchmarks/cycle-2/comparison.json) records every block, launch order, threshold check, bundle hashes, opponent/runtime provenance, and the validated budget. [Official results, telemetry, runner logs, and candidate source/test patch](../../benchmarks/cycle-2/) are checked in for durable review.
- [Original artifact](https://github.com/stephanelee/robocode/actions/runs/41493197522/artifacts/13164613547) includes both complete immutable bundles. Artifacts may expire; recorded source revisions/hashes allow rebuilding. Previous JAR SHA256: `0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2`. Candidate: `959e0494f0e6ff0787257b8e881f496b7d9264941ae744b1931463dc2a2def90`.
- Bot build: Java 17. Match runtime: Temurin 25.0.5+8-LTS. Server/runner: Tank Royale 1.4.0. Opponent: unchanged Java Retcon 0.2.0 at source `28981c7a469ac064194b41001cd7183ec5ecf189`, with its pinned API 1.3.1. Exact source/API/runner hashes are in the JSON. Bot descriptors and launchers were identical between arms.

To repeat this historical experiment, build isolated checkouts at the compared revisions and run [the generic comparison command](../version-comparisons.md) with this plan path and the full registered plan commit. Fresh unpaired starts will produce different outcomes. The candidate patch is retained for inspection and is not active gameplay code.
