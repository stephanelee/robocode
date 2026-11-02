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

Pending. No cycle 2 candidate battle outcomes have been collected. Record the cycle 1 integration decision, final compared revisions/JAR hashes, every attempted run, primary and conditional secondary evidence, and final retain/reject decision here after the experiment.
