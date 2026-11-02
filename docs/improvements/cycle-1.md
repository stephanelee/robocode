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

Pending. Record actual outcomes and the retain/reject decision here after the preregistered comparison finishes. If rejected, revert gameplay and its tracked JAR while retaining the experiment harness and evidence.
