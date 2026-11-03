# Shot-reactive 1v1 movement with engine-conformant simulation (v2)

Status when this file is first committed: **preregistered, not yet measured.** The hypothesis, budget, metrics and decision rule are fixed before any registered match. Results are appended under the final heading in a later commit; nothing above that heading is edited after the run starts.

## Why a fresh comparison

[Duel-surf v1](duel-surf.md) passed its registered 700-round protocol. Under the fixed policy, Retcon score share rose from 14.79% to 58.56%.

The independent PR #38 review reproduced that result, but found two defects in the controller's internal simulation:

1. `nextSpeed` did not match the pinned engine's `calcNewBotSpeed`. In the engine, stopping is immediate, and an 8 → −8 reversal gives +3 on its first turn.
2. `score()` moved along the proposed direction straight away. The engine updates speed, moves along the **current** heading, and only then turns, by at most `10 − 0.75·|newSpeed|`. A near-wall counterexample showed a chosen reversal that the engine would drive into the wall.

Fixing these changes the candidate. The v1 evidence is preserved unchanged, and this document registers a fresh comparison of the corrected controller. The v1 result is not relabelled.

## Baseline, candidate, environment

- **Accepted baseline:** master `dccc19d1d37ae07ce37ea5d300659d45395bfc1f`. Its gameplay is identical to `241156f`, and its runnable JAR SHA256 is `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883`.
- **Candidate:** the commit containing this file, with its source, tests and Java 17 JAR. The workflow records the exact SHA and JAR hash.
- **Opponents:**
  - Retcon 0.2.0 at `28981c7a469ac064194b41001cd7183ec5ecf189` (Bot API 1.3.1, run on JDK 25).
  - Tank Royale 1.4.0 Runner.
  - The official 1.4.0 Target, SpinBot and Walls bots.
- **Build and runner:** Java 17, exactly `ubuntu-26.04`.

## Candidate change

The candidate is the v1 `DuelSurf` controller. As in v1, it runs only in 1v1 and only for action 0; melee movement and actions 1–5 are unchanged. It keeps v1's options, scoring, tilt, hysteresis, wall envelope and corner rotation. It also uses the same `AimCorridor` reconstruction from our own recorded state.

Only the internal simulation changes:

- **Speed update.** Speed is now an exact port of Tank Royale 1.4.0 `calcNewBotSpeed` (`rules/math.kt`), including immediate stopping and the zero-crossing rule.
- **Engine order.** Each simulated turn runs in engine order:
  1. the speed update;
  2. movement along the **current** heading;
  3. a turn limited by the bot's own request at the old speed and by the engine limit at the new speed.
- **Wall contact.** A path that crosses the legal 18-unit wall boundary is now rejected as a hit. The old envelope rule (rejecting movement further into the 30-unit envelope) still applies.

The tests now include:

- conformance fixtures for engine stopping and reversal, in both directions;
- the engine turn limit;
- the reviewer's counterexample, `choose(780, 300, 0, 8, …, 800, 600)`, as a regression: the first engine step must stay at x ≤ 782.

## Development disclosure (exploratory, not evidence)

I ran three development checks of the corrected build on local Ubuntu 24.04 with OpenJDK 17 and 25:

- two 5-round Retcon games on seeds 9701–9702, which scored 64.6% and 30.4%;
- three 3-round sample games on seed 9703;
Across all five games there were zero skipped turns and at most 2 wall hits per game.

These results did not set any threshold, and the registered seeds below are new. The v1 registered result is known, and this registration is not independent of it.

## Hypotheses (not facts)

- **H1:** with engine-conformant simulation, Retcon's actual and virtual hit rates on AISurvivor stay well below baseline.
- **H2:** AISurvivor's fixed-policy Retcon score share rises by at least 5 pp, with statistical support.
- **H3:** no material regression against the sample bots or in default online training.

## Budget and procedure (single run, fixed in advance)

The budget is 8 blocks of 20 rounds per arm, from the A/A calibration (PR #37, merged). The confidence bound is the **Student-t** one-sided 95% bound from the #37 review correction, with t(0.95, 7) = 1.894579. Under the calibration's normal model, this rule accepts a true +7.5 pp about 89% of the time and a true 0 about 0.5% of the time, counting score checks only.

All stages run in one `Duel surf v2 experiment` workflow job:

```
tools/block_experiment.py --blocks 8 --rounds 20 --seed-base 20261501 --sample-seed 20261509 --bound t
```

There is no early stopping, no favourable rerun and no seed replacement. An infrastructure failure marks the run invalid; its partial evidence is preserved and any retry is disclosed.

- **Stage A, primary (320 Retcon rounds):** fixed minimum-risk, 8 blocks with seeds 20261501–20261508, arm order alternating.
- **Stage B, sample regression (60 rounds):** Target, SpinBot and Walls, 10 fixed rounds per arm. Always run.
- **Stage C, online-training veto (320 Retcon rounds):** run only if Stage A passes. Uses the real default `train` mode, with fresh models per arm and block and the same seeds.

## Acceptance criteria (all must hold; unrounded values)

**Stage A**
1. Mean paired block delta ≥ **+5 pp**.
2. Student-t lower bound, `mean − 1.894579 × SD/√8`, **> 0**.
3. Candidate total round wins ≥ baseline total round wins.
4. **Zero** candidate skipped turns.
5. Candidate walls per 1,000 turns ≤ baseline + 10.

As the #37 correction states, guards 3 and 5 are not calibrated under an effect. They are kept from v1 unchanged rather than retuned.

**Stage B:** for each sample bot:
- candidate share ≥ baseline − **5 pp**;
- all 10 rounds per arm completed;
- zero candidate skipped turns.

**Stage C:** Student-t lower bound of the mean online delta ≥ **−5 pp**, and zero candidate skipped turns.

**Decision.** If all three stages pass, **retain**, and gameplay merges on this v2 evidence. Otherwise **reject**: restore baseline gameplay and the JAR exactly, and keep the patches, protocols and all evidence, including v1.

A retention is a Retcon-specific tactical gain. It is not yet a fresh confirmation of beating Retcon.

## Mechanism observables (descriptive)

Reported per arm and stage:
- Retcon's actual hits per shot, and its virtual linear and head-on hits per shot (offline exports only);
- AISurvivor's damage taken and dealt, turns, wins, hit rate and wall collisions.

## Results

Not yet measured.
