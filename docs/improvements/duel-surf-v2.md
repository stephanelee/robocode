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

The text above this heading is the registered plan, unchanged since `8db129f3d405d12ea14fe662e7e7adf5366e3f69`; that commit was also the measured candidate source. The driver stored that commit's Git blob, with its LF line endings, byte-for-byte as `raw/preregistered-plan.md`.

The single registered run was [41556987786](https://github.com/stephanelee/robocode/actions/runs/41556987786): job 125658162180, artifact 13194444591, zip SHA256 `e197207d8ac1bc53630b3adc0a09dffd56e45ea8b29eab1c7bd8a5aff4b3c833`. Every step was green and there were no reruns.

- **Runtime:** actual Ubuntu 26.04.1 LTS. Java 17 JAR parity passed, and matches ran on Temurin 25.0.5+8.
- **Candidate JAR:** `d047bf99325d2ad73aedd87eac6a4baa1677170afff0c86b948f03ea93390e1d`. The baseline JAR hash was verified before launch, and bundle hashes were checked before every battle.
- **Completion:** all 700 registered scored rounds completed.
- **Online models:** all 16 are preserved, and their hashes match `experiment.json`.

**Decision: retain.** Stages A, B and C all passed under the registered Student-t rule. This v2 evidence is the basis for merging the gameplay change. It is a Retcon-specific tactical gain, not yet a fresh confirmation of beating Retcon.

### Stage A: 320 fixed minimum-risk Retcon rounds (pass)

| Metric (160 rounds per arm) | Baseline | Candidate |
| --- | ---: | ---: |
| Mean block score share | 17.35% | **52.83%** |
| Round wins | 8 | **80** |
| AIS turns survived | 108,578 | 169,221 |
| AIS damage taken per 1,000 turns | 89.01 | **20.19** |
| AIS damage dealt per 1,000 turns | 33.68 | 26.88 |
| AIS shots / hit rate | 5,460 / 9.304% | 9,409 / 7.886% |
| Wall collisions (per 1,000 turns) | 386 (3.555) | 13 (0.077) |
| AIS skipped turns | 0 | 0 |

| Seed | Order | Baseline share / wins | Candidate share / wins | Delta |
| ---: | --- | ---: | ---: | ---: |
| 20261501 | previous, candidate | 13.08% / 0 | 53.14% / 10 | +40.06 |
| 20261502 | candidate, previous | 19.00% / 1 | 49.69% / 10 | +30.70 |
| 20261503 | previous, candidate | 24.97% / 2 | 45.52% / 8 | +20.55 |
| 20261504 | candidate, previous | 19.51% / 2 | 44.60% / 8 | +25.09 |
| 20261505 | previous, candidate | 12.83% / 0 | 72.19% / 15 | +59.36 |
| 20261506 | candidate, previous | 16.86% / 1 | 47.55% / 9 | +30.69 |
| 20261507 | previous, candidate | 17.04% / 1 | 57.86% / 10 | +40.82 |
| 20261508 | candidate, previous | 15.52% / 1 | 52.11% / 10 | +36.59 |

The mean delta was **+35.481 pp** with an SE of 4.220 pp. The Student-t lower bound, `mean − 1.894579 × SE`, was **+27.486 pp**.

Checks:
- mean ≥ +5 pp: pass
- t lower bound > 0: pass
- wins ≥ baseline: pass (80 vs 8)
- zero skipped turns: pass
- wall rate ≤ baseline + 10 per 1,000 turns: pass (0.077 vs 3.555)

Every block improved, by at least 20.6 pp.

### Mechanism (Retcon offline native export)

| Retcon counter, 160 rounds per arm | Baseline | Candidate |
| --- | ---: | ---: |
| Actual hits / shots | 1,304 / 7,026 (**18.56%**) | 667 / 9,156 (**7.28%**) |
| Linear virtual hits / completed | 1,096 / 6,813 (16.09%) | 547 / 8,962 (6.10%) |
| Head-on virtual hits / completed | 1,068 / 6,813 (15.68%) | 400 / 8,962 (4.46%) |

H1 is supported: Retcon's actual hit rate fell by about 61%. The gain comes from survival. AISurvivor's own hit rate and damage dealt per turn were again slightly lower.

### Stage B: official samples, 10 fixed rounds per arm (pass)

| Opponent | Baseline share / wins / walls | Candidate share / wins / walls |
| --- | --- | --- |
| Target | 99.890% / 10 / 6 | 100.000% / 10 / 6 |
| SpinBot | 97.839% / 10 / 2 | 97.145% / 10 / 0 |
| Walls | 96.936% / 10 / 3 | 100.000% / 10 / 1 |

### Stage C: 320 default online-training Retcon rounds (pass)

The mean delta was +0.894 pp (SE 1.206). The Student-t lower bound was −1.391 pp, which clears the −5 pp non-inferiority margin. Wins went from 0 to 4, with zero skipped turns.

| Seed | Baseline share / wins | Candidate share / wins |
| ---: | ---: | ---: |
| 20261501 | 16.19% / 0 | 13.71% / 0 |
| 20261502 | 13.49% / 0 | 19.28% / 1 |
| 20261503 | 15.11% / 0 | 19.04% / 1 |
| 20261504 | 13.83% / 0 | 10.55% / 0 |
| 20261505 | 13.24% / 0 | 12.42% / 0 |
| 20261506 | 13.00% / 0 | 17.56% / 2 |
| 20261507 | 11.35% / 0 | 10.57% / 0 |
| 20261508 | 13.39% / 0 | 13.62% / 0 |

In this mode Retcon's actual hit rate fell from 40.99% to 29.73%.

A plausible explanation for the small default-mode effect is that the DQN selects action 0 only part of the time. Action-frequency telemetry was not captured, so this is untested.

### Comparison with v1, and limitations

- **v1 vs v2:**
  - v1 measured 14.79%→58.56% on different seeds; v2 measures 17.35%→52.83%. The runs are separate unpaired samples.
  - The difference between the two candidate means (5.7 pp) is within about 1.2 combined SEs. This run therefore does not show whether the corrected physics helps or hurts relative to v1.
- **Beating Retcon (not established):** the candidate averages above 50% and wins exactly half the rounds (80 of 160). It exceeded 50% in only 4 of 8 blocks. A majority of rounds and blocks would require a separately registered, fresh, frozen-version confirmation.
- **Default training mode:** this mode remains about 15% share.
- **Guards:** the wins and wall guards are uncalibrated under an effect (PR #37 review correction).
- **Sampling:** starts are unpaired and unseeded, and Retcon adapts within each game.
- **Retcon statistics:** Retcon's statistics are offline diagnostics only.

PR: [#38](https://github.com/stephanelee/robocode/pull/38). Evidence: [raw evidence](../../benchmarks/duel-surf-v2/README.md).
