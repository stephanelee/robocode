# Shot-reactive 1v1 movement against head-on and linear aim

Status when this file is first committed: **preregistered, not yet measured.** The hypothesis, budget, metrics and decision rule are fixed before any registered match. Results are appended under the final heading in a later commit, and nothing above that heading is edited after the run starts.

## Baseline, candidate, environment

| Item | Value |
| --- | --- |
| Accepted baseline | master `241156f7d6b78c4ca2365859884b1486c9f41194`, JAR SHA256 `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883` |
| Candidate | the commit containing this file, with its source, tests and Java 17 JAR; the workflow records the exact SHA and JAR hash |
| Opponents | Retcon 0.2.0 at `28981c7a469ac064194b41001cd7183ec5ecf189` (API 1.3.1, JDK 25); Tank Royale 1.4.0 Runner; official 1.4.0 Target, SpinBot and Walls |
| Build and runner | Java 17, exactly `ubuntu-26.04` |

## Facts motivating the change

- **Retcon's duel gun:** it fires only along iterative wall-clamped linear aim or head-on aim, chosen by cumulative virtual hit rate. It fires when the current gun heading is within `atan(18/d)` of the chosen aim. Sources: [Retcon.java](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/Retcon.java), [Gun.java](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/framework/roles/Gun.java).
- **How often it hits us:** in the [A/A calibration](../aa-calibration.md) (400 rounds), Retcon hit the accepted bot with 18.6% of its shots. Both of its virtual guns score similarly against us in every recorded cohort.
- **Our fixed duel policy:** action 0 re-selects a minimum-risk destination point every turn. It never commits to a trajectory that leaves both aim points.
- **What has already failed:** aim-corridor penalties inside that point scorer ([#32](https://github.com/stephanelee/robocode/pull/32), [#34](https://github.com/stephanelee/robocode/pull/34)) did not reduce Retcon's hit rate.

## Candidate change (one coherent movement controller)

The scope is narrow. In **1v1 only** (`getEnemyCount() == 1`, one scanned enemy) and **for action 0 only**, the pure `DuelSurf` controller replaces minimum-risk movement:

- **Body and travel direction.** The body stays near perpendicular to the opponent. Travel is tilted toward the opponent when it is far and away when it is close, by `(range − 450)/10` degrees, capped at ±30°.
- **Options.** Each turn there are three: full speed one way along that line, full speed the other way, or stop. The bot drives forward or backward, whichever keeps the body orientation, so reversals never need a U-turn.
- **Scoring.** Each option is held and simulated with the engine's acceleration (1) and deceleration (2) until every legally inferred, not-yet-passed wave arrives. Its score is the minimum, over waves, of the lateral distance to the nearer of two reconstructed aim points, capped at 60.
  - The head-on and linear aim points are rebuilt with Retcon's published algorithms from **our own recorded state at the inferred aim turn** (`AimCorridor`).
  - Arrival uses Retcon's own sweep convention: at turn `d+k`, the wave covers `v·k` to `v·(k+1)`.
- **Tie-breaks.** +3 for keeping the current direction (hysteresis) and +1 for moving rather than stopping.
- **Walls.**
  - A path that moves *further* into a 30-unit wall envelope within the horizon is rejected.
  - A bot already touching a wall may slide along it or leave.
  - Only when both directions along the line are blocked (corners) does the line rotate toward the arena centre, in 10° steps up to 90°.

Unchanged: melee movement, actions 1–5, targeting, firing, radar, the DQN, rewards, persistence, and the minimum-risk code itself. In default training mode the DQN still picks actions; when it picks action 0 in a duel, the new controller runs.

### Information boundary

The controller uses only AISurvivor's own state history, the opponent's scanned position, and waves inferred from energy drops. Retcon's source chose the two model families; its private statistics are offline diagnostics only.

### Development disclosure (exploratory, not evidence)

Before this registration, I made controller-design checks on local Ubuntu 24.04 with OpenJDK 17/25, using development seeds 9401–9402, 9501–9504, 9601–9603 only. Those games led to two fixes made before freezing:
- a corner rotation, after 41 wall hits in one game;
- a contact-escape rule, after 244 wall hits in 3 Target rounds.

Development Retcon games scored 28.7% and 29.6% for the first build, then 39.6–73.9% for later builds (five games). Baseline development games in other sweeps scored roughly 15–29%. Damage taken fell to roughly 11–36 per 1,000 turns. A 1-D kinematic simulation idealized away walls, curvature and gun tolerance; under those conditions the controller evaded both models.

None of these numbers set a threshold below. Registered seeds are new.

## Hypotheses (not facts)

- **H1.** Retcon's actual and virtual hit rates on AISurvivor fall.
- **H2.** AISurvivor's fixed-policy Retcon score share rises by at least 5 pp, with statistical support.
- **H3.** There is no material regression against the samples or in default online training.

## Budget and procedure (single run, fixed in advance)

The budget follows the A/A calibration's proposal (PR [#37](https://github.com/stephanelee/robocode/pull/37)): **8 blocks of 20 rounds per arm**, giving 80% power at +5 pp. A block-delta SD of 5.59 pp was measured with identical bots.

All three stages run in one `Duel surf experiment` workflow job via `tools/block_experiment.py --blocks 8 --rounds 20 --seed-base 20261401 --sample-seed 20261409`. There is no early stopping, no favourable rerun and no seed replacement. An infrastructure failure marks the run invalid; partial evidence is preserved and any retry is disclosed.

- **Stage A, primary (320 Retcon rounds):** fixed minimum-risk, 8 blocks with seeds 20261401–20261408 and alternating arm order.
- **Stage B, sample regression (60 rounds):** Target, SpinBot and Walls, 10 fixed rounds per arm. Always run.
- **Stage C, online-training veto (320 Retcon rounds):** run only if Stage A passes. Real default `train` mode, fresh models for every arm and block, same seeds.

The maximum budget is 700 scored rounds.

## Acceptance criteria (all must hold; unrounded values)

These criteria use the calibration-based rule proposed in PR #37 and are registered here for this experiment. I chose the guards from the calibration because the A/A data showed the old ratio-based wall and wins guards to be dominated by noise.

**Stage A**
1. Mean paired block delta (candidate − baseline score share) **≥ +5 pp**.
2. One-sided 95% lower bound, `mean − 1.6449 × SD/√8`, **> 0**.
3. Candidate total round wins **≥** baseline total round wins.
4. Candidate records **zero** skipped turns.
5. Candidate wall collisions per 1,000 turns **≤ baseline + 10**. This is an absolute margin; identical arms differed by 7.1 per 1,000 turns.

**Stage B:** for each sample bot, candidate share ≥ baseline share − **5 pp**, all 10 rounds per arm, and zero candidate skipped turns.

**Stage C (non-inferiority):** one-sided 95% lower bound of the mean online block delta **≥ −5 pp**, and zero candidate skipped turns.

**Decision:**
- **Retain** only if A, B and C all pass. That would be a Retcon-specific tactical gain under this protocol, not proof of beating Retcon overall or of general strength.
- **Otherwise reject:** restore baseline gameplay and the JAR exactly, and keep the patch, protocol and evidence.

The reviewer must accept the calibration-based rule. If the reviewer rejects that rule, a retention cannot merge gameplay on this evidence alone.

## Mechanism observables (descriptive)

Reported per arm and stage:
- Retcon's actual and virtual (linear and head-on) hits per shot, from offline native exports;
- AISurvivor damage taken and dealt, turns survived, wins, hit rate and wall collisions.

## Results

The text above this heading is the registered plan, unchanged since `5f5ff460eaad190a6bc3dec507f6eee732474164`, which was also the measured candidate source. The driver stored that commit's Git blob, which has LF line endings, as `raw/primary/preregistered-plan.md`. That copy is byte-identical to the blob. A working-tree checkout converted to CRLF differs only in line endings. *(Wording corrected after the PR #38 review, 2026-11-03.)*

There was a single registered run, [41551240407](https://github.com/stephanelee/robocode/actions/runs/41551240407), with no reruns:
- **CI:** job 125639064038, every step green.
- **Artifact:** 13191578758, zip SHA256 `178e09d50e432634505b6114dae1cb8c565999dfa448dabdebec1d364dc8b1de`.
- **Runtime:** actual Ubuntu 26.04.1 LTS. The Java 17 JAR parity check passed, and matches ran on Temurin 25.0.5+8.
- **Candidate JAR:** `1a588ead187a7fa90fbf1468cb383f5f75696b6fbfc4fcef9621b05c09d59d02`. The baseline JAR hash was verified before launch, and bundle hashes were verified before every battle.
- **Rounds:** all 700 registered scored rounds completed (320 primary, 60 sample, 320 online).

**Decision: retain.** Stages A, B and C all passed. This is a Retcon-specific tactical gain under this protocol. The retention is conditional on the reviewer accepting the calibration-based rule. The review correction on PR #37, which came after this registration, is addressed below. Beating Retcon still needs a separately registered fresh confirmation; see the limitations.

### Stage A: 320 fixed minimum-risk Retcon rounds (pass)

| Metric (160 rounds per arm) | Baseline | Candidate |
| --- | ---: | ---: |
| Mean block score share | 14.79% | **58.56%** |
| Round wins | 6 | **87** |
| AIS turns survived | 109,751 | 175,307 |
| AIS damage taken per 1,000 turns | 87.03 | **13.43** |
| AIS damage dealt per 1,000 turns | 28.48 | 24.96 |
| AIS shots / hit rate | 5,465 / 8.381% | 10,067 / 7.063% |
| Wall collisions (per 1,000 turns) | 717 (6.533) | 29 (0.165) |
| AIS skipped turns | 0 | 0 |

Per block (score share and wins):

| Seed | Order | Baseline | Candidate | Delta |
| ---: | --- | ---: | ---: | ---: |
| 20261401 | previous, candidate | 17.65% / 1 | 54.32% / 10 | +36.68 |
| 20261402 | candidate, previous | 17.59% / 2 | 66.03% / 13 | +48.44 |
| 20261403 | previous, candidate | 13.00% / 1 | 61.99% / 12 | +48.99 |
| 20261404 | candidate, previous | 15.15% / 1 | 48.73% / 8 | +33.58 |
| 20261405 | previous, candidate | 11.25% / 0 | 61.73% / 11 | +50.47 |
| 20261406 | candidate, previous | 16.38% / 1 | 56.39% / 10 | +40.01 |
| 20261407 | previous, candidate | 14.63% / 0 | 67.93% / 14 | +53.30 |
| 20261408 | candidate, previous | 12.70% / 0 | 51.39% / 9 | +38.69 |

The mean delta was **+43.769 pp** with SE 2.602 pp.
- **Registered bound:** the z lower bound is **+39.489 pp**.
- **Corrected bound:** under the Student-t bound from the PR #37 review correction (t = 1.8946, df 7), the lower bound is **+38.84 pp**, so the result is unchanged.

Checks:

| Check | Result |
| --- | --- |
| Mean ≥ +5 pp | pass |
| Lower bound > 0 | pass |
| Wins ≥ baseline | pass (87 vs 6) |
| Zero skipped turns | pass |
| Wall rate ≤ baseline + 10 per 1,000 turns | pass (0.165 vs 6.533) |

Every block improved by at least 33.6 pp. The candidate's share exceeded 50% in 7 of 8 blocks.

### Mechanism, from Retcon's offline native export

| Retcon counter, 160 rounds per arm | Baseline | Candidate |
| --- | ---: | ---: |
| Actual hits / shots | 1,337 / 7,082 (**18.88%**) | 521 / 9,829 (**5.30%**) |
| Linear virtual hits / completed | 1,139 / 6,857 (16.61%) | 425 / 9,629 (4.41%) |
| Head-on virtual hits / completed | 1,047 / 6,857 (15.27%) | 375 / 9,629 (3.89%) |

**H1 is supported.** Both of Retcon's modelled guns, and its actual gun, lost about three quarters of their hit rate. AISurvivor survived 60% more turns, and its damage taken per turn fell 85%.

The score gain comes from survival. Our own hit rate and damage dealt per turn were slightly *lower*, so targeting is a separate opportunity.

### Stage B: official samples, 10 fixed rounds per arm (pass)

| Opponent | Baseline share / wins / walls | Candidate share / wins / walls |
| --- | --- | --- |
| Target | 99.890% / 10 / 0 | 100.000% / 10 / 0 |
| SpinBot | 99.004% / 10 / 5 | 99.024% / 10 / 4 |
| Walls | 98.150% / 10 / 6 | 99.354% / 10 / 4 |

### Stage C: 320 default online-training Retcon rounds (pass)

| Seed | Baseline share / wins | Candidate share / wins |
| ---: | ---: | ---: |
| 20261401 | 13.79% / 0 | 20.65% / 1 |
| 20261402 | 16.86% / 0 | 20.92% / 1 |
| 20261403 | 11.12% / 0 | 15.57% / 0 |
| 20261404 | 15.09% / 0 | 13.72% / 0 |
| 20261405 | 16.45% / 0 | 18.48% / 1 |
| 20261406 | 15.75% / 0 | 21.61% / 1 |
| 20261407 | 18.39% / 0 | 20.48% / 1 |
| 20261408 | 8.68% / 0 | 12.74% / 0 |

- **Score:** the mean delta was +3.507 pp. The registered z lower bound was +2.011 pp, against a non-inferiority margin of −5 pp. The t lower bound was +1.784 pp.
- **Other measures:** wins went from 0 to 5, with zero skipped turns.
- **Retcon hit rate:** Retcon's actual hit rate fell from 40.18% to 32.19% in this mode.
- **Why the gain is small here (hypothesis):** default training mode lets the DQN choose among six actions, and the duel controller runs only when the DQN picks action 0. A plausible explanation for the small default-mode benefit is that action 0 is chosen only part of the time. Action-frequency telemetry was not captured, so this is untested. *(Qualified after the PR #38 review.)* This stage is a safety veto, not evidence of a learning benefit.

### Interpretation and limitations

- **Fact:** under the registered fixed duel policy, AISurvivor now outscores pinned Retcon 0.2.0 on average (58.56% share), wins a majority of rounds (87 of 160) and wins 7 of 8 blocks.
- **Hypothesis:** whether AISurvivor robustly beats Retcon in fresh games is a separate question. Per the [strategy](../retcon-strategy.md), that needs a frozen-version confirmation on fresh games, with its own prospectively registered budget and pass rules. Development games and this run cannot serve as that confirmation.
- **Default mode:** default training mode remains much weaker (about 18% share). Whether this is because the DQN rarely selects action 0 is a hypothesis; action frequencies were not recorded. Making the duel controller dominant in default play would be a separate, registered change.
- **Rule caveats** (disclosed in the PR #37 review correction):
  - The registered z bound is a large-sample approximation.
  - The wins and wall guards are uncalibrated under an effect.
  - The effect here is about 14.9 standard errors above the +5 pp floor ((43.769 − 5) / 2.602), so neither caveat changes the decision. *(Arithmetic corrected after the PR #38 review; the earlier text said "about eight".)*
- **Exclusions:** starts are unpaired and unseeded, Retcon adapts within each game, and Retcon counters are offline diagnostics only. The development games disclosed above are excluded from this decision.

PR: [#38](https://github.com/stephanelee/robocode/pull/38). Evidence: [raw evidence](../../benchmarks/duel-surf/README.md).

## Review outcome and follow-up (2026-11-03)

The independent PR #38 review reproduced all 700 telemetry rows, the official games, the Retcon exports, the 180 file hashes, the 16 online trajectories and checkpoints, the source/JAR identity and the registered decisions.

It also found two physics defects in the controller's own simulation:
1. `nextSpeed` used textbook braking instead of the pinned engine's `calcNewBotSpeed`. The engine stops immediately for a target of 0, and goes 8 → +3 on a full reversal.
2. `score()` translated along the proposed direction, whereas the server moves along the current heading and only then applies a speed-limited turn. A near-wall counterexample showed a chosen reversal that would actually hit the wall.

**What this result does and does not establish.**
- These defects do not invalidate the measured result above, which concerns the code exactly as measured.
- Fixing them changes the candidate. Gameplay is therefore not merged on this run's evidence.
- The corrected controller is registered as a fresh comparison in [duel-surf-v2](duel-surf-v2.md).
- The 16 online-training `model.dat` files are now preserved under `benchmarks/duel-surf/raw/online/`, with hashes matching `experiment.json`.

