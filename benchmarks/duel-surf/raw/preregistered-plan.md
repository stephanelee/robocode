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

Not yet measured.
