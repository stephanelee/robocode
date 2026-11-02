# Head-on/linear aim-corridor danger in minimum-risk movement

Status when this file is first committed: **preregistered, not yet measured.** This document fixes the hypothesis, budget, metrics and decision rule before any candidate match. Results are appended below the registration in a later commit; nothing above the results heading is edited after the run starts.

## Baseline and candidate

- Accepted baseline: master `cea19c8c451149cd5377db79972c591489a58ef1` (gameplay identical to accepted `204ad4c9`), runnable JAR SHA256 `1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a`.
- Candidate: the commit that contains this file, its source, tests and rebuilt Java 17 JAR. The workflow records the exact candidate SHA and JAR hash.
- Opponent: unchanged Retcon 0.2.0, source `28981c7a469ac064194b41001cd7183ec5ecf189`, Bot API 1.3.1, run under JDK 25 by the pinned Tank Royale 1.4.0 Runner. Official 1.4.0 Target, SpinBot and Walls for regression. AISurvivor built with Java 17; the job runs on exactly `ubuntu-26.04`.

## Evidence that motivates this change (facts)

1. Retcon's 1v1 gun chooses between exactly two aim models, iterative wall-clamped [linear](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/gun/LinearAimModel.java#L21-L39) and [head-on](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/gun/HeadOnAimModel.java), and only fires when its current gun heading is within `atan(18/distance)` of the chosen aim ([Gun.java](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/framework/roles/Gun.java#L52-L60)). Both models are pure functions of Retcon's own position, our position/heading/speed at its scan, bullet speed and the arena.
2. In the recorded fixed-minimum-risk observability cohorts (30 rounds, offline Retcon exports), Retcon hit AISurvivor with 15.5–18.8% of its actual shots, and **both** of its virtual guns scored similar rates against us: linear 96/591 and 100/666, head-on 78/591 and 98/666. The accepted arm of the fire-alignment screen (15 rounds) repeats the pattern: actual 128/714 (17.9%), linear 121/694, head-on 106/694. Our movement is therefore being hit by *both* simple models at comparable rates; neither model is being evaded.
3. The accepted minimum-risk `evaluateRisk` scores inferred enemy waves only by radial wave-front distance at the candidate's estimated arrival time. It contains no angular term for where a bullet on that wave could have been aimed. Its only angular term uses the enemy *body* heading, which is not the gun heading.
4. AISurvivor already infers enemy waves legally from energy drops between consecutive scans (`fireTurn = previous scan turn`, origin = shooter position at that scan). It knows its own position, heading and speed at every turn.

## Mechanism (hypothesis)

When a wave is inferred with fire turn `T`, reconstruct the two bearings Retcon's models would produce from the wave origin using **our own recorded state at turn `T`** and the inferred bullet speed:

- head-on: bearing from origin to our position at `T`;
- linear: Retcon's algorithm, stepping our turn-`T` velocity one turn at a time, clamping at 18 units from walls, until `bulletSpeed * k` reaches the predicted point (max 150 steps).

For each minimum-risk candidate point, while the wave has not yet reached our current position, add `5.0 * max(dangerHeadOn, dangerLinear)` to the risk. Each danger is `1 - offset/corridor` inside an angular corridor `atan(18/aimDistance) + atan(18/pointDistance)` around the ray, and 0 outside: the opponent's alignment tolerance plus our hit radius. If our state at `T` is not recorded, the wave contributes no angular danger (unchanged baseline behaviour). Everything else, including the existing radial term, wall/centre/enemy terms, targeting, firing, radar, DQN and the other five actions, is unchanged. Minimum risk is also the `flee` action and the surf fallback, so those inherit the term.

Hypotheses to be tested, not facts: (H1) steering candidate points out of both corridors lowers Retcon's actual hit rate on us; (H2) that improves AISurvivor's Retcon score share by at least the established 5-point tactical threshold; (H3) it does not harm sample bots or online-training play. Risks: a dominant corridor penalty may pull movement toward walls, increase wall collisions, or shorten range; Retcon may switch guns; Retcon's real gun heading and scan timing are only approximated; third-party/our own bullet events can create false waves.

Information boundary: inputs are only AISurvivor's own state history and legally inferred waves. Retcon's source is used to choose the two model families; Retcon's private statistics are never read during play and appear only as offline mechanism diagnostics below. No learned model is introduced: the existing learning experiments did not provide a deployable predictor, and the opponent's two aim models are deterministic, so explicit geometry is the testable choice here.

## Budget and procedure (single run, fixed in advance)

All stages run in one `Aim corridor experiment` workflow job via `tools/aim_corridor_experiment.py`. No early stopping, no reruns for favourable outcomes, no seed/round replacement. An infrastructure failure (missing telemetry, timeout, invalid Retcon export) marks the run **invalid**; partial evidence is preserved and disclosed, and any retry is reported as such.

- **Stage A — tactical primary (120 Retcon rounds):** `compare.py --category tactical --policy fixed-minrisk --purpose primary --rounds 20 --seeds 20261201,20261202,20261203 --require-retcon-stats`. Arm order alternates by block (previous/candidate, candidate/previous, previous/candidate). Seeds label independent unpaired games; fixed mode uses no bot RNG.
- **Stage B — sample regression (60 rounds):** official Target, SpinBot, Walls; fixed minimum-risk; 10 rounds per arm per opponent; seed label 20261204; orders previous/candidate, candidate/previous, previous/candidate. Always run.
- **Stage C — mandatory online-training veto (120 Retcon rounds):** run **only if Stage A retains**: `compare.py --category correctness --policy online-train --purpose primary --rounds 20 --seeds 20261201,20261202,20261203`; fresh models per arm/seed, real default `train` mode.

Maximum planned budget: 300 scored rounds (120 + 60 + 120).

## Acceptance criteria (all must hold)

Stage A uses `compare.py`'s existing unchanged tactical thresholds, evaluated on unrounded values:

1. Equal-weight mean Retcon score share gain **≥ +5 percentage points**.
2. At least **2 of 3** blocks strictly improve.
3. No block loses more than **5 percentage points**.
4. Candidate round wins no more than **3 below** baseline.
5. Candidate records **zero** skipped turns.
6. Candidate wall collisions per 1,000 turns **≤ 1.2×** baseline.

Stage B: for each of Target, SpinBot and Walls, candidate score share ≥ baseline − **5 percentage points**, all 10 rounds per arm complete, and candidate skipped turns = 0.

Stage C: `compare.py` correctness thresholds unchanged: mean share change ≥ 0, ≥ 2 of 3 blocks nonnegative, no block loses > 5 points, wins no more than 3 below, zero skipped turns, wall rate ≤ 1.2×.

Decision: **retain** only if A, B and C all pass. Otherwise **reject**: restore baseline gameplay source/tests and the tracked JAR exactly, keep the candidate patch, this protocol, the driver and all raw evidence. A retained result means "Retcon-specific tactical gain under this protocol"; it is not a claim of beating Retcon or of general strength.

## Mechanism observables (descriptive, not gates)

Reported per arm from Retcon's offline native export and AISurvivor telemetry: Retcon actual hits/shots on AISurvivor; Retcon virtual linear and head-on hits/completed shots; AISurvivor damage taken per 1,000 turns; AISurvivor hit rate, shots, wins, wall collisions. Expected direction under H1: lower Retcon actual and virtual hit rates. A mechanism change without the score gate is not an accepted improvement.

## Results

Everything above this heading is byte-identical to the registered plan at commit `b2f4ccbe69cf2cdecb103b9d501f9d3f934e56e5`, which was also the measured candidate source (`compare.py` stored the plan blob in `preregistered-plan.md`). The single registered [run 41532956899](https://github.com/stephanelee/robocode/actions/runs/41532956899) (job 125577564106, artifact 13183485154, zip SHA256 `c841e19b8d923fd2028913c8933080724151acba1783fe9b17d7db8a01d8c793`) ran on actual Ubuntu 26.04.1 (kernel 7.0.0-1017-azure, x86_64). It built with Temurin 17.0.21 and passed the Java 17 JAR parity check. Matches ran on Temurin 25.0.5+8. Candidate JAR SHA256 was `a804716b881710488c980e64b84428346060dc28a07554417e57eba8bfd05bac`; the baseline JAR hash was verified before launch. The experiment step completed. A later log-printing step exited 1 only because the online-veto file was legitimately absent; this did not affect the evidence. No match was rerun.

**Decision: reject.** Stage A failed, so Stage C (online veto) was not run, as registered. Stage B passed. Gameplay source, tests and tracked JAR are restored exactly to `cea19c8` (SHA256 `1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a`; a local OpenJDK 17 rebuild of the restored source matched all 800 entries).

### Stage A: 120 fixed minimum-risk Retcon rounds

| Metric (60 rounds per arm) | Baseline | Candidate |
| --- | ---: | ---: |
| Mean game score share | 15.6958% | 15.1193% |
| Round wins | 3 | 2 |
| AIS turns survived | 41,941 | 35,963 |
| AIS shots / hit rate | 2,112 / 8.097% | 1,779 / 10.062% |
| AIS damage dealt per 1,000 turns | 28.11 | 36.01 |
| AIS damage taken per 1,000 turns | 80.87 | 109.85 |
| Wall collisions per 1,000 turns | 2.670 | 14.737 |
| AIS skipped turns | 0 | 0 |

The block deltas were +0.868789, −2.448578 and −0.149835 percentage points (seeds 20261201/02/03), for a mean of −0.576541 pp. Checks:
- mean gain ≥ +5 pp: **fail**
- ≥ 2 improved blocks: **fail** (1)
- no block loses > 5 pp: pass
- wins not more than 3 lower: pass (3→2)
- zero skips: pass
- wall rate ≤ 1.2×: **fail** (5.52×)

### Mechanism, from Retcon's offline native export (not a gate)

| Retcon counter, 60 rounds per arm | Baseline | Candidate |
| --- | ---: | ---: |
| Actual hits / shots | 480 / 2,724 (17.62%) | 495 / 2,344 (21.12%) |
| Linear virtual hits / completed | 414 / 2,638 (15.69%) | 429 / 2,257 (19.01%) |
| Head-on virtual hits / completed | 383 / 2,638 (14.52%) | 424 / 2,257 (18.79%) |

H1 is contradicted. Instead of falling, both modelled guns and Retcon's actual hit rate rose by about 3.3–4.3 points. AIS damage taken per turn rose 36%. Wall collisions rose fivefold.

A plausible explanation is unverified: the +5 corridor penalty frequently outweighs the existing wall term (at most 3). That pulls the selected point toward walls, where collisions stop the bot. A stopped bot sits on both the head-on and the linear prediction. The penalty also re-selects destinations every turn as rays change, which may reduce lateral speed. The aim reconstruction itself was only unit-tested; it was not validated against Retcon's actual bullet headings in play. These explanations are hypotheses for any future candidate; they were not tested here.

### Stage B: official samples, 10 fixed rounds per arm (pass)

| Opponent | Baseline share / wins / walls | Candidate share / wins / walls |
| --- | --- | --- |
| Target | 99.89% / 10 / 7 | 100.00% / 10 / 91 |
| SpinBot | 95.23% / 10 / 98 | 99.03% / 10 / 5 |
| Walls | 96.94% / 10 / 3 | 96.95% / 10 / 3 |

All candidate sample shares were within the 5-point guard, with zero skipped turns. Sample wall counts moved in both directions and are descriptive only.

### Stage C: not run

The registered rule runs it only after Stage A retains.

### Limitations

These are unpaired, unseeded starts. There are three game blocks per arm. Retcon adapts within each game. The sample stage used one game per opponent per arm. Retcon counters are offline diagnostics, never AIS inputs. No strength claim follows in either direction beyond this protocol. The registered thresholds were applied unchanged.

PR: [#32](https://github.com/stephanelee/robocode/pull/32). Evidence: [raw evidence and patch](../../benchmarks/aim-corridor/README.md).
