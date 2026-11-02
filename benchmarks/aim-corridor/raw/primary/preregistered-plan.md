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

Not yet measured.
