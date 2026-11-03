# Two-way nearest-safe wall smoothing (defect repair)

Status when this file is first committed: **preregistered, not yet measured.** The hypothesis, budget, metrics and decision rule are fixed before any registered match. Results are appended under the final heading in a later commit. Nothing above that heading is edited after the run starts.

## Baseline and candidate

- **Accepted baseline:** master `241156f7d6b78c4ca2365859884b1486c9f41194`. Movement is identical to accepted `204ad4c9`. The only change since `116c47c` is the default-off recorder repair. The runnable JAR SHA256 is `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883`.
- **Candidate:** the commit containing this file, together with its source, tests and Java 17 JAR. The workflow records the exact SHA and JAR hash.
- **Opponents:**
  - Retcon 0.2.0, unchanged, at commit `28981c7a469ac064194b41001cd7183ec5ecf189`, with Bot API 1.3.1 and JDK 25.
  - The pinned Tank Royale 1.4.0 Runner and the official 1.4.0 Target, SpinBot and Walls bots.
- **Build and runner:** Java 17 build, on exactly `ubuntu-26.04`.

## Defect (facts)

The accepted `AISurvivor.wallSmooth` works as follows:
1. It starts from the requested heading.
2. It tries at most 36 headings, rotating in one direction only, which covers half a circle.
3. Each heading is accepted if the point 80 units ahead lies inside the 80-unit wall envelope.
4. If none qualifies, it returns the last heading tried. That heading is unchecked.

Near a wall, the safe arc can lie entirely in the unsearched half. The bot then steers into the wall, collides at speed 0, and collides again every two turns while turning slowly.

These events were traced in development games, which are not evidence. Recorded examples:
- At (275, 582), requested 330°: the safe arc is 231°–309°, but the search returned 150°.
- At (782, 402), requested 243°: the search returned 62°.

Registered evidence shows the defect in the accepted bot itself:
- The [v2 run](aim-corridor-v2.md) baseline arm recorded 323 wall collisions in one 20-round Retcon block, against 22 and 10 in the others.
- The same arm recorded 233 collisions in 10 Target rounds.

## Candidate change

This is a single gameplay change. `wallSmooth` delegates to a pure `CombatMath.nearestSafeHeading`:
- It searches 5° steps in **both** directions over the full circle. The smallest deviation wins, and positive rotation wins ties.
- It returns the first heading whose 80-unit endpoint lies strictly inside the envelope.
- If no heading qualifies, as in a deep corner, it heads toward the arena centre.

An already-safe requested heading is returned unchanged. The aim corridor is **not** included. Targeting, firing, radar, the DQN, rewards and the rest of movement are untouched.

This differs from the rejected [cycle 2](cycle-2.md), which searched the full circle in one direction only. Cycle 2 picks the far-side safe heading, for example 235° instead of 305° in the first example above.

### Disclosed development observations (exploratory, not evidence)

These come from six 5-round local Retcon games per arm, on development seeds 9301–9306 only:

| Arm | Mean share | Walls per 1,000 turns | Retcon hit rate |
| --- | ---: | ---: | ---: |
| Baseline | 18.9% | 1.38 | 15.7% |
| Wall fix | 20.7% | 0.18 | 16.5% |

Variance between games is very large. These results did not set any threshold below.

## Hypotheses (not facts)

- **H1:** wall collisions fall, especially multi-hit pinning episodes.
- **H2:** removing wall pinning does not reduce, and may improve, Retcon score share, wins and damage taken.
- **H3:** no regression against the sample bots or in online-training play.

A strength gain of 5 pp is **not** expected. Cycle 2's related fix measured +1.39 pp.

## Gate category, chosen prospectively

This candidate repairs a deterministic, unit-tested defect in which the bot returns an unchecked heading into a wall. It is therefore registered under the existing **correctness non-regression** category. That category was previously approved for the scan-lag repair, and the [strategy](../retcon-strategy.md) states that other changes do not automatically inherit it. This assignment therefore **requires independent reviewer acceptance at PR review**.

The run also records the tactical (+5 pp) classification from the same games:
- If the reviewer does not accept the category, the tactical classification decides.
- Any retention under the correctness category is labelled **"correctness retained, no strength gain established"** unless the tactical gate also passes.

## Budget and procedure (single run, fixed in advance)

One `Wall search experiment` workflow job runs:

```text
tools/aim_corridor_experiment.py --plan-path docs/improvements/wall-search.md \
  --seeds 20261221,20261222,20261223 --sample-seed 20261224 --primary-category correctness
```

There is no early stopping, no favourable reruns and no seed replacement. An infrastructure failure marks the run **invalid**. Partial evidence is then preserved, and any retry is disclosed.

**Stage A: primary (120 Retcon rounds).**
- `compare.py --category correctness --policy fixed-minrisk --purpose primary --rounds 20 --require-retcon-stats`.
- Seeds 20261221, 20261222 and 20261223 (new; earlier experiment seeds are not reused).
- Arm order alternates by block.
- `comparison.json` reports both the correctness decision and `tacticalStrengthGatePassed`.

**Stage B: sample regression (60 rounds).**
- Target, SpinBot and Walls, fixed minimum-risk, 10 rounds per arm each, seed label 20261224.
- Always run.

**Stage C: mandatory online-training veto (120 Retcon rounds).**
- Run only if Stage A passes.
- `compare.py --category correctness --policy online-train`, same seeds.
- Fresh models for each arm and seed, in the real default `train` mode.

The maximum budget is 300 scored rounds.

## Acceptance criteria (all must hold)

**Stage A, correctness thresholds** (unchanged; applied to unrounded values):
1. Mean score-share change **≥ 0**.
2. At least **2 of 3** blocks nonnegative.
3. No block loses more than **5 pp**.
4. Candidate wins no more than **3** below baseline.
5. **Zero** candidate skipped turns.
6. Candidate walls per 1,000 turns **≤ 1.2×** baseline.

**Stage B:** for each sample bot:
- candidate share ≥ baseline share − **5 pp**;
- all 10 rounds per arm complete;
- zero candidate skipped turns.

**Stage C:** the same six correctness thresholds, applied to the online-training games.

**Decision:**
- If A, B and C all pass, the result is **retain (correctness)**, conditional on reviewer acceptance of the category.
- Otherwise it is **reject**. On rejection:
  - restore the baseline gameplay source and JAR exactly;
  - keep the patch, the protocol and the raw evidence.

A retained correctness repair is not a claim of strength against Retcon or in general.

## Mechanism observables (descriptive)

For each arm:
- wall collisions per 1,000 turns, and per-block wall counts;
- AISurvivor damage taken and dealt per 1,000 turns, hit rate and shots;
- Retcon's actual and virtual (linear and head-on) hit rates on AISurvivor, from offline native exports only.

## Results

Not yet measured.
