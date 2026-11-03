# A/A noise calibration against Retcon

Status when this file is first committed: **preregistered, not yet measured.** The procedure, the statistics to be computed and how they will be used are fixed before the run. Results are appended under the final heading in a later commit, and nothing above that heading is edited after the run starts.

## Why

Three recent experiments ran the same accepted movement and targeting code as their baseline arm against Retcon. The [aim-corridor](improvements/aim-corridor.md), [aim-corridor v2](https://github.com/stephanelee/robocode/pull/34) and [wall-search](https://github.com/stephanelee/robocode/pull/35) baselines measured mean score shares of 15.70%, 18.04% and 21.63%. Individual 20-round block deltas ranged from −12.39 to +1.35 pp. The [review response](retcon-review-response.md) estimated that the 3-block tactical rule passes a true +5 pp effect only about 50% of the time. That estimate used an SD taken from six old cold-DQN games, not from the fixed-policy arm.

Each 180-round experiment took about 2–3 minutes of runner time, so a much larger budget is affordable. This calibration measures fixed-policy noise directly, so that the next experiment's budget can be **registered prospectively** from data rather than guessed.

## What is run

- **Bot:** master `241156f7d6b78c4ca2365859884b1486c9f41194`, JAR SHA256 `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883`, used as **both** arms (`a` and `b`). Both arms are immutable copies of the same bundle, with hashes verified before every battle. No gameplay source or JAR change is included.
- **Opponent:**
  - Retcon 0.2.0 at `28981c7a469ac064194b41001cd7183ec5ecf189`, with API 1.3.1, launched under JDK 25.
  - Tank Royale 1.4.0 Runner.
  - Native Retcon stats are required and used as offline diagnostics only.
- **Runner and build:** exactly `ubuntu-26.04`; the Java 17 build/JAR parity check passes before the run.
- **Design:**
  - 10 blocks, with seed labels 20261301–20261310.
  - Each block plays one 20-round fixed minimum-risk game per arm, so 400 scored rounds in total.
  - Arm order alternates by block (`a`/`b`, `b`/`a`, …).
  - Fixed mode uses no bot RNG, so seeds only label independent games.
- **Command:** `tools/aa_calibration.py --blocks 10 --rounds 20 --seed-base 20261301`.
- **Procedure:** one run, no early stopping, no reruns and no block replacement. An infrastructure failure marks the run invalid; partial evidence is preserved and any retry is disclosed.

## Statistics (computed by the script, unrounded)

- **Per-block score share** for each arm; the pooled mean; the pooled SD of the 20 block shares.
- **Paired block deltas** `a − b`: their mean, sample SD (`deltaSD`), standard error and range. Both arms are identical, so the expected mean is 0. `deltaSD` estimates the noise of a 20-round block comparison.
- **Required blocks per arm** for 80% power on a one-sided 5% test of the mean block delta at a true +5 pp: `ceil(((1.6449 + 0.8416) × deltaSD / 0.05)²)`.
- **Simulated pass rate of the current tactical score checks** at true effects of 0, +5 and +10 pp, with 200,000 seeded draws:
  - the checks are 3 blocks, mean ≥ +5 pp, ≥ 2 blocks improved, and no block below −5 pp;
  - block deltas are modelled as independent normals with SD `deltaSD`.
- **Descriptive only:** Retcon's actual hits and shots on AISurvivor, and wins, for each arm.

## How the result is used (fixed now)

- **This is not a gameplay experiment.** It retains or rejects nothing, and its games are never reused as an arm of a future comparison.
- **Proposal only.** The output is a proposal for the next registered experiment's Retcon budget:
  - the computed `blocksPerArmFor80PctPowerAt5pp`, rounded up to an even number;
  - with a cap of 30 blocks (600 rounds) per arm;
  - with a minimum of the existing 3 blocks.
- **Proposed decision rule** for a future tactical experiment, to be adopted only if the reviewer agrees:
  - the mean block delta is at least +5 pp, **and** its one-sided 95% lower bound is above 0;
  - the existing win, skipped-turn and wall-rate guards still apply.
- **Existing rules unchanged.** The existing `compare.py` thresholds and earlier experiments' decisions are not changed or reinterpreted by this calibration.
- **Limitations, stated in advance:**
  - The normal and independence model is an approximation: Retcon adapts within a game, and rounds within a block are correlated.
  - Online-training noise is not measured here.
  - A different candidate might have a different variance.

## Results

Not yet measured.
