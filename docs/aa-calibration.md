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

Everything above this heading is byte-identical to the registered plan at `79359f27dc6fd011d472030bdd916c565d564c26`. The script stored the plan blob.

There was one run: [41548798991](https://github.com/stephanelee/robocode/actions/runs/41548798991), job 125630877789, artifact 13189987653, zip SHA256 `f592a70f57204b4e0839f7e17f4590af3781e05e58bf48648d4ec21cc8ea9bd5`.
- It ran on Ubuntu 26.04.1 LTS. The Java 17 JAR parity check passed, and matches ran on Temurin 25.0.5+8.
- All 400 rounds completed, with zero skipped turns.
- The bundle hashes matched before every battle.

### Preregistered statistics

| Statistic | Value |
| --- | ---: |
| Mean share, arm a / arm b | 17.696% / 19.410% |
| Pooled mean / SD of the 20 block shares | 18.553% / 4.020 pp |
| Mean paired delta (a − b) | −1.714 pp (SE 1.767) |
| **Block delta SD** | **5.586 pp** |
| Delta range | −12.339 to +8.543 pp |
| **Blocks per arm for 80% power at +5 pp (one-sided 5%)** | **8** (160 rounds per arm) |
| Current tactical score checks: simulated pass rate at true 0 / +5 / +10 pp | 5.9% / 49.3% / 93.5% |
| Retcon actual hit rate on AIS, a / b | 18.63% / 18.59% |
| Round wins, a / b | 12 / 18 |

Per block, as `a`/`b` share and wins:

| Seed | Order | a | b | a − b |
| ---: | --- | ---: | ---: | ---: |
| 20261301 | a, b | 19.651% / 2 | 19.405% / 1 | +0.245 |
| 20261302 | b, a | 13.943% / 0 | 26.282% / 4 | −12.339 |
| 20261303 | a, b | 12.380% / 0 | 14.525% / 1 | −2.145 |
| 20261304 | b, a | 17.478% / 1 | 19.283% / 2 | −1.805 |
| 20261305 | a, b | 19.009% / 2 | 10.466% / 0 | +8.543 |
| 20261306 | b, a | 16.760% / 1 | 19.591% / 2 | −2.831 |
| 20261307 | a, b | 20.610% / 1 | 22.422% / 2 | −1.812 |
| 20261308 | b, a | 21.793% / 3 | 21.200% / 2 | +0.593 |
| 20261309 | a, b | 22.797% / 2 | 20.601% / 2 | +2.196 |
| 20261310 | b, a | 12.540% / 0 | 20.327% / 2 | −7.786 |

**Proposed budget (rule fixed in the plan above):** 8 blocks per arm, already even, within the 30-block cap. This replaces the 3-block design for the next registered tactical experiment, if the reviewer agrees.

**Proposed decision rule:** mean block delta ≥ +5 pp, **and** the one-sided 95% lower bound (mean − 1.6449 × SE) > 0.

### Additional observations (exploratory; not preregistered)

These were computed after the run from the same raw files. They are disclosed for review and should be treated as hypotheses about the existing guards, not as registered results.

- **Wall collisions are extremely noisy between identical bots.** Arm a recorded 235 collisions (1.788 per 1,000 turns) and arm b recorded 1,216 (8.910 per 1,000 turns), a 5.0× ratio. Single blocks ranged from 0 to 360, driven by rare multi-hit pinning episodes.
- **I applied the existing 3-block rules to every 3-block subset of these A/A blocks**, in both directions: 240 dependent comparisons, none of which changes behaviour.
  - The 1.2× wall-rate guard failed in **48.3%** of them.
  - The ±3 wins guard failed in **13.8%**.
  - The full **correctness** non-regression gate passed only **10.8%**, so it rejects a true no-op about 89% of the time.
  - The full tactical gate passed 0.4%.
- **Implication.** Rejections under these guards, including the [wall-search](https://github.com/stephanelee/robocode/pull/35) correctness rejection, are weak evidence against a candidate. They are not evidence of harm. The earlier decisions stand under their registered rules; this calibration does not reverse them.
- **Suggestion for review.** Replace ratio-based wall and wins guards with noise-calibrated or damage-based guards before the next correctness-category experiment.

### Limitations

- Fixed minimum-risk policy only. Online-training noise and candidate-specific variance are not measured.
- Rounds within a block share an adaptive Retcon, and the normal/independent model is approximate.
- The exploratory subsets overlap and are not independent.

Evidence: [raw calibration data](../benchmarks/aa-calibration/README.md).
