# Coherent default melee (minimum risk) with two-way wall search

Status when this file is first committed: **preregistered, not yet measured.** Results are appended under the final heading in a later commit. Nothing above that heading is edited after the run starts.

## Arms

- **Baseline:** master `e4e4c94`, the confirmed default product. JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`.
- **Candidate:** the commit containing this file on `claude/melee-coherent`. It builds on #48 (replay harness) and #47 (warm-up and telemetry), neither of which changes decisions. The workflow records the exact SHA and JAR hash.
- **Opponents:**
  - The official 1.4.0 Corners, Crazy, RamFire, TrackFire, VelociBot, SpinBot and Walls, as a seven-opponent melee.
  - Retcon 0.2.0 at `28981c7a469ac064194b41001cd7183ec5ecf189` on JDK 25.
  - The official Target, SpinBot and Walls in 1v1.
- **Infrastructure:** Tank Royale 1.4.0 Runner; Java 17 build; exactly `ubuntu-26.04`.

## Change (two decision changes, registered as one candidate)

1. **Coherent melee policy.** In default `train` mode, the DQN's choice is forced to action 0 whenever more than one enemy is alive. The bot then runs minimum-risk movement every melee turn, instead of switching among six strategies turn by turn. It uses the same mechanism as the duel controller in 1v1 (#42).
   - The DQN still records the executed action, with its existing forced-transition handling.
   - `learned`, `eval` and `fixed` modes are unchanged.
2. **Two-way wall search.** `wallSmooth` searches 5° steps in both directions over the full circle, taking the smallest deviation. If no direction is safe, it heads toward the arena centre. Previously it searched only half a circle in one direction and could return an unchecked heading into the wall.
   - Default 1v1 duel movement does not use `wallSmooth` (`DuelSurf` has its own wall handling).

The replay harness confirms the scope. The pre-change melee golden first diverges at round 1, turn 3, and the Retcon and Walls 1v1 goldens still replay identically. The melee golden is re-recorded, and the old one is kept under `benchmarks/replay-goldens/superseded/`. `MasterParityTest` also shows that master `e4e4c94`'s unmodified code reproduces the Retcon and Walls 1v1 goldens and the pre-change melee golden bit for bit, and that it diverges from the candidate's melee golden at that same point.

**Why Stage B uses 16 blocks.** The [#48 parity run](replay-harness-parity.md) showed that default-mode Retcon block deltas have an SD of about 7.1 pp. At that SD, an 8-block −5 pp margin passes decision-identical bots only about half the time. Under the same noise model, 16 blocks (t(0.95, 15) = 1.753) pass identical bots about 85% of the time. That choice was made before this run, using that earlier run's data.

## Evidence that motivated it (development, local, not evidence)

**Observations.**
- The audit found a 35-wall-hit melee probe.
- The DQN switches strategies every turn in melee.

**Development sweep.** Three variants were compared on development seeds 9950–9955: 6 games × 3 rounds against the seven samples, on Ubuntu 24.04.

| Variant | Mean share | Games ranked 1st | Round wins | Wall hits |
| --- | ---: | ---: | ---: | ---: |
| Current bot | 24.5% | 4/6 | 13 | 144 |
| Coherent policy only | 28.3% | 6/6 | 16 | 21 |
| Coherent policy + wall search (this candidate) | 30.7% | 6/6 | 18 | 13 |

These figures selected the candidate. They did not set any threshold. The registered seeds are new.

## Budget and procedure (single run, fixed in advance)

The run is one `Coherent melee experiment` workflow job:

```
tools/melee_experiment.py --melee-blocks 10 --melee-rounds 10 --retcon-blocks 16 --retcon-rounds 20 --seed-base 20261901
```

Every stage uses default `train` mode with fresh models.

- **Stage A, primary (200 melee rounds):** 10 blocks. Each block is one 10-round game per arm, eight bots per game, seeds 20261901–20261910, with arm order alternating.
- **Stage B (640 Retcon rounds):** 16 blocks of 20 rounds per arm, seeds 20262001–20262016.
- **Stage C (60 rounds):** Target, SpinBot and Walls, 10 rounds per arm each, seed 20262101.

The total is 900 scored rounds. There is no early stopping, no favourable rerun and no replacement of blocks. An infrastructure failure makes the run invalid; its partial evidence is kept, and any retry is disclosed.

## Acceptance (all must hold; unrounded)

**Stage A, superiority.** Let the paired block delta be candidate melee score share minus baseline melee score share.
1. All 10 blocks complete.
2. The mean paired block delta is **≥ +2 pp**.
3. The Student-t one-sided 95% lower bound, `mean − t(0.95, 9) × SD/√10` with t = 1.833113, is **> 0**.
4. Candidate first places are **≥** the baseline's.
5. Candidate skipped turns are **≤** the baseline's.
6. Candidate walls per 1,000 turns are **≤ baseline + 10**.

**Stage B, Retcon non-inferiority.**
- The Student-t lower bound of the mean block delta is **≥ −5 pp**.
- Candidate skipped turns are ≤ baseline.
- Candidate wall rate is ≤ baseline + 10 per 1,000 turns.

**Stage C.** For each sample bot:
- candidate share ≥ baseline − 5 pp;
- all 10 rounds completed;
- zero candidate skipped turns.

**Decision:**
- **Retain** only if A, B and C all pass. That means the candidate is measured as better than the current bot in melee, and not worse against Retcon or the samples. Only then will a PR be opened.
- **Otherwise reject:** no PR. The branch, protocol and evidence are kept for the record.

## Limitations, stated in advance

- **No melee noise calibration.** Melee noise has not been calibrated by an A/A run; the budget comes from the development spread.
- **Wide melee gap.** The development effect (about +6 pp) is under 2× its own between-game SD.
- **Bundled attribution.** The two changes are bundled, so the result is not attributed to either part.
- **Round-1 skips.** Round-1 melee skips are an environmental start-up effect (see [warm-up reliability](warmup-reliability.md)). The skip guard compares arms, and both are exposed equally.

## Results

The text above this heading is the registered plan, unchanged since `c4b2cae035e34605b8e7bd48dc62956b70a1d388`, which is also the measured candidate. The driver stored that blob byte for byte.

**The run:**
- **Single registered run:** [41645248808](https://github.com/stephanelee/robocode/actions/runs/41645248808) (job 125948759531, artifact 13237120936, zip SHA256 `db49537c4b5ceacb69eccde26b17a45507c91863c01c9fbcaee50b930221b603`), every step green, no reruns.
- **Environment:** Ubuntu 26.04.1. Java 17 JAR parity passed; Temurin 25.0.5+8 for matches.
- **Candidate JAR:** `351a006829dfeb0a43850f0d54de7a1442296d19518d68ee8b17d8f9e32c16a1`.

**Registered decision: reject.** Stage A (melee superiority) and Stage B (Retcon non-inferiority) failed; Stage C passed. Per the plan, no PR is opened. The branch, protocol and evidence are kept. Master is unchanged.

### Stage A: melee, 100 rounds per arm (fail)

| Melee, 8 bots | Master | Candidate |
| --- | ---: | ---: |
| Mean block score share | 27.994% | 28.514% |
| First places | 86 | 87 |
| Skipped turns | 7 | 6 |
| Wall collisions | 1,536 | 65 |
| Walls per 1,000 turns | 13.45 | 0.56 |
| Damage taken | 4,679 | 4,426 |
| Hits / shots | 3,255 / 6,047 | 3,386 / 6,222 |

| Seed | Master | Candidate | Change (pp) |
| --- | ---: | ---: | ---: |
| 20261901 | 25.10 | 26.18 | +1.08 |
| 20261902 | 31.88 | 26.90 | −4.98 |
| 20261903 | 31.62 | 27.10 | −4.52 |
| 20261904 | 25.82 | 31.04 | +5.22 |
| 20261905 | 27.97 | 30.83 | +2.86 |
| 20261906 | 22.05 | 32.15 | +10.11 |
| 20261907 | 28.14 | 24.65 | −3.50 |
| 20261908 | 29.50 | 29.16 | −0.34 |
| 20261909 | 28.95 | 29.76 | +0.81 |
| 20261910 | 28.91 | 27.36 | −1.55 |

- **Mean delta:** +0.520 pp, SE 1.478 pp. **Fails** the ≥ +2 pp rule.
- **Student-t lower bound:** −2.190 pp. **Fails** the > 0 rule.
- **Guards:** first places, skipped turns and wall rate all pass.

### Stage B: Retcon, 320 rounds per arm (fail)

| Retcon, default product | Master | Candidate |
| --- | ---: | ---: |
| Mean block score share | 87.840% | 85.476% |
| Wins | 301 | 297 |
| Skipped turns | 0 | 0 |
| Walls per 1,000 turns | 0.095 | 0.061 |
| AIS hits / shots | 1,290 / 18,522 | 1,151 / 18,856 |
| Retcon actual hit rate on AIS | 2.10% | 2.36% |

- **Mean delta:** −2.364 pp, SE 1.755 pp (16 blocks).
- **Student-t lower bound:** −5.441 pp. **Fails** the ≥ −5 pp margin.
- **Guards:** skipped turns and wall rate pass.

In 1v1 the melee gate never applies (it needs more than one enemy). The two-way wall search runs only in legacy movement, which 1v1 uses just for the turns before the opponent is first scanned. After that, `DuelSurf` moves the bot. So the 1v1 change is confined to those opening turns, though they can shift every later state of the round. A gap this size was also seen in [replay-harness parity](replay-harness-parity.md) (−2.15 pp for decision-identical code), so game variation is a plausible explanation. That is an interpretation, not a registered finding; the registered failure stands and is not rerun.

### Stage C: samples (pass)

| Sample | Master | Candidate |
| --- | ---: | ---: |
| Target | 100.000% | 100.000% |
| SpinBot | 99.875% | 98.048% |
| Walls | 100.000% | 100.000% |

All 10 rounds completed for each, with zero candidate skipped turns.

### What was learned (not a gameplay claim)

- **The wall defect is fixed in melee.** Wall collisions fell 96% (1,536 → 65). This did not translate into a measurable score gain at this budget.
- **Exploratory only:** the two blocks with the most baseline wall hits (428 and 478) gave the candidate +1.08 and +10.11 pp, but other blocks with few baseline wall hits went either way. The melee block SD (4.7 pp) is larger than the effect sought, so a +2 pp gain would need far more than 10 blocks to establish.
- Any future melee candidate needs its own registered plan, with a melee A/A calibration first.

Evidence: [raw evidence](../../benchmarks/coherent-melee/README.md).
