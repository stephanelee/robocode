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

Not yet measured.
