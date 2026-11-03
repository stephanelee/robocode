# Replay-harness restructure: non-inferiority against master

Status when this file is first committed: **preregistered, not yet measured.** Results are appended under the final heading in a later commit. Nothing above that heading is edited after the run starts.

## What is compared

| Arm | Code | JAR | Notes |
| --- | --- | --- | --- |
| **Baseline** | master `e4e4c94`, the confirmed default product ([#46](final-default-confirmation-redo.md)) | SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6` | |
| **Candidate** | the commit containing this file on `claude/replay-harness` (PR #48, stacked on #47) | recorded by the workflow, with the exact SHA | Adds #47's `Warmup` and latency telemetry. Moves all decision logic verbatim into `SurvivorBrain` behind `BotPort`. Adds `GameTrace`/`TraceReplay`. |

Neither change is intended to alter any decision.

**Environment:**
- **Opponents:** Retcon 0.2.0 at `28981c7a469ac064194b41001cd7183ec5ecf189` on JDK 25, plus the official 1.4.0 Target, SpinBot and Walls.
- **Runner:** Tank Royale 1.4.0 with a 100 ms turn timeout.
- **Build and platform:** a Java 17 build, on exactly `ubuntu-26.04`.

## Why a battle benchmark

The extraction is backed by:
- a verbatim move;
- all existing tests, including the 288 exact-bit `aimGF` goldens;
- three live golden traces that replay identically.

The pre-extraction monolith could not emit traces, though, so its parity with the extracted brain has not been compared directly. This benchmark checks empirically that the shipped default product has not lost strength.

## Procedure (single run, fixed in advance)

Everything runs in one `Replay harness parity` workflow job:

```text
tools/block_experiment.py --policy train --primary-rule noninferiority --bound t --skip-online \
  --blocks 8 --rounds 20 --seed-base 20261801 --sample-seed 20261809
```

- **Stage A (320 Retcon rounds):** the default product, `train` mode with a fresh model per arm and block, 8 blocks of 20 rounds per arm. Seeds are 20261801–20261808, and arm order alternates by block.
- **Stage B (60 rounds):** Target, SpinBot and Walls, 10 default-mode rounds per arm each, seed label 20261809. Always run.
- **No online stage.** Stage A already measures the default online-training product.

**Run rules:**
- No early stopping, no favourable rerun, and no replacement games.
- An infrastructure failure marks the run invalid. Its partial evidence is preserved, and any retry is disclosed.

## Acceptance (all must hold; unrounded values)

**Stage A (non-inferiority):**
1. All 8 blocks complete.
2. The Student-t one-sided 95% lower bound of the mean paired block delta (candidate − baseline score share), `mean − 1.894579 × SD/√8`, is **≥ −5 pp**.
3. The candidate's total skipped turns are **≤** the baseline's.
4. The candidate's wall collisions per 1,000 turns are **≤ baseline + 10**.

**Stage B:** for each sample bot:
- candidate share ≥ baseline − **5 pp**;
- all 10 rounds per arm are complete;
- the candidate has zero skipped turns.

**Decision:**
- **Pass:** strength is non-inferior within 5 pp. This supports merging #47 and #48. It is not a strength gain.
- **Fail:** #48 is not merged as is. The difference must be investigated using the replay harness and the latency telemetry.

## Limitations, stated in advance

- **Margin:** at the A/A noise of 5.59 pp per block (SE ≈ 2.0 pp over 8 blocks), the rule passes only if the observed mean is above roughly −1.2 pp. A true loss of 5 pp would therefore pass only about 3% of the time, but a true loss of 1–2 pp cannot be ruled out.
- **Unpaired starts:** starts are unpaired. Retcon adapts within each game.

## Results

The text above this heading is the registered plan. It is unchanged since `22dd537e52bf4234f991abb37bb452eb62d9e1bf`, which is also the measured candidate, and the driver stored that blob byte for byte.

**The run:**
- **Single registered run:** [41642071703](https://github.com/stephanelee/robocode/actions/runs/41642071703) (job 125937824276, artifact 13234135776, zip SHA256 `e3f0114d6396a34c0703f5a6e9f47ed8d1a11e52b8bebaa9a3c4b01f0c8d9f30`), with every step green and no reruns.
- **Environment:** Ubuntu 26.04.1. Java 17 JAR parity passed; Temurin 25.0.5+8 for matches.
- **Candidate JAR:** `28e6322721cea1e4430dccc9b34031ead60513372f0e94bbeaeec1d629e129e9`.

**Registered decision: fail (non-inferiority not shown).** Stage A missed the margin; Stage B passed. Under this protocol, #48 is not merged on this evidence alone.

| Retcon, default product, 160 rounds per arm | Master | #48 |
| --- | ---: | ---: |
| Mean block score share | 87.177% | 85.030% |
| Wins | 149 | 146 |
| Skipped turns | 0 | 0 |
| Walls per 1,000 turns | 0.101 | 0.028 |
| AIS hits / shots | 614 / 9,416 | 579 / 9,554 |
| Retcon actual hit rate on AIS | 1.97% | 2.28% |

Per-block (master → #48):

| Seed | Master | #48 | Change |
| --- | ---: | ---: | ---: |
| 20261801 | 87.01 | 79.77 | −7.24 |
| 20261802 | 86.65 | 87.71 | +1.06 |
| 20261803 | 87.36 | 84.75 | −2.61 |
| 20261804 | 87.03 | 95.16 | +8.12 |
| 20261805 | 90.42 | 84.82 | −5.60 |
| 20261806 | 83.91 | 75.07 | −8.84 |
| 20261807 | 81.59 | 89.35 | +7.75 |
| 20261808 | 93.44 | 83.61 | −9.83 |

- **Mean delta:** −2.147 pp, SE 2.520 pp.
- **Student-t lower bound:** −6.921 pp, which **fails** the ≥ −5 pp margin.
- **Other Stage A guards:**
  - skipped turns not more than master: pass;
  - wall rate: pass.

**Stage B (pass):**

| Sample | Master | #48 |
| --- | ---: | ---: |
| Target | 100.000% | 100.000% |
| SpinBot | 94.412% | 98.874% |
| Walls | 100.000% | 100.000% |

All three had zero skipped turns.

### Investigation required by the protocol (after the run)

The protocol required investigating a failure with the replay harness. To do that I added `MasterParityTest`. It drives an **unmodified copy of master `e4e4c94`'s `AISurvivor`** (`src/test/java/MasterAISurvivor.java`; only the class name and constructor differ). Mockito stubs every final Bot API getter from the recorded state and captures every intent setter. The copy is replayed through all three golden traces recorded by the extracted brain.

**Result: master issues bit-identical commands on every batch.**

| Golden trace | Command batches |
| --- | ---: |
| Retcon 1v1 | 3,044 |
| Walls 1v1 | 939 |
| 8-bot melee, including seeded DQN Boltzmann selection | 3,007 |

**What this means:**
- **Decisions are identical.** The extraction and #47's additions do not change any decision in these 6,990 recorded states.
- **The score gap is consistent with game-to-game variation** (z ≈ −0.85; the default-mode block-delta SD was 7.13 pp, above the 5.59 pp fixed-mode calibration). The fixture proof covers decisions in the visited states only. It does not cover runtime or timing behaviour (for example, the added warm-up and telemetry), nor states the goldens never visited. A behavioural cause of the gap is therefore not ruled out, only unsupported by the decision evidence.
- **The margin was underpowered.** At that SD, this 8-block rule passes two decision-identical bots only about half the time.
- **#48 is on hold.** The registered failure stands unchanged. Trace parity does **not** waive the registered gate, and no retrospective change to the margin is made. Before any merge, a separately reviewed, prospectively registered protocol must pass.
- **Coverage limit.** A trace proves identity only for the states it visited.

Evidence: [raw evidence](../../benchmarks/replay-harness-parity/README.md).

## Review follow-up (2026-11-03)

The independent PR #48 review confirmed the registered failure and two gaps in the replay guarantees. Both gaps are now fixed, and the conclusions are stated conditionally.

1. **Comparison was not bit-exact.** Gson's `JsonArray.equals` treats `0.0` and `-0.0` as equal. `TraceReplay.sameCalls` now compares numeric arguments with `Double.doubleToLongBits`, and `MasterParityTest` uses the same comparator. A test now changes a recorded `0.0` to `-0.0` in both the Retcon and Walls goldens and requires the mismatch to be reported.
2. **Incomplete traces counted as success.** Replay now requires at least one complete game:
   - at least one command batch;
   - equal numbers of started and ended rounds;
   - a `GameEndedEvent`.

   Tests show that a header-only trace and a 6,000-line truncation both fail. Each golden's exact coverage is pinned: Retcon 3,044 batches / 3 rounds, melee 3,007 / 3, Walls 939 / 3.

After these fixes, master `e4e4c94`'s unmodified code still reproduces all three complete goldens bit for bit under the stricter comparator.

The registered non-inferiority **failure stands**. This PR is not to be merged on the fixture proof alone. Merging it would need a new, prospectively registered parity check with adequate power, disclosed as a second attempt after this failure.

## Separately authorized fresh confirmation

The user subsequently authorized a new, prospectively registered second attempt against accepted PR47. Its [1,260-round protocol and results](replay-harness-parity-redo.md) pass all unchanged noninferiority guards: mean difference −0.00090508 pp, t lower bound −2.30542198 pp above the −5 pp margin, zero skips and all sample guards. [Fresh raw evidence and66models](../../benchmarks/replay-harness-parity-redo/README.md) are separate; this original failure is not rewritten or pooled. The first archive contains no model checkpoints, a historical provenance limitation. The fresh result supports bounded parity, not a strength gain; merge still requires independent architectural review and final protected CI.

