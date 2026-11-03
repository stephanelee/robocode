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

Not yet measured.
