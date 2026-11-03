# PR48 fresh parity confirmation: user-authorized second attempt

Registered before outcomes. The user explicitly requested a new parity benchmark after PR48's first registered check failed. That original 380-round result remains a failure, with its raw evidence and thresholds unchanged. No old games are pooled with this attempt, and fixture parity does not waive the battle gate.

## Frozen arms and budget

- Baseline: accepted master after PR47, `cfdb508b0492df88667696f9bc0de9c61751a273`, committed JAR SHA256 `21c5c3d2cb4007e3e0603556af313df7f5717732e384f039b005e9e72b2daecd`. Using accepted PR47 on both sides isolates the PR48 extraction from warm-up/telemetry; this differs prospectively from the first attempt's pre-PR47 baseline.
- Candidate: PR48 `f224bc7c4bb0ac854874e5832c7df9e8e9ae4e24`, committed JAR SHA256 `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800`. Later commits may add only this registration, power evidence and the one-shot workflow. Production Java, assets and acquisition/decision helpers must remain byte-identical to that frozen tree. Tracing is disabled.
- Stage A: **30 new blocks, 20 Retcon rounds per arm per block: 1,200 rounds**, default TRAIN, independent initially missing checkpoint for every arm/game, alternating baseline/candidate execution order. Fresh AIS seeds `20262801` through `20262830`; each seed labels the two arms, not paired arena positions.
- Stage B: **60 rounds**: Target, SpinBot and Walls, ten default-TRAIN rounds per arm and opponent, seed `20262899`, private cold models. Always run after Stage A, regardless of its score decision. No additional online stage: TRAIN already is the shipped online-training product.
- Total: **1,260 physical rounds, 66 game starts, 66 private checkpoint endpoints**. No new opponent or gameplay code. Reuse unchanged `tools/block_experiment.py --policy train --primary-rule noninferiority --bound t --skip-online --blocks 30 --rounds 20 --seed-base 20262801 --sample-seed 20262899`.
- Pinned Retcon0.2.0 source `28981c7a469ac064194b41001cd7183ec5ecf189`; official Tank Royale1.4.0 runner/assets and 800x600 1v1 arena, 100000-microsecond timeout. Exactly Ubuntu26.04; Java17 clean build/test/committed-JAR parity, then JDK25 matches. Isolated embedded ports; the GUI/server on7654 is untouched.

## Why thirty blocks

The spent first attempt's eight default-TRAIN block differences had sample SD about7.13 percentage points, larger than the older fixed-policy calibration. This is planning evidence, not fresh confirmation. Thirty blocks replace the original eight while retaining the same five-point margin and one-sided95% Student-t rule (df29, quantile `1.699127` in the unchanged helper). The accompanying [deterministic normal-block simulation](replay-harness-parity-redo-power.json), reproduced by `tools/fixtures/replay-parity-redo/power.py`, uses50,000 draws per scenario and seed20262800. Under true parity its estimated score-gate acceptance is98.1% with fixed pilot SD, or92.1% when allowing for the pilot's seven-degree-of-freedom SD uncertainty. At a true five-point deficit it is about5%; at the old observed deficit of2.15points it is68.8% or64.1%, respectively. A genuine deficit is harder to clear, intentionally. It estimates score-gate operating characteristics only; sample, skip and wall guards can further lower full-rule acceptance. Normality, independence and future variance are assumptions, not guarantees.

## Binding rule and failure handling

All the existing non-inferiority guards remain binding, computed from unrounded official whole-game score shares:

1. All thirty new Retcon blocks complete with valid rosters, telemetry and provenance.
2. The one-sided95% Student-t lower bound of the mean candidate-minus-baseline block share is at least `-0.05` (the existing helper's numerical tolerance is unchanged).
3. Candidate skipped turns do not exceed baseline skipped turns.
4. Candidate wall collisions per1,000 turns do not exceed baseline by more than10.
5. All three sample comparisons complete; each candidate share is no more than five points below baseline, with zero candidate sample skips.

Only if every guard passes is this second parity check accepted. No strength gain is claimed. A failure remains a failure; no margin change, extra games, favorable rerun, optional stopping or subset promotion follows. Invalid acquisition stops and preserves partial evidence; any later acquisition continuation requires its own disclosed prospective review. Source changes during collection invalidate this frozen-candidate result.

## Publication and review

Independently review the exact registration, workflow, source/JAR freezes and power evidence before the sole execution. Preserve both failed and successful logs, original reports/official arrays, raw metrics, native Retcon exports, all private numeric endpoints, source/runtime/opponent/asset hashes, and the executed workflow. Record the original failure alongside this separately authorized attempt. Update `docs/improvements.md` after the experiment with the decision, budget, metrics and PR/report/raw links. Independent review and protected-branch checks precede any merge of PR48; implementers do not self-merge.

## Fresh result: every registered noninferiority guard passed

[Run 41649569184](https://github.com/stephanelee/robocode/actions/runs/41649569184), execution `6a5f467ad8d82485923b7cf55e733b1b8cbbfbf5`, completed its sole attempt: all 1,260 rounds, 66 official games/cold finite checkpoints and 60 native Retcon exports. Source/JAR remained the frozen arms above. [Original raw evidence and models](../../benchmarks/replay-harness-parity-redo/README.md) retain exact outcome, runtime, hashes, tooling and immutable `2edd072` registration.

| Retcon measure | Baseline | Extraction |
| --- | ---: | ---: |
| Equal-game mean share | 87.220223% | 87.219318% |
| Official first places / 600 | 561 | 559 |
| AIS skipped turns | 0 | 0 |
| Walls / 1,000 reported turns | 0.084985 | 0.108867 |

Mean candidate-minus-baseline difference was **−0.00090508 pp**, standard error **1.35629467 pp**, and the binding one-sided Student-t lower bound **−2.30542198 pp**, passing the unchanged −5 pp margin. All 30 blocks, skip and wall guards pass. Samples also pass: Target share99.943662→100%, SpinBot97.134328→100%, Walls100→98.675914%, with zero candidate skips. No additional online stage was needed because all measured policies were already default TRAIN.

This separately authorized second attempt establishes the registered bounded noninferiority result, not an improvement, equality at every state, or unrestricted timing equivalence. Whole-game variation and the Student-t model's assumptions remain relevant; AIS seeds do not pair arena starts. The original 380-round failed gate and all its raw bytes remain unchanged and are not pooled. Its archive lacks checkpoint bytes; the fresh archive retains all66. Trace fixtures cover only their visited states. Independent architectural review and protected final-head CI are still required before merge; the score result alone is not permission to restructure further.
