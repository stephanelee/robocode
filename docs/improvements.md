# Measured AISurvivor improvements

[Tracker: three measured cycles](https://github.com/stephanelee/robocode/issues/8)

Each cycle compares one narrowly scoped candidate with the previously accepted version. Keep source commits, JAR hashes, opponent pin, runtime, round budget, AISurvivor seeds, raw telemetry, failures, and the final accept/reject decision. Do not describe short in-roster trials as proof of generalization or competitive superiority.

## Baselines

- [Original sample baseline](benchmarks/samples-2026-11-02.md): 140 rounds completed. All evaluation policies beat the easy sample roster; trained wall collisions exposed a weakness. This validates integration, not strength.
- [Retcon 1v1 baseline](benchmarks/retcon-2026-11-02.md) / [issue #7](https://github.com/stephanelee/robocode/issues/7): 35 rounds completed. Fixed movement won 1/5, cold DQN 0/5, trained DQN 0/5; wall collisions and damage exposure motivate movement diagnostics. No AISurvivor gameplay change is included in the integration PR.

## Completed experiments

| Cycle | Candidate hypothesis | Status | Required evidence |
| --- | --- | --- | --- |
| 1 | [Back-as-front steering](https://github.com/stephanelee/robocode/issues/10) may reduce turn exposure and unnecessary full rotations. | **Rejected after 120 measured rounds; gameplay reverted.** | [Report and preregistration](improvements/cycle-1.md): mean score share 11.545% → 10.262%; walls/1,000 turns 80.72 → 15.94. |
| 2 | [Wall-smoothing direction](https://github.com/stephanelee/robocode/issues/11) may reduce avoidable wall collisions. | **Rejected after 120 measured rounds; gameplay reverted.** | [Report and preregistration](improvements/cycle-2.md): mean score share 13.685% → 15.076% (+1.391 pp, below required 5 pp); walls/1,000 turns 61.348 → 6.521. |
| 3 | [Frozen targeting statistics](https://github.com/stephanelee/robocode/issues/12) may discard saved class statistics during trained evaluation. | **Rejected after 60 training + 120 evaluation rounds; gameplay reverted.** | [Report and preregistration](improvements/cycle-3.md): mean share 12.294% → 13.440% (+1.146 pp, below 5 pp); one block lost 5.037 pp. Three identical per-seed checkpoint pairs verified. |

All three planned attempts are documented: 420 measured cycle rounds, including 60 training rounds, in addition to the 35-round Retcon integration baseline. None met every preregistered retain criterion, so gameplay and the JAR remain at the accepted version from before these cycles. Secondary gains in collisions or score are observations, not an accepted strength improvement. Issues #10, #11, and #12 remain open. Future candidates require a new documented plan; the sample suite remains a regression/integration check while performance work focuses on Retcon 1v1.

## Source-grounded Retcon strategy

The [new strategy](retcon-strategy.md) separates opponent observability, narrowly verified observation mechanics, and tactical counters. [Native Retcon diagnostics](retcon-observability.md) use unchanged opponent exports as offline evidence.

| Experiment | Outcome | Evidence |
| --- | --- | --- |
| [Bullet-energy scan visibility, issue #18](https://github.com/stephanelee/robocode/issues/18) / [PR #22](https://github.com/stephanelee/robocode/pull/22) | **Rejected after 120 fixed-policy rounds; online veto skipped; gameplay/JAR reverted.** | [Report and preregistration](improvements/foundation-scan-lag.md) / [raw evidence](../benchmarks/foundation-scan-lag/README.md): actual engine confirms `k+1` visibility, but mean score share 18.507%→18.418% (−0.089543 pp, required ≥0) and wall-rate ratio 1.671 (>1.2) failed the approved guards. |

The new foundation experiment adds 120 scored rounds without changing the original three outcomes or their criteria. A real source-contract defect remains documented, but its tested correction is not retained; issue #18 stays open. No strength improvement has been established.


## Controlled component benchmarks

| Experiment | Measured budget / decision | Evidence |
| --- | --- | --- |
| Dual-model swept-corridor v2 / [PR #24](https://github.com/stephanelee/robocode/pull/24) | **Offline diagnostic retained; live selector not promoted.** 816 development +90 withheld states ×4 uncertainty profiles ×3 delays =10,872 deterministic rows; 13 fixture tests; zero score matches. Fresh/delay0 conditional safety167→686/816 and29→76/90, but age5+15°/delay3 only47→103/816 and2→4/90; unrestricted heading certifies none. Wall contact remains common. | [Method/results](dual-aim-diagnostic.md), [summary](../benchmarks/dual-aim-v2/summary.json), full raw rows in the PR's `Component diagnostics` workflow artifact `dual-aim-v2-raw`. Conditional geometry is not gameplay strength or independent collision-rate evidence. |

Conformance correction: [v1 exploratory report](dual-aim-v1-invalid.md) is invalidated by a missing second engine corner-constraint pass; its source/raw artifacts are preserved. Repaired v2 reran the unchanged full matrix:167 development rows changed (wall counters plus2 clearances), no safety decision/command/withheld row changed. The fixed withheld set was already inspected and is not fresh generalization evidence. [Full comparison](../benchmarks/dual-aim-v2/conformance-diff.json).
