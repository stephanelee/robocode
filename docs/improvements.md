# Measured AISurvivor improvements

[Tracker: three measured cycles](https://github.com/stephanelee/robocode/issues/8)

The Retcon 1v1 baseline is recorded; gameplay candidates remain unimplemented in this integration PR. Each cycle compares one narrowly scoped candidate with the previously accepted version. Keep source commits, JAR hashes, opponent pin, runtime, round budget, AISurvivor seeds, raw telemetry, failures, and the final accept/reject decision. Do not describe short in-roster trials as proof of generalization or competitive superiority.

## Baselines

- [Original sample baseline](benchmarks/samples-2026-11-02.md): 140 rounds completed. All evaluation policies beat the easy sample roster; trained wall collisions exposed a weakness. This validates integration, not strength.
- [Retcon 1v1 baseline](benchmarks/retcon-2026-11-02.md) / [issue #7](https://github.com/stephanelee/robocode/issues/7): 35 rounds completed. Fixed movement won 1/5, cold DQN 0/5, trained DQN 0/5; wall collisions and damage exposure motivate movement diagnostics. No AISurvivor gameplay change is included in the integration PR.

## Planned cycles

| Cycle | Candidate hypothesis | Status | Required evidence |
| --- | --- | --- | --- |
| 1 | [Back-as-front steering](https://github.com/stephanelee/robocode/issues/10) may reduce turn exposure and unnecessary full rotations. | Planned; choose after baseline diagnostics. | Candidate vs previous version, repeated Retcon 1v1 trials, damage/score/wall and skipped-turn telemetry. |
| 2 | [Wall-smoothing direction](https://github.com/stephanelee/robocode/issues/11) may reduce avoidable wall collisions. | Planned; no implementation/results yet. | Same comparison method, report collision changes and damage/score tradeoffs. |
| 3 | [Frozen targeting statistics](https://github.com/stephanelee/robocode/issues/12) may limit cold evaluation; examine the intended evaluation contract before changing it. | Planned; no implementation/results yet. | Isolate the behavior change; document checkpoint/statistics semantics and any gains or regressions. |

Candidates are provisional until Retcon diagnostics identify the next useful change. A failed hypothesis is a recorded result, not a reason to omit the trial. Keep or reject each candidate on the evidence and document uncertainty. Preserve the sample regression/smoke suite; new melee work is deferred while the first challenge focuses on 1v1.

