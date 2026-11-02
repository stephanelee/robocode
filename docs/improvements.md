# Measured AISurvivor improvements

[Tracker: three measured cycles](https://github.com/stephanelee/robocode/issues/8)

Each cycle compares one narrowly scoped candidate with the previously accepted version. Keep source commits, JAR hashes, opponent pin, runtime, round budget, AISurvivor seeds, raw telemetry, failures, and the final accept/reject decision. Do not describe short in-roster trials as proof of generalization or competitive superiority.

## Baselines

- [Original sample baseline](benchmarks/samples-2026-11-02.md): 140 rounds completed. All evaluation policies beat the easy sample roster; trained wall collisions exposed a weakness. This validates integration, not strength.
- [Retcon 1v1 baseline](benchmarks/retcon-2026-11-02.md) / [issue #7](https://github.com/stephanelee/robocode/issues/7): 35 rounds completed. Fixed movement won 1/5, cold DQN 0/5, trained DQN 0/5; wall collisions and damage exposure motivate movement diagnostics. No AISurvivor gameplay change is included in the integration PR.

## Planned cycles

| Cycle | Candidate hypothesis | Status | Required evidence |
| --- | --- | --- | --- |
| 1 | [Back-as-front steering](https://github.com/stephanelee/robocode/issues/10) may reduce turn exposure and unnecessary full rotations. | **Rejected after 120 measured rounds; gameplay reverted.** | [Report and preregistration](improvements/cycle-1.md): mean score share 11.545% → 10.262%; walls/1,000 turns 80.72 → 15.94. |
| 2 | [Wall-smoothing direction](https://github.com/stephanelee/robocode/issues/11) may reduce avoidable wall collisions. | **Rejected after 120 measured rounds; gameplay reverted.** | [Report and preregistration](improvements/cycle-2.md): mean score share 13.685% → 15.076% (+1.391 pp, below required 5 pp); walls/1,000 turns 61.348 → 6.521. |
| 3 | [Frozen targeting statistics](https://github.com/stephanelee/robocode/issues/12) may discard saved class statistics during trained evaluation. | Planned; no implementation/results yet. | Compare both versions using identical trained checkpoint copies; document statistics semantics and any gains or regressions. |

Candidates are provisional until Retcon diagnostics identify the next useful change. A failed hypothesis is a recorded result, not a reason to omit the trial. Keep or reject each candidate on the evidence and document uncertainty. Preserve the sample regression/smoke suite; new melee work is deferred while the first challenge focuses on 1v1.

