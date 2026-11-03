# Advanced opponent improvement task

[GitHub issue53](https://github.com/stephanelee/robocode/issues/53) tracks this work. The user's priority is **win 1v1 first, then improve melee**. Measured intermediate gains are progress, not completion of the winning objective.

## Starting evidence

| Frozen production test | Official first places | Mean score share | Status |
| --- | --- | --- | --- |
| Retcon0.2, fresh confirmation | 187/200 | 86.745% | Passed the registered confirmation; [PR46](https://github.com/stephanelee/robocode/pull/46) |
| Diamond Port1.8.28, 1v1 | 0/60 | 3.886% | Valid negative baseline; [PR51](https://github.com/stephanelee/robocode/pull/51) |
| Adaptive Prime1.0, 1v1 | 3/60 | 17.683% | Valid negative baseline; [PR51](https://github.com/stephanelee/robocode/pull/51) |
| Retcon + Diamond + Adaptive, four-bot melee | 1/60 | 12.212% | Rank4 in every game; [PR52](https://github.com/stephanelee/robocode/pull/52) |

All those cohorts had zero AIS skipped turns. Their opponents, source/runtime pins, raw records and cold models are retained in the reports linked from [improvements.md](improvements.md). No failed or invalid attempt is replaced or pooled into a new confirmation.

## Current 1v1 work

The default duel controller learns hit-danger bins but originally ignored them when selecting movement: it optimized clearance from head-on and linear aim only. Diamond Port and Adaptive Prime use richer learned guns. This is a concrete source mismatch and a plausible weakness; it is not yet proof of the losses' cause.

The [first candidate](improvements/learned-duel-danger.md) uses those existing legally matched-hit profiles in the physics-aware controller. With insufficient history it preserves the analytic fallback. The gun, radar, energy policy and non-default modes stay unchanged. Code and focused tests are implemented; acceptance requires the prospectively registered screen and fresh confirmation, including absolute Retcon-win and sample regression guards.

Root integrates code and freezes the protocol. The benchmark agent owns the existing native-opponent harness and actual collection. A separate reviewer independently clears the exact premeasurement source/protocol, audits every outcome and merges only accepted changes after protected CI. Work is delegated in parallel without duplicating experiments or changing a running candidate.

After every experiment, update [improvements.md](improvements.md), its detailed report and issue53 with budgets, metrics, accept/reject decisions, limitations and PR/raw links. Rejected gameplay/JAR changes are restored while patches and evidence remain available. Future candidates require a new prospective design; no post-outcome threshold relaxation or favorable rerun.

## Melee backlog, after 1v1

The advanced melee baseline recorded549 wall and309 bot collisions. A reproducible command defect exists: at `(975,500)`, heading east with speed8 in a1000-square arena, an inward desired heading passes endpoint-only smoothing, but the engine first moves on the old heading to `x983`, beyond the legal wall at982. Even reversal can be unsafe near the wall; an instant stop may be necessary. This does not prove the defect caused every observed collision. A future bounded wall guard needs physics fixtures and its own whole-melee comparison; no melee prototype or extra games are currently scheduled.

## Follow-up and execution

An hourly follow-up in this chat checks actual agents, GitHub PRs/issues/Actions and results, resumes idle work, and continues the authorized improvement loop. It stays quiet for unchanged pending work and reports meaningful results, failures or required input. Its job is to advance measured improvements, not merely report activity.

Local runs on XPSDESTKOP1 are authorized when useful; Java17 builds already run there. Preserve each experiment's registered environment and isolate headless ports from the interactive game on7654. All GitHub runners use exactly Ubuntu26.04; build/test/JAR parity uses Java17 before JDK25 Retcon/native-opponent matches. Preserve user author identity and published evidence hashes.
