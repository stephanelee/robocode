# Learned aim and outcome-based power: final decisions

**Both candidates are rejected. Accepted KeelHelm 0.2.1 remains byte-exact.** The two isolated candidates were built, tested and compared separately on minimind. Neither enters the default bot or changes the completed tournament.

| Candidate | Control melee share | Candidate melee share | Change | Failed registered gates |
| --- | ---: | ---: | ---: | --- |
| Learned displacement aim 0.5.0 | 29.87% | 31.52% | +1.65 pp | Gain below +2 pp; Diamond Port duel −6.85 pp exceeds the −5 pp allowance |
| Confirmed-outcome power 0.6.0 | 30.04% | 31.60% | +1.55 pp | Gain below +2 pp; 85 changed shots below the required 100 |

Each comparison used its own 40 cold serial games / 520 rounds: eight 20-round top-four melee games per arm, four 10-round duels per arm against each Diamond Port and Retcon, and two 5-round duels per arm against each Walls and SpinBot. The combined fresh cost is **80 games / 1,040 completed rounds, zero partial rounds**. Cohorts and historical results are not pooled. Cumulative KeelHelm development/diagnostic accounting is 3,030 completed rounds plus three historical partial rounds, including the separate 80-round snapshot in PR #128.

The learned gun changed 6,349 actual confirmed shot directions across all three melee opponents. The power policy changed only 85 of 10,459 confirmed melee shots (0.81%), all from 1.6 to 2.0 power: 79 aimed at Keel and six at Diamond Port. Its restrictive rule did not obtain the registered action exposure. These measurements do not establish general strength or justify relaxing either gate after the results.

Independent audits validate every original recording and official score, legal shot traces, feedback-window chronology, unchanged frozen inputs, staged entrant/shared libraries, resources and owned cleanup. Both studies have zero own skipped turns. The first gun auditor incorrectly assumed one terminal outcome per physical bullet; the pinned engine allows collision and hit events in the same tick. The corrected audit preserves all events, and both auditor versions and engine evidence are retained. No game, score or experimental input changed to correct the auditor.

Java 17 builds pass 30 gun-candidate tests and 29 power-candidate tests, with source/committed-JAR parity. Two gun-driver and four power-driver contracts pass. Games use pinned JDK 25, runner, Python/API and opponents with unchanged deadlines/heaps/GC. Contract fixtures establish behavior only. All 40 normal-project files still match the accepted published baseline, including JAR `19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b`.

Engine starts are unpaired; matching seeds govern bot choices only. The inherited from-start duel policy is unchanged, so its score deltas are transfer-guard measurements, not proof of a causal duel-policy change. No threshold tuning, selective replacement, retry, resume, extra games or promotion followed rejection.

All original recordings, staged source/JAR/support, runtime/opponent hashes, invocations, resource/cleanup logs, diagnostics and auditor sources are retained in persistent private archives. Source and compiled rejected candidates remain under `benchmarks/keelhelm-learned-gun/candidate-project` and `benchmarks/keelhelm-outcome-power/candidate-project`, outside eligible/default entrants.

- [Learned-gun report and raw-evidence links](keelhelm-learned-gun-results.md), [prospective protocol](keelhelm-learned-gun-protocol.md), [issue #129](https://github.com/stephanelee/robocode/issues/129).
- [Outcome-power report and raw-evidence links](keelhelm-outcome-power-results.md), [prospective protocol](keelhelm-outcome-power-protocol.md), [issue #131](https://github.com/stephanelee/robocode/issues/131).
- [PR #130](https://github.com/stephanelee/robocode/pull/130), [improvements ledger](../improvements.md).
