# KeelHelm melee follow-ups: final decisions

All four fresh comparisons completed on minimind, exactly one engine game at a time: **80 valid games / 1,040 rounds**, no replacements, tuning, pooling, invalid competitive game or own skipped turn. Candidate source/JAR and gameplay constants remained frozen throughout. Each comparison used 80 melee rounds per arm plus Diamond Port / Retcon / Walls / SpinBot 1v1 guards.

| Candidate | Control melee share | Candidate melee share | Delta | Decision / failed gate |
| --- | ---: | ---: | ---: | --- |
| [predicted-risk](keelhelm-next-1-predicted-risk-results.md) | 25.96% | 33.17% | +7.21 pp | reject: Diamond Port 1v1 guard |
| [radar-revisit](keelhelm-next-2-radar-revisit-results.md) | 35.31% | 29.84% | -5.48 pp | reject: melee gain |
| [shot-obstruction](keelhelm-next-3-shot-obstruction-results.md) | 32.25% | 28.84% | -3.41 pp | reject: melee gain |
| [duel-transfer](keelhelm-next-4-duel-transfer-results.md) | 32.62% | 25.74% | -6.87 pp | reject: melee gain and Diamond Port 1v1 guard |

**No new policy qualified for promotion.** Accepted control remains KeelHelm 0.2.1, mask 0, JAR SHA256 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. The normal project path is restored to that exact source/JAR. The rejected 0.4 project/JAR is retained separately under `benchmarks/keelhelm-next/candidate-project`, outside the eligible entrant catalog; its JAR SHA256 is 6ffb14abea4013943577af662f66ddd7b1d47a18a6d484a903b11bafbd058aab.

The movement candidate finished first on aggregate in its four melee games: KeelHelm 14,954, Prism 14,606, Diamond Port 13,037, Keel 2,564; round wins 36, 17, 25, 2 respectively. Its Diamond Port duel mean fell 54.17%→47.10% (−7.07 pp), beyond the fixed −5 pp guard, so it was rejected despite that melee result. From-start duel policies are inherited; the observed difference is not proof of causal degradation.

Each mean weights its four melee games equally. Engine placement was not paired by bot-side seeds. These are bounded engineering probes with adaptive staged decisions, not significance or general-strength claims. The nineteen obstruction suppressions are sparse exposure. Fixtures validate contracts and acquisition, not strength. No gate was relaxed after observing results.

Timing diagnostics did not establish a cause or repair for earlier round-start skips. The fresh 80 games had zero own skips, but this does not guarantee responsiveness. JFR collector setup/capture failures, a readable 5-second trace and the zero-game exact-shell correction remain separately documented. No extra diagnostic games ran after the registered stops.

Fresh competitive budget 1,040 completed rounds; prior development/diagnostic accounting 870 completed plus three partial rounds. Final project accounting: **1,910 completed physical rounds plus three historical partial rounds**. The completed tournament itself remains frozen and separate. All recordings, staged inputs, source/JAR snapshots, any emitted files, telemetry, resources, cleanup, failures and auditor sources are preserved in persistent private archives; public reports link exact archive SHA256 proofs.

After the final audit, packaging relocation changes only project selection and documentation. The historical source heads/freezes refer to the pre-relocation path `experiments/KeelHelm`; archived execution/source snapshots and Git commits preserve those exact bytes and paths. Do not run an old freeze against a differently packaged tree or resume an attempted evidence directory. New preparations explicitly freeze the resolved `candidateProject`, and execution uses that frozen value. A future study requires a fresh prospective protocol and independent approval.

[Protocol](keelhelm-next-protocol.md), [fixed implementations](keelhelm-next-implementations.md), [timing D1](keelhelm-timing-d1-results.md), [timing D2](keelhelm-timing-d2-results.md), [zero-game collector correction](keelhelm-timing-shell-fix.md), [issue125](https://github.com/stephanelee/robocode/issues/125), [PR126](https://github.com/stephanelee/robocode/pull/126).

Post-study verification passed: 23 accepted-project Java 17 tests, 39 rejected-project Java 17 tests and 16 current Python checks. Both rebuilt JARs match the exact recorded SHA256 values and every entry. All 40 tracked files in the normal project match the accepted published source; the relocated candidate retains all 44 measured project files unchanged. These are packaging/configuration checks, not additional games or strength evidence.
