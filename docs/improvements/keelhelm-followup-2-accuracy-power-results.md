# KeelHelm follow-up 2: accuracy-power

**Decision: reject.** Control mask 0; candidate mask 2. No games are pooled with another stage or historical runs.

This is a fresh serial continuation; the original parallel power attempt (32 completed rounds plus one partial) is excluded. No serial game is pooled with parallel results. Previous project accounting was 552 completed rounds plus one partial; this continuation adds 260 rounds per completed stage.

This is a bounded cohort experiment under serial load, not a significance or general-strength claim. From-start duel policies are inherited; 1v1 deltas are regression guards and are not attributed to the inactive melee feature.

| Scenario | Control mean share | Candidate mean share | Delta |
| --- | ---: | ---: | ---: |
| melee | 30.78% | 30.61% | -0.17 pp |
| Diamond Port | 51.61% | 50.74% | -0.88 pp |
| Retcon | 92.73% | 94.65% | +1.92 pp |
| Walls | 98.52% | 94.77% | -3.75 pp |
| SpinBot | 100.00% | 100.00% | +0.00 pp |

Melee first places: 34/80 control; 34/80 candidate. Diagnostic policy-change exposure: 3564.

| Gate | Passed |
| --- | --- |
| meleeShare | True |
| meleeGain | False |
| Diamond Port | True |
| Retcon | True |
| Walls | True |
| SpinBot | True |
| observedPolicyChange | True |

Next independently accepted mask: **0**. Retain the last accepted control; this feature remains disabled by default.

## Fixed budget and validation

Exactly 20 cold games / 260 physical rounds completed: top-four melee 160 rounds; Diamond Port 1v1 forty; Retcon 1v1 forty; Walls/SpinBot 1v1 twenty. Twenty serial waves ran with exactly one engine game at a time on minimind. No replacements, tuning, early score stopping or GitHub Actions gameplay.

All raw CRC/roster/setup/tick/round/final-score/lifecycle/skip/timing checks pass independently. All 7,019 frozen input hashes verify. Candidate 33 Java 17 tests, control 23 Java 17 tests and 12 Python contracts passed before gameplay; both source/JAR entry-parity checks pass. Production AISurvivor remains unchanged with 822-entry parity; its disclosed local arm64 golden-test limitation remains 89 tests, 5 exact-bit failures and 1 skip.

Minimum sampled memory availability was 40%; peak combined owned RSS 1219.0MiB. Maximum normalized swapout rate was 0.00 pages/2seconds. No resource breach. CPU values are process-lifetime percentages, not instantaneous host utilization. Resource sampling used 2-second monotonic deadlines and actual elapsed time; loop timing excludes event-handler work.

## Provenance and retained evidence

Acquisition source head: b8b22abcbc558231696deeb32e4c3b81d75ee23d. Candidate feature JAR SHA256: f305b7f4be7c3309207bdb1569e654bd98b804bb493e914b1b66bbf4dd3215c1. Control JAR: 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Feature-project manifest: 97162a8339d01277cbb5d017ca13a0844ab5307fa80f030440b54c9b5e5f216e. Masks, source/JAR/helper/runtime/package/opponent bytes and schedule were frozen and independently cleared before games. The same feature JAR/project is carried between stages; accepted control hashes bind the previous independent audit.

All original recordings, staged copies, telemetry, invocations, resource samples, snapshots and audit source remain retained. Local archive: local-evidence/keelhelm-followups-serial/stage-2/full-artifacts.tar.gz (307999102bytes), SHA256 77acc31c36a84f1e9db454d4ecb077b53359983ce5153e0e3d8917751a9835cf. Large archives are retained outside Git through the linked evidence directory.

[Per-game scores/telemetry](../../benchmarks/keelhelm-followups/stage-2/results.json), [independent audit](../../benchmarks/keelhelm-followups/stage-2/independent-audit.json), [provenance](../../benchmarks/keelhelm-followups/stage-2/provenance.json), [resources](../../benchmarks/keelhelm-followups/stage-2/resource-summary.json), [artifact proof](../../benchmarks/keelhelm-followups/stage-2/artifact-proof.json).

[Prospective policy protocol](keelhelm-melee-followups-protocol.md), [serial continuation](keelhelm-followups-serial-protocol.md), [issue125](https://github.com/stephanelee/robocode/issues/125), [PR126](https://github.com/stephanelee/robocode/pull/126).
