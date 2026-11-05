# KeelHelm follow-up 1: adaptive-gun

**Decision: reject.** Control mask 0; candidate mask 1. No games are pooled with another stage or historical runs.

This is a bounded cohort experiment under parallel load, not a significance or general-strength claim. From-start duel policies are inherited; 1v1 deltas are regression guards and are not attributed to the inactive melee feature.

| Scenario | Control mean share | Candidate mean share | Delta |
| --- | ---: | ---: | ---: |
| melee | 34.26% | 30.75% | -3.51 pp |
| Diamond Port | 59.32% | 65.13% | +5.81 pp |
| Retcon | 93.68% | 95.16% | +1.48 pp |
| Walls | 95.94% | 93.79% | -2.14 pp |
| SpinBot | 100.00% | 100.00% | +0.00 pp |

Melee first places: 42/80 control; 29/80 candidate. Diagnostic policy-change exposure: 3718.

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

Exactly 20 cold games / 260 physical rounds completed: top-four melee 160 rounds; Diamond Port 1v1 forty; Retcon 1v1 forty; Walls/SpinBot 1v1 twenty. Ten control/candidate waves ran with at most two isolated games simultaneously on minimind. No replacements, tuning, early score stopping or GitHub Actions gameplay.

All raw CRC/roster/setup/tick/round/final-score/lifecycle/skip/timing checks pass independently. All 7,017 frozen input hashes verify. Candidate 33 Java 17 tests, control 23 Java 17 tests and 8 Python contracts passed before gameplay; both source/JAR entry-parity checks pass. Production AISurvivor remains unchanged with 822-entry parity; its disclosed local arm64 golden-test limitation remains 89 tests, 5 exact-bit failures and 1 skip.

Minimum sampled memory availability was 47%; peak combined owned RSS 2323.3MiB. Maximum normalized swapout rate was 0.00 pages/2seconds. No resource breach. CPU values are process-lifetime percentages, not instantaneous host utilization. Resource sampling used 2-second monotonic deadlines and actual elapsed time; loop timing excludes event-handler work.

## Provenance and retained evidence

Acquisition source head: 098b50f72559f6bf8c0319617cb69d46d44aad84. Candidate feature JAR SHA256: f305b7f4be7c3309207bdb1569e654bd98b804bb493e914b1b66bbf4dd3215c1. Control JAR: 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Feature-project manifest: 97162a8339d01277cbb5d017ca13a0844ab5307fa80f030440b54c9b5e5f216e. Masks, source/JAR/helper/runtime/package/opponent bytes and schedule were frozen and independently cleared before games. The same feature JAR/project is carried between stages; accepted control hashes bind the previous independent audit.

All original recordings, staged copies, telemetry, invocations, resource samples, snapshots and audit source remain retained. Local archive: local-evidence/keelhelm-followups/stage-1/full-artifacts.tar.gz (304199438bytes), SHA256 54aa2961560541c0a65c9a74465391ca7449efb708fa6279edf6c90239ea7175.

[Per-game scores/telemetry](../../benchmarks/keelhelm-followups/stage-1/results.json), [independent audit](../../benchmarks/keelhelm-followups/stage-1/independent-audit.json), [provenance](../../benchmarks/keelhelm-followups/stage-1/provenance.json), [resources](../../benchmarks/keelhelm-followups/stage-1/resource-summary.json), [artifact proof](../../benchmarks/keelhelm-followups/stage-1/artifact-proof.json).

[Prospective protocol](keelhelm-melee-followups-protocol.md), [issue125](https://github.com/stephanelee/robocode/issues/125), [PR126](https://github.com/stephanelee/robocode/pull/126).
