# KeelHelm next 4: duel-transfer

**Decision: reject.** Contemporary control mask 0; candidate mask 64. Exactly 20 cold games / 260 rounds ran serially on minimind; no Actions gameplay, tuning, replacement, early score stop or pooling.

| Scenario | Control mean share | Candidate mean share | Delta |
| --- | ---: | ---: | ---: |
| melee | 32.62% | 25.74% | -6.87 pp |
| Diamond Port | 50.48% | 45.12% | -5.36 pp |
| Retcon | 96.02% | 91.26% | -4.76 pp |
| Walls | 93.56% | 98.27% | +4.71 pp |
| SpinBot | 97.45% | 94.19% | -3.26 pp |

Melee round wins: control 35/80; candidate 23/80. Actual policy-action exposure: 5622. Next accepted mask: **0**.

| Gate | Passed |
| --- | --- |
| meleeShare | True |
| meleeGain | False |
| Diamond Port | False |
| Retcon | True |
| Walls | True |
| SpinBot | True |
| observedPolicyChange | True |

Four-contestant aggregate standings, control arm (four 20-round games):

| Bot | Score | Bullet damage | Survival points | Round wins |
| --- | ---: | ---: | ---: | ---: |
| KeelHelm | 14,703 | 4,704 | 8,400 | 35 |
| Diamond Port | 14,512 | 5,728 | 7,400 | 31 |
| Prism | 12,629 | 5,749 | 6,350 | 9 |
| Keel | 3,235 | 1,098 | 1,850 | 5 |

Four-contestant aggregate standings, candidate arm (four 20-round games):

| Bot | Score | Bullet damage | Survival points | Round wins |
| --- | ---: | ---: | ---: | ---: |
| Diamond Port | 15,592 | 6,133 | 7,850 | 34 |
| Prism | 15,001 | 6,477 | 7,500 | 19 |
| KeelHelm | 11,702 | 3,735 | 6,900 | 23 |
| Keel | 3,129 | 1,143 | 1,750 | 4 |

All raw CRC/rosters/setup/contiguous ticks/round endpoints/final scores/telemetry/cleanup and 7,024 frozen hashes verify independently. Zero own skips; maximum logged loop 8ms. Minimum sampled available memory 40%; peak owned RSS 1214.5MiB. Resource guards pass. Loop timing excludes callbacks/termination; no timing-repair or responsiveness guarantee. The prior timing cause remains unresolved.

Candidate 39 Java 17 tests, original baseline 23 tests, 14 Python contracts and source/JAR entry parity pass. Production AISurvivor is unchanged; prior disclosed arm64 validation remains 89 tests with 5 exact-bit golden failures and 1 skip, 822-entry parity. Fixtures establish geometry/state contracts only, not strength.

Primary means weight each game equally. Engine placement is not paired by the bot seed. From-start duel policy is inherited; its score differences are regression guards, not attributed melee improvements. Four finite melee games per arm and adaptive staged promotion support a bounded engineering decision only, not significance or general-strength claims.

Source head: 7cfc2d90bd2ae04d99eeed6d374dc6a97025ffb2. Feature JAR SHA256: 6ffb14abea4013943577af662f66ddd7b1d47a18a6d484a903b11bafbd058aab. Control JAR SHA256: 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Feature-project manifest: 164e9560fd8198bdbd087f890fbc4a42ea4f4590b27da25523c275b9fb50eb03. Opponents/runtime/commands/environment and source snapshots are retained in the exact freeze.

Project accounting after this complete stage: **1910 completed physical rounds plus three historical partial rounds**. This new program adds exactly 1040 complete rounds; invalid historical attempts stay separate.

All recordings, staged bots, source/JAR snapshots and any emitted model files, telemetry, resources, invocation/cleanup, validation logs and independent auditor sources are retained outside Git in robocode-keelhelm-followups/next/stage-4/full-artifacts.tar.gz. SHA256 efca061ee76533304456c1f3d9b2aabd86a63d1f22c9fc654b065c74874f680c.

[Scores](../../benchmarks/keelhelm-next/stage-4/results.json), [independent audit](../../benchmarks/keelhelm-next/stage-4/independent-audit.json), [provenance](../../benchmarks/keelhelm-next/stage-4/provenance.json), [resources](../../benchmarks/keelhelm-next/stage-4/resource-summary.json), [archive proof](../../benchmarks/keelhelm-next/stage-4/artifact-proof.json).

[Prospective protocol](keelhelm-next-protocol.md), [fixed implementations](keelhelm-next-implementations.md), [issue125](https://github.com/stephanelee/robocode/issues/125), [PR126](https://github.com/stephanelee/robocode/pull/126).
