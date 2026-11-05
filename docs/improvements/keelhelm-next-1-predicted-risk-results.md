# KeelHelm next 1: predicted-risk

**Decision: reject.** Contemporary control mask 0; candidate mask 8. Exactly 20 cold games / 260 rounds ran serially on minimind; no Actions gameplay, tuning, replacement, early score stop or pooling.

| Scenario | Control mean share | Candidate mean share | Delta |
| --- | ---: | ---: | ---: |
| melee | 25.96% | 33.17% | +7.21 pp |
| Diamond Port | 54.17% | 47.10% | -7.07 pp |
| Retcon | 94.20% | 92.64% | -1.57 pp |
| Walls | 94.97% | 94.98% | +0.01 pp |
| SpinBot | 99.87% | 99.86% | -0.01 pp |

Melee round wins: control 22/80; candidate 36/80. Actual policy-action exposure: 6621. Next accepted mask: **0**.

| Gate | Passed |
| --- | --- |
| meleeShare | True |
| meleeGain | True |
| Diamond Port | False |
| Retcon | True |
| Walls | True |
| SpinBot | True |
| observedPolicyChange | True |

Four-contestant aggregate standings, control arm (four 20-round games):

| Bot | Score | Bullet damage | Survival points | Round wins |
| --- | ---: | ---: | ---: | ---: |
| Diamond Port | 15,515 | 5,628 | 8,150 | 42 |
| Prism | 15,287 | 7,182 | 7,150 | 12 |
| KeelHelm | 11,846 | 3,794 | 7,100 | 22 |
| Keel | 2,973 | 1,108 | 1,600 | 4 |

Four-contestant aggregate standings, candidate arm (four 20-round games):

| Bot | Score | Bullet damage | Survival points | Round wins |
| --- | ---: | ---: | ---: | ---: |
| KeelHelm | 14,954 | 4,187 | 9,050 | 36 |
| Prism | 14,606 | 6,751 | 7,150 | 17 |
| Diamond Port | 13,037 | 5,425 | 6,400 | 25 |
| Keel | 2,564 | 992 | 1,400 | 2 |

All raw CRC/rosters/setup/contiguous ticks/round endpoints/final scores/telemetry/cleanup and 7,024 frozen hashes verify independently. Zero own skips; maximum logged loop 7ms. Minimum sampled available memory 43%; peak owned RSS 1213.5MiB. Resource guards pass. Loop timing excludes callbacks/termination; no timing-repair or responsiveness guarantee. The prior timing cause remains unresolved.

Candidate 39 Java 17 tests, original baseline 23 tests, 14 Python contracts and source/JAR entry parity pass. Production AISurvivor is unchanged; prior disclosed arm64 validation remains 89 tests with 5 exact-bit golden failures and 1 skip, 822-entry parity. Fixtures establish geometry/state contracts only, not strength.

Primary means weight each game equally. Engine placement is not paired by the bot seed. From-start duel policy is inherited; its score differences are regression guards, not attributed melee improvements. Four finite melee games per arm and adaptive staged promotion support a bounded engineering decision only, not significance or general-strength claims.

Source head: 0d0da85900c75d940384479baa2b6b81e0bb1489. Feature JAR SHA256: 6ffb14abea4013943577af662f66ddd7b1d47a18a6d484a903b11bafbd058aab. Control JAR SHA256: 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Feature-project manifest: 164e9560fd8198bdbd087f890fbc4a42ea4f4590b27da25523c275b9fb50eb03. Opponents/runtime/commands/environment and source snapshots are retained in the exact freeze.

Project accounting after this complete stage: **1130 completed physical rounds plus three historical partial rounds**. This new program adds exactly 260 complete rounds; invalid historical attempts stay separate.

All recordings, staged bots, source/JAR snapshots and any emitted model files, telemetry, resources, invocation/cleanup, validation logs and independent auditor sources are retained outside Git in robocode-keelhelm-followups/next/stage-1/full-artifacts.tar.gz. SHA256 93853a2e8077a650616205b8f4b1aef823bf15dae301aa75213ae6ea9b73046a.

[Scores](../../benchmarks/keelhelm-next/stage-1/results.json), [independent audit](../../benchmarks/keelhelm-next/stage-1/independent-audit.json), [provenance](../../benchmarks/keelhelm-next/stage-1/provenance.json), [resources](../../benchmarks/keelhelm-next/stage-1/resource-summary.json), [archive proof](../../benchmarks/keelhelm-next/stage-1/artifact-proof.json).

[Prospective protocol](keelhelm-next-protocol.md), [fixed implementations](keelhelm-next-implementations.md), [issue125](https://github.com/stephanelee/robocode/issues/125), [PR126](https://github.com/stephanelee/robocode/pull/126).
