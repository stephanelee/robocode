# KeelHelm next 2: radar-revisit

**Decision: reject.** Contemporary control mask 0; candidate mask 16. Exactly 20 cold games / 260 rounds ran serially on minimind; no Actions gameplay, tuning, replacement, early score stop or pooling.

| Scenario | Control mean share | Candidate mean share | Delta |
| --- | ---: | ---: | ---: |
| melee | 35.31% | 29.84% | -5.48 pp |
| Diamond Port | 51.53% | 56.66% | +5.13 pp |
| Retcon | 95.27% | 92.36% | -2.91 pp |
| Walls | 96.92% | 99.18% | +2.26 pp |
| SpinBot | 97.83% | 100.00% | +2.17 pp |

Melee round wins: control 45/80; candidate 33/80. Actual policy-action exposure: 1209. Next accepted mask: **0**.

| Gate | Passed |
| --- | --- |
| meleeShare | True |
| meleeGain | False |
| Diamond Port | True |
| Retcon | True |
| Walls | True |
| SpinBot | True |
| observedPolicyChange | True |

Four-contestant aggregate standings, control arm (four 20-round games):

| Bot | Score | Bullet damage | Survival points | Round wins |
| --- | ---: | ---: | ---: | ---: |
| KeelHelm | 15,918 | 4,726 | 9,300 | 45 |
| Prism | 12,875 | 6,050 | 6,300 | 8 |
| Diamond Port | 12,319 | 5,346 | 6,050 | 19 |
| Keel | 3,968 | 1,235 | 2,350 | 8 |

Four-contestant aggregate standings, candidate arm (four 20-round games):

| Bot | Score | Bullet damage | Survival points | Round wins |
| --- | ---: | ---: | ---: | ---: |
| Diamond Port | 15,948 | 6,262 | 7,950 | 35 |
| KeelHelm | 13,487 | 3,944 | 8,150 | 33 |
| Prism | 12,899 | 6,167 | 6,150 | 8 |
| Keel | 2,924 | 1,027 | 1,700 | 4 |

All raw CRC/rosters/setup/contiguous ticks/round endpoints/final scores/telemetry/cleanup and 7,024 frozen hashes verify independently. Zero own skips; maximum logged loop 7ms. Minimum sampled available memory 43%; peak owned RSS 1259.4MiB. Resource guards pass. Loop timing excludes callbacks/termination; no timing-repair or responsiveness guarantee. The prior timing cause remains unresolved.

Candidate 39 Java 17 tests, original baseline 23 tests, 14 Python contracts and source/JAR entry parity pass. Production AISurvivor is unchanged; prior disclosed arm64 validation remains 89 tests with 5 exact-bit golden failures and 1 skip, 822-entry parity. Fixtures establish geometry/state contracts only, not strength.

Primary means weight each game equally. Engine placement is not paired by the bot seed. From-start duel policy is inherited; its score differences are regression guards, not attributed melee improvements. Four finite melee games per arm and adaptive staged promotion support a bounded engineering decision only, not significance or general-strength claims.

Source head: 2ba108afa3bf1de4cb02eb4f36ab8be19dee19fe. Feature JAR SHA256: 6ffb14abea4013943577af662f66ddd7b1d47a18a6d484a903b11bafbd058aab. Control JAR SHA256: 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Feature-project manifest: 164e9560fd8198bdbd087f890fbc4a42ea4f4590b27da25523c275b9fb50eb03. Opponents/runtime/commands/environment and source snapshots are retained in the exact freeze.

Project accounting after this complete stage: **1390 completed physical rounds plus three historical partial rounds**. This new program adds exactly 520 complete rounds; invalid historical attempts stay separate.

All recordings, staged bots, source/JAR snapshots and any emitted model files, telemetry, resources, invocation/cleanup, validation logs and independent auditor sources are retained outside Git in robocode-keelhelm-followups/next/stage-2/full-artifacts.tar.gz. SHA256 7099b5f8e1c5f5c56b392a1cb7d6dd92f1ab328cc110a4cd6f1757917d67a09d.

[Scores](../../benchmarks/keelhelm-next/stage-2/results.json), [independent audit](../../benchmarks/keelhelm-next/stage-2/independent-audit.json), [provenance](../../benchmarks/keelhelm-next/stage-2/provenance.json), [resources](../../benchmarks/keelhelm-next/stage-2/resource-summary.json), [archive proof](../../benchmarks/keelhelm-next/stage-2/artifact-proof.json).

[Prospective protocol](keelhelm-next-protocol.md), [fixed implementations](keelhelm-next-implementations.md), [issue125](https://github.com/stephanelee/robocode/issues/125), [PR126](https://github.com/stephanelee/robocode/pull/126).
