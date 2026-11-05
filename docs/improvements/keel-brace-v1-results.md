# KeelBrace: reject after the 120-round screen

KeelBrace failed its registered screen. Only **1 of 3 candidate Diamond games exceeded 50% score share**, against the required 2 of 3. All 14 games and 120 physical rounds completed once; no invalid or unplayed rounds, replacements or acquisition failures were found. **No confirmation, rerun or promotion is authorized for this candidate.** Original Keel remains the experimental baseline and production AISurvivor is unchanged.

This is the owner’s result record for [PR115](https://github.com/stephanelee/robocode/pull/115) and [issue114](https://github.com/stephanelee/robocode/issues/114). The [independent verdict](https://github.com/stephanelee/robocode/issues/114#issuecomment-6648716884) and [full independent report](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/independent-outcome/independent-outcome-report.md) support the decision. Workflow success alone did not decide acceptance.

## Hypothesis, treatment and fixed comparison

The prospective hypothesis was that correcting only Keel’s ram-energy observation epoch would improve Retcon score share while retaining scoped Diamond performance. In the pinned engine, ram damage occurs before same-turn scans, while bullet damage and hit bonuses occur afterward. Original Keel applies both kinds of correction one scan later. KeelBrace adds a constant-space same-turn ram bucket, consumed once by the accepted scan, while retaining the original next-scan bullet correction and all gun, movement, power, wave and model policies. It is an isolated original-Keel derivative, with no rejected Helm treatment or production change.

The [original protocol](keel-brace-v1.md), [fixed game registry](../../benchmarks/keel-brace-v1/games.json), [baseline/pins](../../benchmarks/keel-brace-v1/baseline.json) and two prospectively reviewed [runtime amendments](keel-brace-runtime-identity-amendment.md) / [v2](keel-brace-runtime-identity-amendment-v2.md) remain unchanged. The sole screen used 14 serial cold games: each arm played three 10-round Diamond games, two 10-round Retcon games, one 5-round Walls game and one 5-round AISurvivor game. All games were collected regardless of score. The conditional 300-round fresh confirmation was not opened. No historical Helm/Keel games or selected contact-bearing subgames enter this comparison.

## Scores, wins and variation

Shares below are equal-weight means of cold games. Variation is sample SD of game shares in percentage points; a one-game SD is undefined. “First places” counts rounds, not independent game wins.

| Opponent | Brace share | Keel share | Difference (pp) | Brace / Keel first places | Brace / Keel game SD (pp) |
|---|---:|---:|---:|---:|---:|
| Diamond Port | 51.647928% | 44.195134% | +7.452794 | 22/30 / 20/30 | 16.794011 / 6.616385 |
| Retcon | 94.006800% | 91.669406% | +2.337395 | 20/20 / 20/20 | 2.803340 / 0.446923 |
| Walls | 92.660550% | 96.937574% | −4.277023 | 5/5 / 5/5 | N/A / N/A |
| AISurvivor | 89.820359% | 82.392473% | +7.427886 | 5/5 / 5/5 | N/A / N/A |

All registered games, in order:

| ID | Arm / opponent | Share | First places |
|---|---|---:|---:|
| brace-s01 | Keel / Diamond | 46.613833% | 7/10 |
| brace-s02 | Brace / Diamond | 70.252101% | 9/10 |
| brace-s03 | Brace / Diamond | 47.084233% | 8/10 |
| brace-s04 | Keel / Diamond | 36.709721% | 6/10 |
| brace-s05 | Keel / Diamond | 49.261849% | 7/10 |
| brace-s06 | Brace / Diamond | 37.607450% | 5/10 |
| brace-s07 | Keel / Retcon | 91.985428% | 10/10 |
| brace-s08 | Brace / Retcon | 95.989061% | 10/10 |
| brace-s09 | Brace / Retcon | 92.024540% | 10/10 |
| brace-s10 | Keel / Retcon | 91.353383% | 10/10 |
| brace-s11 | Keel / Walls | 96.937574% | 5/5 |
| brace-s12 | Brace / Walls | 92.660550% | 5/5 |
| brace-s13 | Keel / AISurvivor | 82.392473% | 5/5 |
| brace-s14 | Brace / AISurvivor | 89.820359% | 5/5 |

The Diamond mean, delta and round-win thresholds passed; the consistency requirement did not. Retcon passed mean≥90%, delta≥+1pp and≥16/20 first places. Both transfer opponents passed delta≥−5pp and≥4/5 first places. Passing the primary target or the other guards cannot override the failed Diamond consistency guard.

## Damage, firing, timing and treatment exposure

| Opponent / arm | Damage dealt / taken | Hits / shots | Hit rate | Shots / 1000 turns | Mean power |
|---|---:|---:|---:|---:|---:|
| Diamond / Brace | 546.811397 / 1443.371294 | 707/8055 | 8.777157% | 70.637447 | 0.209601 |
| Diamond / Keel | 534.933116 / 1560.048516 | 640/6761 | 9.466055% | 69.840712 | 0.231784 |
| Retcon / Brace | 754.400000 / 120.080000 | 200/1332 | 15.015015% | 64.890145 | 0.856992 |
| Retcon / Keel | 665.660000 / 179.760000 | 188/1292 | 14.551084% | 65.114404 | 0.899941 |
| Walls / Brace | 435.000000 / 60.000000 | 70/110 | 63.636364% | 67.692308 | 1.139318 |
| Walls / Keel | 430.100000 / 20.000000 | 65/116 | 56.034483% | 67.796610 | 1.123707 |
| AIS / Brace | 254.300000 / 68.000000 | 109/286 | 38.111888% | 70.512821 | 0.629312 |
| AIS / Keel | 269.100000 / 131.200000 | 101/300 | 33.666667% | 70.688030 | 0.678278 |

Firing rates use the recorded observer-turn denominator, as retained in [group metrics](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/independent-outcome/group-metrics.csv). Complete own/enemy firing, hit, power, damage, contact and per-round metrics are in that CSV, [all 14 game metrics](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/independent-outcome/all-14-game-metrics.csv) and the original recordings. No opponent private source or telemetry was interpreted for the audit.

Both arms had **0 own skipped turns**, with the registered integer own run-loop maxMs below75 in every round; the global maximum was21ms. This measure does not cover full callback/system latency or private Diamond skips. Candidate counter bounds and inequalities passed. Retcon had13 eligible corrected scans, all in brace-s09; brace-s08 had0. Walls had6. Every candidate Diamond and AIS game had0 eligible scans. All three candidate Diamond games also had0 own bot contacts and0 RAM treatment activation, so their score difference cannot be attributed to this correction. Retcon exposure establishes that the mechanism occurred, not a causal score gain from two unpaired games.

## Measured source, artifacts and actual runtime

The sole measured source was `e659647b3ae6366008ef666bf338ea3ee446ee71`, tree `9094510ceb6b234ed43af1b1e8452779e771cde0`. Effective [freeze-v2](../../benchmarks/keel-brace-v1/freeze-v2.json) SHA256 is `f201fd60fb39cbe2d7ff03adf12f5e95029920bbfc6b2353ed7f2114fcce401f` and binds215 assets. Original v1 freeze and zero-game preparation failure41947040130 remain historical evidence, not measured games.

| Artifact | SHA256 |
|---|---|
| KeelBrace JAR | `8a1f1c3365846b7fb272a5a1885b65fad4f5f6ace29cc770fed04f34287d7eb9` |
| Immutable original Keel JAR | `77b8a5c3114641c12a94f36c83083d27c419416ee691d725fd19f0284a69db6f` |
| Production AISurvivor JAR | `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800` |
| Screen profile | `b670725af5deb0e711dfa51f4ead24c24066ff65e29d5a1675977966438119f8` |
| Registered games | `9526d1a75d7ee99ac3d52e5affbf2487823b696c3d9b1e0f0497b1d42d65b3a3` |

[Run41951522797](https://github.com/stephanelee/robocode/actions/runs/41951522797), attempt1/job126996333991, ran on `ubuntu-26.04`, x86_64, qualified CaseB `native-20261101.186.1_env-20261101.186.1`, boot `47c77544-3ae0-4795-af58-66a4c3382242`. Actual Java17 Temurin17.0.21+9 built/tested and checked committed-JAR parity before Java25 Temurin25.0.5+8 launched Retcon; native Python was3.13.17. Runner1.4.0, original DiamondPort1.8.28, Retcon and official Walls pins, package/native identities, full commands and source bytes are retained in the frozen profile/original evidence. Local offline arithmetic used Python3.15.1/macOS arm64 and is recorded separately from battle runtime.

Actual same-worker reasoned report6647642341 and approval6647660422 preceded execution. The immutable annotated screen receipt `3822b5e3a3f9d03651aeb6fc7e4dc1f3f3ca1046` remains consumed. All15 image revalidations and the mandatory completed same-job provider header were independently reconciled. The early actual HTTP404 was preserved, not relabeled a200. The full completed job log is103464 bytes/SHA256 `1b6010dc470ae1fade29b88d6eadbba7e90b21474a6b10de30f9c032fe7ae9ea`. The [final image review](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/final-image-review/final-image-review.md) is a scoped custody/runtime finding, not canonical VM attestation, historical Helm rescue or numerical acceptance.

## Two reader defects and qualified read-only recovery

The original whole frozen reader returned **PROVENANCE-INCOMPLETE with raw REJECT**, not an overall pass. Its numeric decoder and separately preregistered observer arithmetic agree on all2630 compared game/per-round fields with0 differences, and the original canonical decision is reject. Both metadata failures are retained:

1. The ZIP contains byte-identical stage/profile.json and provenance/profile.json (`b670…`). The reader selected the duplicate root profile, then searched its parent for anchors that exist only in actual provenance/. A read-only selector later routed just that lookup to the original provenance/profile.json; it did not edit or normalize any original byte.
2. With the correct original profile selected, the image reader treated `checkedAtUTC` as admission completion. Producer source records it at check start, before capture. The second failure remains preserved. Separate original-byte chronology verified every capture completed7.068823–8.644107seconds before actual immutable receipt/child-attempt admission, while0.078816–0.084616seconds after check start.

The runtime auditor independently reviewed these same15 rows and source semantics. [Recovery review and original temporal proof](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/independent-outcome/metadata-recovery-review/) document the narrow qualification. No source, ZIP, timestamp, raw decision, gate or admission criterion was changed; no game was replayed. These defects are not new invalid rounds, a numerical waiver or a claim that the whole frozen reader passed. Any reader repair requires a future version and prospective review. The independent reviewer’s [separate attributed reconciliation](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/independent-outcome/actual-reviewer-image-reconciliation-repository.json) preserves the frozen machine author template without falsely attributing this review to it.

## Complete custody and disposition

- [Pregame/postapproval custody](../../benchmarks/keel-brace-v1/screen-provenance-41951522797/) is immutable history at `bb11a50e2ef299b48ad67a25aa26fb6897c375ab` (initial custody6143fb0), including exact source/runtime comments, receipt and original3946762-byte provenance ZIP.
- [Complete original final evidence](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/) was published at09d666f, with independent outcome/final-image append at `0a86be51b1136106e4fcc48b55fd58180f2558b4`. Original artifact13383827989 is108899016bytes/SHA256 `3eb14b072617c3b64d364988c9db8d224cb47e4160214bd14d6e6ef629ca02d8`; companion13383922872 is1366bytes/SHA256 `ffcee88194c9599be4930058853481c85ee0ee91dd5d96d02f396b6343ef89db`. All1765+2 members and14 whole recording.gz files are retained, with exact ZIP parts≤2MiB, member hashes/CRC, command/time/stderr/API receipts and reassembly verifier.
- The [independent outcome manifest](../../benchmarks/keel-brace-v1/screen-evidence-41951522797/independent-outcome/outcome-evidence-manifest.json) preserves52 original outputs plus itself, including both reader failures, raw arithmetic and scoped recovery proofs. Its SHA256 is `f087ccb4d635c84b0587c640268ed3a84010ea8f3047cc87524c312e5787d29b`.

**Reject and retain the failed experiment.** The higher mean share is insufficient independent evidence that Brace beats Diamond, and no general strength claim follows. Keep the baseline and production artifacts unchanged; the failed candidate and consumed receipt remain evidence. Any next improvement needs a new hypothesis, prospective protocol, independent pre-execution review and separately budgeted fresh games. This PR integrates evidence/history and records rejection; it does not promote KeelBrace.

The original raw `verify.py` checks the exact09d666f directory inventory. Run it at that immutable raw commit (or with only its manifest-listed files staged); later independent-audit/publication subdirectories have separate custody manifests. The original verifier and manifest are retained unchanged.
