# KeelHelm 0.2.1 melee — measured top-four comparison

**Accept the measured melee improvement for this frozen cohort.** All registered gates pass; this is a bounded engineering result, not a significance or general-strength claim.

The new approach keeps legal scans separate by opponent ID, sweeps radar, selects a fresh target with hysteresis, searches minimum-risk movement paths and uses bounded circular intercept fire. At one remaining opponent it clears transient tracks/waves and resumes duel surfing. From-start 1v1 retains the original duel policy. The completed tournament remains on frozen KeelHelm 0.1.0.

## Comparison and results

Four fresh 20-round games per arm used exactly KeelHelm, Diamond Port, Keel and Prism, with rotated launch order and cold processes. Candidate and control never shared a game. Engine placement was not paired or seeded by the bot seed. Primary metric is equal-game mean official score share.

| Scenario | Original 0.1.0 | Candidate 0.2.1 | Delta |
| --- | ---: | ---: | ---: |
| Top-four melee | 13.19% | 26.98% | +13.79 pp |
| Retcon 1v1 | 94.40% | 92.92% | -1.48 pp |
| Walls 1v1 | 94.50% | 95.02% | +0.51 pp |
| SpinBot 1v1 | 100.00% | 100.00% | +0.00 pp |

Melee first places rose from **7/80 to 27/80 rounds**. Candidate game shares ranged from 22.50% to 35.69%; control ranged from 8.76% to 16.41%. No game from the invalid first attempt is included.

| Launch rotation | Control score | Share | First places | Candidate score | Share | First places |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,494 | 12.50% | 1 | 4,139 | 35.69% | 11 |
| 2 | 1,818 | 15.10% | 3 | 3,023 | 27.10% | 7 |
| 3 | 1,977 | 16.41% | 3 | 2,581 | 22.65% | 6 |
| 4 | 1,058 | 8.76% | 0 | 2,508 | 22.50% | 3 |

Aggregate standings in the four **candidate** melee games:

| Bot | Official total score | First places |
| --- | ---: | ---: |
| Diamond Port | 16,160 | 41 |
| Prism | 14,092 | 9 |
| KeelHelm | 12,251 | 27 |
| Keel | 2,794 | 3 |

KeelHelm remains behind Diamond Port and Prism on aggregate score. The intervention combines tracking, radar, movement and firing changes; this comparison does not isolate their individual contributions.

## Decision, validation and budget

All exact-ratio gates pass: candidate melee share >=20%, melee gain >=5 pp and each 1v1 guard no worse than -5 pp. Retcon first places were 20/20 for each arm; Walls and SpinBot were 5/5 for each arm. The smaller Retcon score-share change passes the registered tolerance and is not evidence that every duel result is identical.

All sixteen games / 220 rounds completed: eight melee games (160 rounds), four Retcon games (40), four sample games (20). There were zero own skipped turns and maximum logged own loop time was 7 ms, below the 75 ms guard. That timer excludes event-handler work and is not a latency guarantee. Independent audit validates all recording CRCs, exact rosters/setup, contiguous ticks, round endpoints, final official scores, lifecycle markers, 7,007 frozen hashes and 640 staged input-file copies.

The prior 0.2.0 comparison remains invalid: one valid control plus one lifecycle-broken candidate consumed 40 physical rounds, with fourteen games unplayed. Its earlier folder collision consumed zero rounds. The real callback fixture reproduced the unavailable-tick exception; 0.2.1 resets state without tick getters and activates from live events. No v1 game was resumed, replaced or pooled. Combined project budget consumed: **260 physical rounds**. [Invalid attempt and diagnosis](keelhelm-melee-v1-invalid.md).

## Source, runtimes and artifacts

Acquisition source: 24ae1bcdd221f52a19188550c83bd339921125f4. Candidate JAR SHA256: 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Control JAR: 60ef5b1d33c3deb7100d1e66135f311d4b7b6fd5651cd1146effb899b9ea08c2. Full source/JAR hashes, per-game schedule, opponent pins, versions and package inventory are in the public provenance record.

Java 17 builds/tests and JAR entry parity passed for candidate (23 tests) and control (15 tests); three harness contracts passed. Games ran sequentially on minimind with Temurin 25.0.5+8 and the tournament Python 3.12.16 / Bot API 1.3.1 guest environment. Runner 1.4.0 and Java adapter were unchanged. Official samples use API 1.4.0; Retcon uses its frozen 1.3.1 source launcher under JDK 25. No gameplay ran in GitHub Actions.

Production AISurvivor source/JAR remains unchanged and parity passes (822 entries). Its full local Java 17 suite had 89 tests, five existing exact-bit golden replay failures and one skip on arm64. Initial missing sparse fixtures were restored; both failed-build logs are retained. These failures were disclosed before the new comparison and were not repaired or used to relax a gameplay gate.

All original recordings, isolated bot copies, source/JAR snapshots, telemetry, commands, staging failure, callback reproduction, runtime freeze and independent audit are retained. Local archives under local-evidence/:

- keelhelm-melee-v1/invalid-attempts.tar.gz: 80033461 bytes; SHA256 f77536ea98046d81a0163dd4ea3bedf779bef1ee4563c7305d88cb0ac6f194cc.
- keelhelm-melee-v2/full-artifacts.tar.gz: 234796250 bytes; SHA256 d9010f018fa60c4061fe5421014bfb2bf9b24adc69ced2055c2967b777ac2fc0.

Independent audit source scripts are retained alongside the v2 archive; their hashes are included in the public artifact proofs.

[Full per-game scores/telemetry](../../benchmarks/keelhelm-melee-v2/results.json), [source/runtime provenance](../../benchmarks/keelhelm-melee-v2/provenance.json), [independent raw audit](../../benchmarks/keelhelm-melee-v2/independent-audit.json), [archive proofs](../../benchmarks/keelhelm-melee-v2/artifact-proofs.json).

## Running a future comparison

The single-use helper is tools/keelhelm_melee.py; --help lists the inputs. Prepare a fresh evidence directory with control-source/ containing the unchanged KeelHelm 0.1.0 project from parent 2bd4afb, and retain Java 17 build/tests/parity. FROZEN_SOURCE is the prepared tournament source/runtime snapshot; PINNED_EVENT supplies its entry-manifests.json and shared-inputs.json.

    python3 tools/keelhelm_melee.py --source "$FROZEN_SOURCE" --evidence "$MELEE_EVIDENCE" --prepare --pins "$PINNED_EVENT" --java "$JAVA25" --python "$BOT_PYTHON"
    python3 tools/keelhelm_melee.py --source "$FROZEN_SOURCE" --evidence "$MELEE_EVIDENCE" --approval-sha "$REVIEWED_FREEZE_SHA" --stage-only
    # Independently review the exact source/JAR/runtime freeze and all staged rosters before games.
    python3 tools/keelhelm_melee.py --source "$FROZEN_SOURCE" --evidence "$MELEE_EVIDENCE" --approval-sha "$REVIEWED_FREEZE_SHA"

Existing attempted output cannot be resumed or overwritten. A future execution needs its own prospective budget/registration and review; the commands do not authorize reusing this experiment. All public site links remain external configuration; no production hostname is introduced.

[V2 prospective protocol](keelhelm-melee-v2-protocol.md), [issue #123](https://github.com/stephanelee/robocode/issues/123), [PR #124](https://github.com/stephanelee/robocode/pull/124).
