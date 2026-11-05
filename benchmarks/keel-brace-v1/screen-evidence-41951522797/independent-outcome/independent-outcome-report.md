# KeelBrace v2 screen: independent REJECT

**Reject KeelBrace for this registered experiment. Do not run confirmation, rerun this candidate, promote it, or change the criteria.** All 14 registered games and 120 physical rounds completed. The Diamond consistency gate failed: only **1 of 3** candidate games exceeded 50% score share; the prospective requirement was at least 2 of 3. All other registered numerical/own-timing/exposure gates passed. The original canonical decision is `reject`; independent raw arithmetic agrees.

The frozen whole automated reader did **not** return an overall pass: it returned `PROVENANCE-INCOMPLETE` with raw disposition `REJECT` because of two reader/metadata integration defects described below. Its original outputs are preserved verbatim. Actual original-byte provenance, image and before-admission chronology were separately verified and independently reviewed; this report does not hide those failures, repair inputs or claim the whole frozen reader passed.

Reviewer: `/root/brace_independent_protocol_review`, independent of candidate/controller/decoder implementation. Prospective audit plan was fixed before outcomes at SHA256 `b23300a5a1fc2a55a4a5e98fe3b17dfd82b3d96bcd0d152564df12a44046f0f6`; supplemental observer arithmetic source was fixed before outcomes at `a4c727202209910aa27fa314ae47ff43c9f14bea6c2d6f975d3256d320789617`. No game, source, label, receipt or merge mutation by this reviewer.

## Registered identity and budget

Run [41951522797](https://github.com/stephanelee/robocode/actions/runs/41951522797), attempt 1, actual job 126996333991. Exact source `e659647b3ae6366008ef666bf338ea3ee446ee71`; effective freeze-v2 `f201fd60fb39cbe2d7ff03adf12f5e95029920bbfc6b2353ed7f2114fcce401f`; protocol `cd1ec9f5bba78b47f487dc829b59d99ed52953266f15c1c3b5407186b2ad20a8`; games registry `9526d1a75d7ee99ac3d52e5affbf2487823b696c3d9b1e0f0497b1d42d65b3a3`; DATA-v2 `fb30ed512138d4f1c862c7999b9f7c96dcae4d6abbecdccb59664473754011cd`. Candidate KeelBrace JAR `8a1f1c3365846b7fb272a5a1885b65fad4f5f6ace29cc770fed04f34287d7eb9`, immutable Keel JAR `77b8a5c3114641c12a94f36c83083d27c419416ee691d725fd19f0284a69db6f`, production AISurvivor JAR `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800`.

Fixed screen budget: 14 serial cold games / 120 rounds, all consumed once; 0 unplayed IDs, 0 raw invalid games/rounds, 0 acquisition failures. No replacements or extra games. The conditional 300-round confirmation was not opened and is not authorized for this failed candidate. Permanent screen receipt SHA `3822b5e3a3f9d03651aeb6fc7e4dc1f3f3ca1046` remains consumed; do not reset it.

## All opponents, controls and variation

Shares are equal-weight means of independent cold games, not pooled rounds or historical results. SD is sample standard deviation of game shares in percentage points; one-game sample SD is undefined.

| Opponent | Candidate mean share | Keel mean share | Difference (pp) | Candidate wins | Keel wins | Candidate / Keel game SD (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Diamond Port | 51.647928% | 44.195134% | +7.452794 | 22/30 | 20/30 | 16.794011 / 6.616385 |
| Retcon | 94.006800% | 91.669406% | +2.337395 | 20/20 | 20/20 | 2.803340 / 0.446923 |
| Walls | 92.660550% | 96.937574% | -4.277023 | 5/5 | 5/5 | N/A / N/A |
| AISurvivor | 89.820359% | 82.392473% | +7.427886 | 5/5 | 5/5 | N/A / N/A |

All 14 games, in registered order:

| ID | Arm | Opponent | Score share | First places |
|---|---|---|---:|---:|
| brace-s01 | Keel | Diamond | 46.613833% | 7/10 |
| brace-s02 | Brace | Diamond | 70.252101% | 9/10 |
| brace-s03 | Brace | Diamond | 47.084233% | 8/10 |
| brace-s04 | Keel | Diamond | 36.709721% | 6/10 |
| brace-s05 | Keel | Diamond | 49.261849% | 7/10 |
| brace-s06 | Brace | Diamond | 37.607450% | 5/10 |
| brace-s07 | Keel | Retcon | 91.985428% | 10/10 |
| brace-s08 | Brace | Retcon | 95.989061% | 10/10 |
| brace-s09 | Brace | Retcon | 92.024540% | 10/10 |
| brace-s10 | Keel | Retcon | 91.353383% | 10/10 |
| brace-s11 | Keel | Walls | 96.937574% | 5/5 |
| brace-s12 | Brace | Walls | 92.660550% | 5/5 |
| brace-s13 | Keel | AISurvivor | 82.392473% | 5/5 |
| brace-s14 | Brace | AISurvivor | 89.820359% | 5/5 |

The sole candidate Diamond game above 50% does not satisfy the predeclared consistency guard. Neither the higher mean nor the Retcon result can override it.

## Damage, firing and exposure

| Opponent / arm | Damage dealt / taken | Hits / shots | Hit rate | Shots per 1000 turns | Mean shot power |
|---|---:|---:|---:|---:|---:|
| Diamond / Brace | 546.811397 / 1443.371294 | 707/8055 | 8.777157% | 70.637447 | 0.209601 |
| Diamond / Keel | 534.933116 / 1560.048516 | 640/6761 | 9.466055% | 69.840712 | 0.231784 |
| Retcon / Brace | 754.400000 / 120.080000 | 200/1332 | 15.015015% | 64.890145 | 0.856992 |
| Retcon / Keel | 665.660000 / 179.760000 | 188/1292 | 14.551084% | 65.114404 | 0.899941 |
| Walls / Brace | 435.000000 / 60.000000 | 70/110 | 63.636364% | 67.692308 | 1.139318 |
| Walls / Keel | 430.100000 / 20.000000 | 65/116 | 56.034483% | 67.796610 | 1.123707 |
| AIS / Brace | 254.300000 / 68.000000 | 109/286 | 38.111888% | 70.512821 | 0.629312 |
| AIS / Keel | 269.100000 / 131.200000 | 101/300 | 33.666667% | 70.688030 | 0.678278 |

Complete own/enemy firing/hit/power/damage/contact/timing and variation metrics are retained in `group-metrics.csv/json`, `all-14-game-metrics.csv`, and every raw round in `manual-raw-results.json` / `frozen-independent-audit.json`. No opponent private source or telemetry was inspected.

Candidate Retcon exposure is 13 eligible corrected scans, all in brace-s09. Its five round-local aggregate diagnostics are each 13 (raw events, eligible correction events/scans, changed power/gate scans); brace-s08 has 0. All counter bounds and inequalities pass. Walls has 6 each; every candidate Diamond and AIS game has 0 each. **All three candidate Diamond games had 0 own bot contacts and 0 RAM treatment activation**, so their mean difference cannot be credited to the RAM correction. Retcon activation demonstrates mechanism exposure, not causal score attribution in two unpaired games. This screen does not establish general strength or that Brace beats Diamond.

## Every screen gate

- All 14 original games / 120 rounds, cold processes/directories, exact source/commands/IDs/order and original canonical acquisition: pass. No imported Keel model; AIS cold TRAIN telemetry/model completion checked.
- Both arms: 0 own skipped turns; integer maxMs <75 throughout. Global maximum 21 ms. This is the registered own run-loop timing measure, not full callback/system latency or Diamond private skip visibility.
- Candidate counter limits/invariants and stage-local Retcon eligible scans >0: pass (13).
- Diamond mean >50%, contemporaneous delta >=-3 pp, >=16/30 first places: pass.
- **Diamond >=2/3 candidate games above 50%: FAIL (1/3).**
- Retcon mean >=90%, delta >=+1 pp, >=16/20 first places: pass.
- Walls and AIS each delta >=-5 pp and >=4/5 first places: pass.

Only the fixed screen population is used. No historical Keel/Helm games, pooled confirmation, contact-based selection or extra battles were included.

## Raw custody, canonical failure handling and independent comparison

Original final evidence artifact 13383827989: 108899016 bytes, SHA256 `3eb14b072617c3b64d364988c9db8d224cb47e4160214bd14d6e6ef629ca02d8`. Separate companion 13383922872: 1366 bytes, SHA256 `ffcee88194c9599be4930058853481c85ee0ee91dd5d96d02f396b6343ef89db`. The packager performed the sole downloads; this reviewer streamed the shared originals, independently verified whole ZIP digests, every original member's complete bytes/size/CRC/SHA/Git-blob identity/metadata, all 14 whole compressed recordings and both companion files. Evidence ZIP has 1765 regular members, 129218801 uncompressed bytes. Nothing was selected or re-ZIPped. Complete originals are preserved at raw evidence commit `09d666fbd66807f18ed5f44e065eb05333bcb608` under `benchmarks/keel-brace-v1/screen-evidence-41951522797/`.

Frozen independent decoder `8cef410dc20c0af7b317cf0de62c8cf65d65d10b559ea44145e00c69710e2cd5` and separate pre-outcome observer/marker arithmetic agree on all 2630 compared game/per-round fields, with 0 differences. Frozen authority, storage, upstream applicability and canonical-reconciliation checks pass. Original raw decision `reject`, SHA256 `bd1602bdf4c507fe3b42d799d1aa7e58659380bdd737c62421ce7f56e592f84c`, has 120 observed/completed rounds, no invalid acquisition rows, no unplayed IDs or retained stage/game failure. It was never rewritten to make this audit agree.

## Frozen reader integration failures, preserved and separately verified

1. The original ZIP contains stage/profile.json and provenance/profile.json with **identical** original bytes, SHA256 `b670725af5deb0e711dfa51f4ead24c24066ff65e29d5a1675977966438119f8`. Freeze/origin/source anchors live only under actual provenance/. The frozen reader picks the duplicate root profile and then looks for the wrong parent freeze. Original whole output remains `PROVENANCE-INCOMPLETE`, raw `REJECT`.
2. An explicit read-only path selector maps only the profile lookup directly to actual byte-identical provenance/profile.json. All other lookups still use the original ZIP. Frozen provenance verification then passes, but image verification fails because it treats `checkedAtUTC` as an admission-completion deadline. Producer source records this field at check **start**, before image capture. The profile-only second failure is also retained verbatim.

No original file, timestamp, raw decision, source, numeric gate or admission criterion was edited. Separate explicit fact proof verifies all 15 original image captures' complete frozen raw/stat/hash/cap/case/profile identities and chronology: capture completion is 7.068823–8.644107 seconds **before** the actual immutable receipt/child-attempt admission, while naturally 0.078816–0.084616 seconds after the check-start field. This meets the actual registered before-admission method. The same actual rows were independently rechecked by `/root/runner_image_provenance_audit`; scoped recovery review SHA256 `9b23d4414b6b87a50bf28bbe981854fa52535edc740e764ee64bf237d12d4598`, detailed review `b1abc8066e9fdbc33e9d024b2e8a1264f2008194216d0205d2078133905dd8b7`, full original timing proof `bca0f002b5120f7c7a95d190b0b23760c8c205d58c9a3c07eca5fe39c330ec36`. These are metadata/custody findings, not new invalid rounds, an overall frozen reader pass or a numerical waiver. Fix reader/source integration prospectively in a later version, retaining both failures.

## Actual runtime/provider reconciliation and attribution

Actual run/jobs APIs independently confirm sole attempt 1 at source E659, registered job completed success at 2026-11-05T02:50:01Z (run update 02:50:02Z). Boot `47c77544-3ae0-4795-af58-66a4c3382242`, qualified case `native-20261101.186.1_env-20261101.186.1`; both native files and both environment sources match this case throughout all 15 phase revalidations. Original pregame 260-member provenance artifact 13380953306 is byte-identical to its final preserved capsule; source 215-asset/freeze/JAR/commands and actual Java17-before-Java25/Python3.13.17 execution graph retain their independent pregame approval. No canonical VM attestation or old Helm rescue is claimed.

Actual early correctly bound in-progress job HTTP404 evidence is retained, attributed to the actual runtime collector. The mandatory **completed same-job** full original provider startup log was independently fetched and parsed: 103464 bytes, SHA256 `1b6010dc470ae1fade29b88d6eadbba7e90b21474a6b10de30f9c032fe7ae9ea`, exact case-B native four strings before workflow output. Original final run API 12836 bytes / SHA `832a2109d66d744f8779a306149c336bcc62e0569eb34220158c678c2002f77a`; jobs API 4807 bytes / SHA `cc3e398b72c3812fe90d817835bafdd6437caa9086c0071f54154df6ff081ff8`. Actual step UTC/source/job/attempt bindings and finalization/review chronology checked. Runtime auditor's separate final-image review `ca289f8f4275dbff440264ae396209f1bd21dbf1bba5fe779f5688c848506189` independently agrees.

The frozen machine template hardcodes decoder-author reviewerTask. It is preserved as machine analysis, not falsely credited with this actual audit. The separate `actual-reviewer-image-reconciliation-repository.json` is independently authored from the checked original provider/hash/case facts and correctly names `/root/brace_independent_protocol_review`; it uses exact DATA-v2 fields and retained repository-relative full API/log references. Runtime auditor's own separately attributed reconciliation is preserved too. No raw decision or frozen machine output was rewritten for attribution.

Local offline audit runtime is Python3.15.1/macOS arm64, recorded separately from actual pinned battle runtime. Commands, original outputs, supplementary source, provider tool provenance and all file hashes are retained. The actual machine-reading limitations are explicit; no game was replayed to remedy an analysis or custody issue.

## Decision

**SCREEN-REJECT.** Preserve failed candidate and all evidence; keep original Keel baseline and production AISurvivor unchanged. Do not open confirmation or promote Brace. Any next improvement needs a new prospective hypothesis/protocol, independent pre-execution review and fresh separately budgeted controls, not a rerun or retuned criterion for this consumed screen. Publish these results/raw evidence and update the improvement ledger before marking the experiment complete.
