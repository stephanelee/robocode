# Independent KeelHelm confirmation outcome recovery review

**Decision: REJECT the registered competitive improvement; retain immutable original Keel0.1.0.** Confirmation acquisition and evidence are valid. The failed criterion is candidate-minus-baseline Diamond score share >=2 percentage points; actual difference is -2.31773022pp. This is not a runtime, storage or invalid-round rejection. No promotion, rerun, replacement battle or retrospective gate change follows.

Issue107 / [PR108](https://github.com/stephanelee/robocode/pull/108). Original protocol `docs/improvements/keel-helm-v1.md` and registry `benchmarks/keel-helm-v1/games.json` remain unchanged.

## Actual reviewer and recovered custody

This report is authored by the separate post-crash recovery reviewer `/root/helm_final_review_recovery`, not by the prior outcome task `/root/keel_helm_outcome_audit` or runtime task `/root/stop24_acquisition_audit`. Parent explicitly assigned read-only outcome recovery after the user freed disk space. No candidate/harness reducer import, gameplay implementation, game, dispatch, receipt, branch, push, merge or public comment was performed by this reviewer.

The original finalized artifact13365609151 ZIP is still exactly351,880,028bytes / SHA256 `bcbd2e27ff64dc585f7341aa32ae506ea9819903c1893db8a723ba91c7036ad6`. Fresh recovery verification streamed every one of2,194 unique regular members, checked safety/CRC/SHA against the prior complete stream manifest, and verified all18 whole battle gzip identities (326,725,805bytes). No recompression or original-byte mutation occurred.

One previously removed, reproducible c01 gzip extraction was restored byte-for-byte from that intact ZIP solely for reanalysis:25,337,644bytes/SHA256 `c09a5a51649cd663c0d7cb61497cd31795d46c772ffa97080cd31d2d4265e857`. Its restore receipt is `keel-helm-confirmation-recovery-restored-derived.json`. This did not execute another battle.

Freshly reran the unchanged preexisting stdlib-only independent reducer (`audit.py` SHA256 `1a1c8e3de7a6f05e8b2cac987665d3b0ce7d3c9772e502b427df2dd615f1984a`) against all18 recordings. Its NEW `-recovery-independent-audit.json` equals the prior completed independent result exactly. Freshly reran the prior provider/authority/input audit with only its output destination changed to a NEW recovery filename; all2,217checks, including1,890staged-input hashes read directly from the original ZIP, pass and the complete output equals its preserved predecessor. The output-redirected script is separately retained and hashed; its metric/binding logic was not changed.

The prior6,150field comparison against the immutable raw decision records zero mismatches; the original raw decision SHA256 is `89b5255b498df05573e84352492d0319948848f4caa3f8ff6f4522d69cf9d8c2` and literal decision is `not confirmed`. The prior conditional-screen audit independently retains301files/14whole screen recordings and passing eligibility; screen data is not pooled with confirmation.

## Fixed budget and results

All18 registered fresh cold games/300physical rounds completed valid, with0invalid/partial,0unplayed and no replacements. Screen14games/120rounds remains a distinct earlier passing stage. Whole experiment:32games/420physical rounds consumed, exactly its maximum. The positive screen cannot override the fresh failed confirmation.

| Opponent | Keel mean share | KeelHelm mean share | Candidate minus baseline | Official first places Keel / KeelHelm |
|---|---:|---:|---:|---:|
| Diamond | 56.43873576% | 54.12100553% | -2.31773022pp | 101/120 / 95/120 |
| Retcon | 93.08497595% | 93.74743222% | +0.66245627pp | 30/30 / 30/30 |

Only the frozen `diamondDeltaAtLeast2pp` gate fails. Candidate Diamond mean is>50%,95/120wins exceeds60, and5/6games exceed50%; Retcon mean>=90%, delta>=-2pp and30/30wins all pass. Both arms have zero own skips and maximum integer per-round maxMs18, below75. All19live authority checks (before receipt and before every game) match the original accepted authority. All18owned teardowns are normal. Conservative actual-child duration upper bounds are all<1,800seconds (maximum1,107.798686seconds), and the minimum remaining job margin before a new child was9,021.840623seconds, above the fixed2,100-second requirement.

| Arm/opponent | Cold games | Share min / max | Sample SD | Games>50% |
|---|---:|---:|---:|---:|
| baseline/diamond | 6 | 47.564576% / 64.213396% | 6.511949pp | 5/6 |
| baseline/retcon | 3 | 87.688642% / 98.181818% | 5.252995pp | 3/3 |
| candidate/diamond | 6 | 48.591292% / 57.740113% | 3.826653pp | 5/6 |
| candidate/retcon | 3 | 92.523364% / 95.211268% | 1.359898pp | 3/3 |

## Damage, firing, contacts and exposure

Rates below pool counted events/turns only; they do not replace the equal-cold-game score-share comparison. Full per-game/per-round counts, energy/power mixes and timing are in the retained CSV/JSON.

| Arm/opponent | Shots / hits | Hit rate | Incoming shots / hits | Incoming hit rate | Damage dealt / taken | Shots / 1,000 turns | Bot / wall contacts |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline/diamond | 33069 / 2964 | 8.963077% | 36651 / 3841 | 10.479932% | 2343.345375 / 5638.924151 | 70.588763 | 1 / 54 |
| baseline/retcon | 1931 / 287 | 14.862765% | 1847 / 35 | 1.894965% | 1088.880000 / 200.480000 | 64.448301 | 36 / 15 |
| candidate/diamond | 32751 / 2881 | 8.796678% | 36261 / 3788 | 10.446485% | 2289.316792 / 5626.522893 | 70.628042 | 2 / 65 |
| candidate/retcon | 2026 / 295 | 14.560711% | 1964 / 44 | 2.240326% | 1065.780000 / 194.560000 | 64.861058 | 7 / 19 |

| Arm/opponent | Shot energy | Opponent shot energy | Token power0.1 shots / all | Fixed power1.2 shots |
|---|---:|---:|---:|---:|
| baseline/diamond | 6787.147402 | 13270.277581 | 29882 / 33069 | 3084 |
| baseline/retcon | 1760.910000 | 2028.800000 | 471 / 1931 | 1357 |
| candidate/diamond | 6683.490296 | 13187.352777 | 29584 / 32751 | 3045 |
| candidate/retcon | 1766.909625 | 2088.160000 | 538 / 2026 | 1334 |

Both Diamond arms lost the bullet-damage exchange; there is no claim that Helm improved aiming or the registered comparative score result. Baseline Retcon had36official bot-contact events versus candidate7; Diamond had1versus2. The unchanged ram-energy bookkeeping limitation therefore has actual exposure, but contact counts neither prove harmful phantom-wave frequency nor isolate a causal effect. Do not remove these games, retune Helm, or adopt a new correction from this report. Any separate candidate needs prospective registration and independent review.

| Candidate opponent | Stopped rollout steps | Heading-changed steps | Second-wave evaluations | Executed stop-now |
|---|---:|---:|---:|---:|
| diamond | 37547917 | 7324957 | 6202194 | 43485 |
| retcon | 3122934 | 558485 | 295519 | 3193 |

Baseline exposure counters are N/A by registration. Candidate counts demonstrate activation only; they do not count final choices changed or prove benefit.

## Actual runtime and authority

Sole [confirmation run41889609469](https://github.com/stephanelee/robocode/actions/runs/41889609469), attempt1, completed/success21:51:44UTC at exact executed head `f2a25d3309062a362a13902329de405bf63094a7`. Fresh current provider/ref reads corroborate that state and fixed annotated receipt `42b194e58f8b9b4dc3a477827658157a8582ad72`. The original90-asset source freeze SHA256 is `ef876a82069d5c73baf7171aa95cf4058e3bc4a196f46d51bdf5ff1cf0b81a9b`; this recovery freshly rehashed all90 local frozen assets, including original Keel, candidate and production AISurvivor.

Keel control JAR SHA256 `77b8a5c3114641c12a94f36c83083d27c419416ee691d725fd19f0284a69db6f`; KeelHelm JAR `60ef5b1d33c3deb7100d1e66135f311d4b7b6fd5651cd1146effb899b9ea08c2`. Runtime Ubuntu26.04/x86_64 image20261026.180.1, exact Temurin17.0.21+9 source/tests/parity first, then25.0.5+8-LTS for Retcon/battles, CPython3.13.17 and pinned TankRoyale1.4.0/DiamondPort1.8.28/Retcon0.2.0 graph. New boot `1de439b6-c262-4a5c-b6a3-5c61934b2df2`; workspace `/home/runner/work/robocode/robocode`; full profile SHA256 `1c46c0b53b525a7266b31e10294ee28e3b068144d5040837fc398bd2b6b452ea`.

Prospective actual independent assignment6638977373 at18:41:00UTC precedes run creation18:41:21UTC. Finalized pregame artifact13352733399:3,316,037bytes/SHA256 `fdaa95e8bd0869e25f45d87467e7f554bbb4d59f761cf0fdb67bd543cca0c216`, updated18:44:28UTC; separate reasoned runtime review6639096662 precedes exact20-field approval6639099878 at18:46:40UTC, before19:14:28deadline. Full approval canonical SHA256 `6cbe2142dd19d680ebf8a8d206e765a2056ac68bd4be31beac420c352dcba4eb`. Receipt and every per-game current authority/input check bind the same approval, source, prepared commands, cold state, seeds/ports and full runtime. Pre-game proof remains permanently preserved at500bc1341437afbdb72e5ecce25c0e101858a55e on its evidence branch. A shared GitHub account alone is not independent review.

## Local storage interruption, publication and limits

Repeated Mac ENOSPC occurred only during later extraction/audit/publication, after the hosted battle run completed. Two local extraction failures, the explicitly quarantined `LOCAL-INCOMPLETE-EXTRACTION-NOT-OUTCOME.json`, and byte-matched derived-copy cleanup records remain evidence. Unique original ZIP, original raw decision, successful full audit outputs, receipts and sources were retained. These local failures are not invalid rounds, stage/runtime failures, permission to replace battles, or evidence for an additional run. The user subsequently freed storage; fresh replay above now verifies the complete outcome.

Final ledger/PR completion remains an owner task. The separate user-authorized XPS offload is now published in [PR113](https://github.com/stephanelee/robocode/pull/113), exact archive head `a665276707bc01694eac8fe11897254b93c6bbb8`, branch `evidence/keel-helm-confirmation-xps-41889609469`; independent preservation review, required CI and merge remain pending. Canonical raw archive path is `benchmarks/keel-helm-v1/confirmation-evidence-41889609469/`; custody report is `docs/improvements/keel-helm-v1-confirmation-publication.md`. Coordination is recorded in issue107comments6643846378/6644196250, with a bounded additional independent audit in6644053839. This recovery reviewer corroborated PR metadata but did not write or independently re-audit that whole published archive or supply its final merge clearance. Whole gzip bytes are preserved as indexed exact original ZIP members and verbatim content-addressed ZIP parts; publication does not waive the rejected competitive gate. Parent must reconcile canonical commit/report links, update the ledger, and obtain separate final exact-head/CI review before marking complete.

Cold games are the comparison units; adaptive rounds are correlated and uncontrolled starts are not paired seeds. These are finite engineering gates, not a prespecified statistical-significance or broad-strength claim. Original screen and historical Keel games are separate and cannot rescue or pool with this confirmation. Diamond private skips are unobservable. Exported source/manifests/proofs/API bindings are independently checked; unexported live native/JDK/Python bytes were rehashed by the reviewed worker, not remotely rehashed by this post-outcome reviewer. No next experiment or fix is authorized by this report.

## Reproduction and retained output hashes

Fresh command: `python3 -I -B -S benchmarks/keel-helm-v1/independent-analysis/audit.py audit-stage --evidence /private/tmp/keel-helm-confirmation-41889609469/robocode/robocode/build/keel-helm/confirmation --source-root /private/tmp/robocode-keel-improvement --stage confirmation --head f2a25d3309062a362a13902329de405bf63094a7 --run 41889609469 --output /private/tmp/keel-helm-confirmation-41889609469-recovery-independent-audit.json`. Exit1 is the expected competitive REJECT, not an execution failure.

Binding reproduction uses `python3 -I -B -S /private/tmp/keel-helm-confirmation-41889609469-recovery-bindings-audit.py`; the copy differs from its preserved predecessor only in output destination.

| Retained file | SHA256 |
|---|---|
| `keel-helm-confirmation-41889609469-recovery-independent-audit.json` | `87d856bdac398249a1392603704b594e0a3423076f6128ea1d3984fb172b6463` |
| `keel-helm-confirmation-41889609469-independent-audit.json` | `87d856bdac398249a1392603704b594e0a3423076f6128ea1d3984fb172b6463` |
| `keel-helm-confirmation-41889609469-recovery-bindings-audit.json` | `64dfaa856d9ac7b8aad33d21d99bf20aa6084a191fb6f74ddac6711272c2c497` |
| `keel-helm-confirmation-41889609469-bindings-audit.json` | `64dfaa856d9ac7b8aad33d21d99bf20aa6084a191fb6f74ddac6711272c2c497` |
| `keel-helm-confirmation-41889609469-bindings-audit.py` | `113a8e9552575b0b745eedfb9edbcab8a23b89c94fbcf7f9564b9b130046d351` |
| `keel-helm-confirmation-41889609469-recovery-bindings-audit.py` | `153b1c94428f4a37a388da9517e9379e8f447a14f83faa755b3ae9d0e1e0a890` |
| `keel-helm-confirmation-41889609469-recovery-verification.json` | `fba388ec48aa47cfaead48aa8f1c15e46f4fb02d42f09ab886e80a61bebb8c77` |
| `keel-helm-confirmation-41889609469-archive-stream-audit.json` | `f0162e12a92ce072e11bce054b9dd0edcc9e3a4906a0e7317997b09e25189ec5` |
| `keel-helm-confirmation-41889609469-decision-comparison.json` | `a2acbf442a3da92450433fc4c1ce300170451e8d71cfb947fe78b5c14db2bb6c` |
| `keel-helm-confirmation-41889609469-descriptive-groups.json` | `e51d25fb193a76e8c97b3993371a11dc120bc79035e63b49654f445b74b7637e` |
| `keel-helm-confirmation-41889609469-conditional-screen-gate-audit.json` | `c32583cfbcc78142dd266a1b0b8ca30326243bf33d6f7b24c679151d513bd36d` |
| `keel-helm-confirmation-41889609469-process-time-upper-bounds.json` | `471db882d4df468ef9cbc538c40c640c3ecbb114757d1bc34a42aa58ec0b9176` |
| `keel-helm-confirmation-41889609469-games.csv` | `1ac04e83c2983972789a50424e27c8e0590ad67687819df944ac1b237ae7e860` |
| `keel-helm-confirmation-41889609469-rounds.csv` | `0f025a71e099d5c2468407938b2188b16718998e5c504a669330ad466759dba0` |
| `keel-helm-confirmation-41889609469.log` | `4175fcdba6f03bdb4234f5d088f7a1c6702170dbdee4e90c91d7c5500136ed3e` |

## Review of owner result and ledger text

The recovery reviewer compared the current three owner-edited documents against the independently replayed result: `docs/improvements/keel-helm-v1-results.md`, `benchmarks/keel-helm-v1/README.md`, and the existing Helm row in `docs/improvements.md`. Their rejection, budget, score/win/damage/hit/firing/contact/exposure numbers, original-byte identities, nineteen authority checks and prior6,150field/2,217binding counts match the retained outputs. Their explicit pending-XPS-custody scope was correct at initial text review; update those links to the now-published PR113 archive commit while preserving its pending independent-review/CI/merge scope. Small clarity suggestion: say the screen *opened* eligibility rather than that it *opens* eligibility now. This factual text review is not final published-head/CI/archive or merge clearance.

Reproducible fresh stream verification: `python3 -I -B -S /private/tmp/keel-helm-confirmation-41889609469-recovery-stream-verify.py`. Its source SHA256 is `cb236a8804e6db46ce14c353fcd9332a7546ddd16a9cd4e8c5704ea53c737aad`.
