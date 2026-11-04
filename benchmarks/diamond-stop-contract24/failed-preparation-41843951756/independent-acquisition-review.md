# Independent STOP24 acquisition-run audit

**Infrastructure failure before activation; numeric budget remains 0/24 cases, 0 games. No STOP prediction result was obtained.**

I monitored only existing run [41843951756](https://github.com/stephanelee/robocode/actions/runs/41843951756), attempt1, reviewed head `6007bcdd616e793030c5900e12bdb2e786964c52`. The run was created14:04:31UTC and finished failure14:05:43UTC. Its profile job was skipped; acquire ran14:04:36–14:05:42UTC. Fresh compile and ordinary read-only prepared validation succeeded. The next acquisition step failed at the very first `verify(True)` call, `diamond_stop24_host.py:198 -> :173`, with `ValueError: Actual profile differs from independently frozen zero-case profile`.

The complete log and reviewed source order show this exception occurred before exact-head API/admission queries, annotated receipt creation, activation.json, evaluator invocation or numeric bridge main. Independent GitHub query of fixed ref `refs/tags/experiment-receipts/diamond-stop-contract24-v1` returned404 after failure. The artifact has no activation.json, admission-jobs.json, acquisition-command.json, acquisition.log or acquisition root/raw rows. Compile provenance records `casesExecuted:0`. Thus the workflow attempt failed, but the permanent numeric receipt was not consumed; no numeric case, helper, predictor, game or new battle was executed in this run or by this audit. No retry or repaired continuation is cleared by this report.

## Exact failure

The frozen full profile selected Ubuntu26.04 image `20261101.186.1`, defining member `profile.json`. The fresh job selected the other originally allowed family member, image `20261026.180.1`, defining member `profile-20261026.json`. Both are Ubuntu26.04.1/x86_64 with Temurin17.0.21+9 and CPython3.13.18. Nevertheless the new full immutable freeze bound the first image's actual bytes, so the second family member is correctly rejected; membership in the original two-image family does not waive whole-profile equality.

Frozen profile SHA256 `dfbd5df2660bf5207144aa5990c2c019ce3662d1fd6efa93b2f9d65567a7ca1d`; fresh actual profile SHA256 `ead57b2dd0deec7189a99ab111fea4a28cdb7c12dd56384ecf9d4ffe9bb024a7`. Freeze remains SHA256 `be6e1bc1271f5e5e0f1d2d1f76b008ed47125c6a3797876788adeb949e242c92`.

A metadata-only recursive comparison found181 differing leaves:168 stdlib pyc hashes, eight repeated Brotli native-library hash entries, two loader-cache hash entries, and three non-Python fields (image, selectedProfile and gh executable hash). These are the exact frozen-profile difference paths; copies/repeated native bindings explain repeated entries. No implication that all differing bytes would affect the unexecuted diagnostic follows. There was no substitution, normalization, cache deletion or runtime repair.

All30 tooling hashes, six core-artifact entries, JDK17 hashes, base Python executable hash and pinned Python package/source/wheel graph match the immutable freeze. In particular bridge compilation reproduced the frozen artifacts; preparation did not establish bridge-main/import/numeric feasibility. Package manifest remains52b7…. The runtime proof changed from35d9f7… to `8d126a4020e61d8cccbbc9cd4aa69fda95e93ced4de79111bbfecdf703765b52`.

Actual boot ID `9d31cbd2-f3b3-47ed-a31f-43e99b0537cb`, workspace `/home/runner/work/robocode/robocode`, run41843951756/attempt1 reconcile in archived host identity. No 300-second numeric process was launched; the failure is preparation/profile rejection, not a timing or gameplay result.

## Preserved evidence

- Original artifact13329404282: `/private/tmp/diamond-stop24-acquisition-41843951756.zip`,19,607,166bytes, SHA256 `38ddbe432db9dc9965a7f8d498c19239f175e9d40d38a8fa6bbc1269aef71e6d`. Hash matches GitHub artifact API digest. Safe extraction at the same basename directory; all72 regular files compared byte-for-byte with original ZIP entries. Includes original frozen preparation evidence and new failed-run preparation/proofs/sources.
- Complete decoded `gh run view --log` output: `/private/tmp/diamond-stop24-acquisition-41843951756.log`,54,914bytes, SHA256 `f563ad6c0496ecc8f0fcf7cb45c3a1fb11d2acd8a56c657386e76a40bbe3c415`. It retains full provider lines, including masked token notation and whitespace; no secret was exposed.
- `/private/tmp/diamond-stop24-acquisition-41843951756-run.json` and `-jobs.json`: actual run/step API metadata.
- `/private/tmp/diamond-stop24-acquisition-41843951756-profile-diff.json`: all181 original old/new difference paths,49,782bytes/SHA256 `4056fe2ec2a34f61e771489fec8adde652bd52addd42e2963756d2429a463d7f`.
- `/private/tmp/diamond-stop24-acquisition-41843951756-static-audit.json`: complete72file hash manifest, preserved extra-file hashes, exact identity/budget/difference summary and reconciled metadata,17,844bytes/SHA256 `267bb5506461863932e461ac7283fa8377922db82f3bd2747520b086b72cfa89`.

## Disposition and scope

Preserve this failed workflow and its original evidence, remove stale stage activation if root chooses, and update the ledger as a zero-case infrastructure failure. Do not rerun the failed workflow or repeatedly relabel until a favorable image appears. The independent exact-head review required equality before activation and expressly forbade normalization/waivers. Any changed runtime plan, freeze or payload requires prospective registration/amendment and independent review; this audit grants none. Obtaining the exact frozen hosted runtime reproducibly is the concrete unresolved resource/provenance requirement.

No STOP discrepancy, movement improvement, bot correction, wins/score/damage/firing/hit rate, competitive strength, promotion or merge conclusion is supported. These metrics are N/A because zero games and zero numeric cards ran. I performed only existing-run monitoring, read-only GitHub queries/downloads, ZIP/hash/source inspection and static JSON comparisons. No branch/PR/label/ref/workflow mutation, numeric helper call, tested import, server or game was performed.
