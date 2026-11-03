# User-authorized fresh default confirmation redo

After [PR44's invalid first attempt](final-default-confirmation.md), the user explicitly authorized **"You can redo it"**. This registers exactly one NEW attempt before its outcomes. The original140 physical rounds, skipped-turn failure and unknown cause remain immutable; no original game, model, subset or score is pooled, substituted or rehabilitated. This is a disclosed user-authorized redo, not an automatic retry.

## Identical bot, setup and binding rules

- Freeze accepted source `94640028a1b5bf072713bc19ea1b608d14f9d26d`, JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`. No production edits, instrumentation, extra diagnostics, timeout changes or optimization.
- Exactly200 fresh rounds: ten20-round games, two independent five-game batches. [Immutable fresh layout](../../tools/fixtures/final-default-confirmation-redo/games.json): `dc946r2-b1-g01`–`g05`, `dc946r2-b2-g01`–`g05`; AIS seeds20262401–20262405 and20262411–20262415, unused in previous performance games.
- Actual default TRAIN/controller, duel override unset; ten independent missing checkpoints/private model and metrics paths/new bot processes/RNGs. Score all20 online-learning rounds per game. Every model endpoint must have finite numeric state, counter20, sequential telemetry1..20 and checkpointLoaded=false. Positive forced-controller mechanism and both official identities must validate.
- Same pinned Tank Royale1.4.0 runner/800×600 1v1 setup/100000µs turn timeout, unchanged Retcon0.2.0 source `28981c7a469ac064194b41001cd7183ec5ecf189`. Ubuntu26.04, Java17 fresh build/tests/normalized JAR parity then Java25 runtime. Reuse the unchanged v2 acquisition helper with isolated launchers; no historical module globals are changed. Record exact source/runtime/asset/bundle/launcher hashes.

ALL original confirmation gates remain binding, using unrounded official equal-game score shares:

1. Overall ten-game mean share **>50%**.
2. Official first-place round count **>100/200**.
3. **>5/10 games** individually above50% share.
4. **Both** five-game batch means **>50%**.
5. One-sided95% Student-t(df9) game-mean lower bound **>50%**, critical `1.8331129326536335` times sample SD/sqrt(10).
6. All200 rounds/ten official/native arrays/ten independent cold models/default mechanism validate, with **zero AIS skipped turns**.

No performance early stop; acquisition invalidity may stop a batch as before. No replacement seed/game, rescue run, second retry after this attempt, favorable subset, gate waiver or tuning. Preserve every partial outcome. Invalid/incomplete/failed rules mean **not confirmed**, without reversing accepted PR43. No aggregate or binding t gate is evaluated for an incomplete population. Both jobs use fail-fast=false so a separate batch already running may finish after the other becomes invalid.

A pass completes this finite fresh confirmation; either outcome ends the automatic benchmark sequence with no further run scheduled. Game-level t assumptions remain approximate; within-game rounds adapt and are dependent. Any round-win Wilson interval is descriptive, not200 independent trials. This tests only this frozen bot against pinned Retcon/default20-round regime, not all opponents or tournaments. Seeds initialize AIS RNG, not engine starts.

## Publication and evidence

Independent exact-source/workflow preclear precedes launch. Preserve the immutable Git plan/layout, executed source/workflow, both original batch reports, all actual per-round metrics, official/native results, models and partial failures, plus original runtime/launcher/source pins. Do not duplicate runnable JARs. Update `docs/improvements.md` with measured results, confirmed/not-confirmed disposition, PR link and evidence before independently reviewed merge and required routine CI. The previous attempt and its NOTCONFIRMED result remain unchanged alongside this separate record.

## Actual outcome: confirmed under every registered gate

[PR #46](https://github.com/stephanelee/robocode/pull/46), [sole fresh redo run41584002511](https://github.com/stephanelee/robocode/actions/runs/41584002511), execution `d52218012dc095f41883746d113786107bb1d1f6`. Both batches and the aggregate completed successfully: **200 valid rounds, ten official/native game records, ten independent finite20-round checkpoints, no loaded model and zero AIS skipped turns**. All six binding conditions passed. The canonical disposition is **confirmed** for this frozen default-TRAIN bot against the pinned Retcon/runtime/regime. Source946/JAR9679 and setup/timeouts remained unchanged; no replacement, additional retry, profiling or outcome-dependent adjustment occurred.

| Registered measure | Actual value | Decision |
| --- | --- | --- |
| Equal-game mean share >50% | **86.744975%** | Pass |
| Official first places >100/200 | **187/200** | Pass |
| Games above50% >5/10 | **10/10** | Pass |
| Both five-game batch means >50% | **91.781954% /81.707997%** | Pass |
| Binding one-sided95% game-mean t lower bound >50% | **82.266939%** | Pass |
| Complete cold/default evidence and zero skips | **200 valid; ten models; zero skips** | Pass |

| Fresh game | Official share | First places /20 |
| --- | --- | --- |
| dc946r2-b1-g01 | 90.826180% | 20 |
| dc946r2-b1-g02 | 88.492063% | 19 |
| dc946r2-b1-g03 | 92.668464% | 20 |
| dc946r2-b1-g04 | 90.056965% | 19 |
| dc946r2-b1-g05 | 96.866097% | 20 |
| dc946r2-b2-g01 | 88.803513% | 19 |
| dc946r2-b2-g02 | 82.101977% | 18 |
| dc946r2-b2-g03 | 68.500000% | 15 |
| dc946r2-b2-g04 | 83.967828% | 19 |
| dc946r2-b2-g05 | 85.166667% | 18 |

Batch first-place counts are98/100 and89/100. Game-share sample SD is7.724998 percentage points. The round-level Wilson interval89.198092–96.162365% is descriptive only: online adaptation makes rounds dependent, so it is not200 independent trials. The binding Student-t gate uses ten game macroblocks with approximate model/independence assumptions. These results do not establish arbitrary-opponent/tournament superiority, an isolated learning benefit, or zero future skip risk.

The original [PR44 invalid attempt](final-default-confirmation.md) remains intact and is **not** pooled into these200 rounds. Its one skipped-turn cause remains unknown; this clean redo does not explain or erase [issue #45](https://github.com/stephanelee/robocode/issues/45). The user authorized exactly this new population after the prior stopping decision. The finite confirmation goal is now met under its registered rules, and no further automatic optimization, games or retry is scheduled.

[All original reports/raw200/models/source/clearance-to-execution provenance](../../benchmarks/final-default-confirmation-redo/README.md). The source/workflow preclear covered `abc7ac94837204b1cb417b752710c2017c0607d4` plus Windows-CRLF workflow SHA256 `b9510b90eada3f110288db4293a688e864bd26666c7f1c42ea4538cfa75b47cd`; executiond522 added only that workflow, normalized by Git to LF (SHA256 `12077252fc789687527555a946fa9d2517825866662be7a6d3fe83a8fd05ebd3`). Both files are preserved and normalized contents are identical. The temporary workflow was removed before any later push.
