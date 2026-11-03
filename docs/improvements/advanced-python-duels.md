# Frozen default AISurvivor against two advanced Python opponents

The user requested separate 1v1 benchmarks against Diamond Port and Adaptive Prime in addition to the independently registered stronger melee. This is a descriptive measurement, not an optimization, acceptance test or comparison of isolated components.

Freeze accepted AISurvivor gameplay `94640028a1b5bf072713bc19ea1b608d14f9d26d`, production JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`; base master `fca6d2ff1101fcf4dd07908cce8df9d18b32dc63`. Open PR47/48 changes are excluded. Reuse the separately pinned `strong-melee.lock.json` and shared opponent preparation loader unchanged.

## Fixed population and acquisition

- Exactly **120 physical/scored AIS rounds**: two opponents, three fresh independent 20-round games against each. Diamond IDs `duelpy946-diamond-g01` through `g03`, AIS seeds20262701–20262703; Adaptive IDs `duelpy946-adaptive-g01` through `g03`, seeds20262711–20262713. These are new and are not reused from the melee or earlier experiments.
- Actual default TRAIN/controller, no duel override. Every game gets a new AIS process, initially missing private checkpoint, private metrics and fresh opponent source/core working tree. All20 online-learning rounds count; do not pretrain or carry models/learning between games. Source-file hashes must match before and after play, excluding generated caches and telemetry.
- Diamond Port1.8.28 and Adaptive Prime1.0 are native Python Tank Royale opponents from `fduch-stranger/robocode-bot@67b68c22e41c18281a3fce9ef638e3c23a0b47cf`. Preserve Apache-2.0 shared/Adaptive notices and Diamond's altered-port zlib notice. Diamond is not the original Java tournament bot; author-reported local results are not independent tournament validation. No Tomcat or unlicensed reference tree is included.
- Use the frozen110-file source selection and12 exact Python3.13 Linux wheels in `tools/opponents/strong-melee.lock.json`. Verify all source Git blobs/SHA256/size and wheel hashes; no dependency resolution or ambient package fallback. Shared loader prepares a private venv and explicit interpreter, cleans inherited AIS/ROBOCODE/PYTHON/PIP settings and excludes `.env`, `.env.guns` and upstream project scripts.
- Reuse `benchmark.battle` with exactly one opponent. Tank Royale1.4.0 official `BattleSetup.oneVsOne`,800×600,100000µs turn timeout,20 rounds,900-second game deadline. Embedded ephemeral server/booter ports must not connect to or alter the user's GUI/server on7654. Seeds affect AIS RNG, not engine starts; no paired-start claim.
- Ubuntu26.04 runner only. Java17 fresh source build/tests/normalized committed-JAR parity first, then Java25 match runtime, Python3.13. Record execution SHA, runner/source/bundle/runtime/launcher/lock hashes, interpreter/dependency provenance and every actual official roster.

## Reporting rule

No score threshold or strength-promotion rule is declared. Run the complete fixed budget without performance stopping, tuning, seed selection or favorable reruns. Retain skipped turns and losses as observations. An acquisition/schema/identity/source/checkpoint failure stops that opponent's acquisition, preserving all partial evidence; no replacement game or automatic retry. The two independent opponent jobs may finish separately if another fails.

For each opponent report all game ranks, official scores/shares/first places, equal-game mean share, wins out of60, shooting/hit rate, damage, collisions, skips and default action/duel counters. Game ranks are not per-round ranks. Validate telemetry1..20, cold-model flag and finite20-round model endpoints; retain all six models and official arrays/logs. With only three adaptive game blocks per opponent, these are descriptive results for this pinned setup, not a broad ranking or isolated neural-learning benefit. Do not pool either opponent with the melee or earlier Retcon games.

Independent exact protocol/driver/workflow review precedes the sole collection. Publish raw evidence and a qualified report, update `docs/improvements.md` after each completed opponent experiment, and have the separate reviewer audit and merge the evidence PR after required CI. No gameplay or instrumentation changes are authorized.

## Attempt1: source acquisition stopped before any games

[PR51](https://github.com/stephanelee/robocode/pull/51), [run41647695154](https://github.com/stephanelee/robocode/actions/runs/41647695154), execution `f25c7ffb54232400c0c4f72c4b674f20169fbdaa`. Both jobs passed Java17 build/JAR parity, then failed in shared opponent preparation with `Unexpected opponent source/config files`. Both original `duels.json` files have status invalid and an empty games array: **0 of120 rounds played, no models, no scores and no used game IDs/seeds**. No battle runner or opponent was launched.

The independently diagnosed packaging defect is a missing-file check: a nonexistent pinned zero-length `__init__.py` was treated as bytes `b''`; its size/hash therefore matched the empty-file pin and the downloader failed to create it. The subsequent exact-file-set guard correctly stopped acquisition. This is not a bot-strength or competitive result. Preserve both original reports, execution tooling/lock, failed-job logs and artifact metadata in the [failed-attempt evidence](../../benchmarks/advanced-python-duels/README.md). The one-shot workflow was removed locally before any subsequent push. No Actions rerun or replacement game was launched.

## Corrected acquisition: Diamond result

The [prospective continuation](advanced-python-duels-continuation.md), frozen `c573055dd78784d8464a62027cc7f17321ec0b6c`, retains the original unspent budget after the independently reviewed packaging-only correction. [Run41648730946](https://github.com/stephanelee/robocode/actions/runs/41648730946), execution `cdc9683319af11b795d4ce5befb846f96fae47e5`, uses the same reviewed workflow bytes SHA256 `74ec3b0f27464d76da7bd674a99d5c6eca49c8c3904b157b990495bc226c13b8`. No gameplay, model, opponent or runtime pin changed. Both independent opponent jobs completed the full registered120 rounds; the one-shot workflow was removed before any subsequent push.

| Diamond game | AIS score / Diamond score | AIS score share | AIS first places /20 | Whole-game AIS rank |
| --- | --- | --- | --- | --- |
| duelpy946-diamond-g01 |130 /3112 |4.009870% |0 |2 |
| duelpy946-diamond-g02 |127 /3141 |3.886169% |0 |2 |
| duelpy946-diamond-g03 |127 /3249 |3.761848% |0 |2 |

**Descriptive result: AIS wins0/60 rounds, equal-game mean share3.885963%, second in all three games.** Retain the complete negative baseline; no candidate was tuned or promoted. Across23228 reported turns it fired1160 shots with72 hits (**6.206897%**), dealt383.6 damage and took5050.5, with9 wall collisions,0 bot collisions and0 skipped turns. There were22943 eligible/forced-duel selections. Three distinct cold, finite20-round checkpoints and all official arrays/raw rounds/source and wheel provenance validate; independent raw review reproduced them.

This demonstrates a substantial weakness against this pinned **Python Diamond Port** in this20-round TRAIN regime, despite AIS's earlier Retcon and easy-sample results. It does not establish equivalence to original Java Diamond or isolate a gun/movement/neural cause. Native opponent turn/skip telemetry was not enabled; do not infer Diamond's zero-skips from AIS's metrics. Historical Retcon/sample evidence is not pooled. Later merged PR47 warm-up changes were not in this measured946/9679 bot; final branch integration must preserve that source distinction.

## Corrected acquisition: Adaptive Prime result

| Adaptive game | AIS score / Adaptive score | AIS score share | AIS first places /20 | Whole-game AIS rank |
| --- | --- | --- | --- | --- |
| duelpy946-adaptive-g01 |546 /2550 |17.635659% |1 |2 |
| duelpy946-adaptive-g02 |541 /2688 |16.754413% |0 |2 |
| duelpy946-adaptive-g03 |590 /2572 |18.659077% |2 |2 |

**Descriptive result: AIS wins3/60 rounds, equal-game mean share17.683050%, second in all three games.** Across44335 reported turns it fired2631 shots with270 hits (**10.262258%**), dealt1464.900003 damage and took3851.8, with2 wall collisions,0 bot collisions and0 skipped turns. There were43964 eligible/forced-duel selections. All60 raw rounds/official identities and three distinct finite cold20-round checkpoints validate; independent review reproduced the original artifact13237191142. All pinned opponent source files and the exact12 runtime distributions were verified.

**Disposition for both experiments: complete negative descriptive baselines; retain all results and unchanged production.** No gameplay candidate was proposed, accepted or rejected, and no tuning, source substitution, favorable rerun, pooled score or model carryover occurred. Both opponents expose weaknesses in this frozen bot; earlier success against Retcon does not imply strength against these opponents. These three-game observations do not establish a general tournament rank or identify which AI/component must change. Native Python ports' private timing/skip telemetry was not captured. The original zero-game acquisition failure stays visible alongside the corrected full120-round cohort. [Raw evidence and per-file hashes](../../benchmarks/advanced-python-duels/README.md).
