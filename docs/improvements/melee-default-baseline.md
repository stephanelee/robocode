# Frozen default-TRAIN four-bot melee baseline

This is a descriptive baseline requested by the user, not a candidate comparison, strength acceptance or automatic optimization. Freeze accepted master `e4e4c94a45c770a494f6a60f6b98456e215bee42` (production source94640028), JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`. Open external PR47/48 warm-up/replay changes are not included or touched.

- Exactly three independent20-round games: **60 physical/AIS rounds**, IDs `melee946-g01/g02/g03`, seeds20262501/20262502/20262503, fresh unused IDs/seeds. Fixed four-bot roster: AISurvivor + official Java Target + SpinBot + Walls, pinned Tank Royale1.4.0 sample ZIP SHA256 `1707c1011a37c6b854fc79d68549ab204007bac6d33d803fa254152ac0765fe2`.
- Actual default TRAIN/controller, duel override unset, all20 learning rounds counted, separate initially missing model/metrics paths and new bot processes per game. Do not reuse a checkpoint across games. Existing default behavior may transition to the modern duel paths as opponents are eliminated; report duelEligibleTurns/forcedDuelTurns/action counts without calling all melee turns purely legacy. These counters describe selection eligibility, not a full component-activation trace.
- Reuse existing `benchmark.battle(policy='train', opponents=['Target','SpinBot','Walls'])` only; no full CLI policy suite or new battle engine. Official1.4.0 `BattleSetup.melee`:1000×1000 arena with explicit min/max4 participants,20 rounds,100000µs turn timeout. Official embedded runner uses isolated ephemeral ports; do not connect to, restart or touch the live GUI/server on7654.
- Ubuntu26.04; Java17 fresh build/tests/normalized committed-JAR parity before JDK25 match runtime. Preserve runner/sample hashes, exact source/JAR/launch/runtime provenance and official participant identities/versions. Strip inherited AIS-specific environment overrides for each invocation, explicitly select TRAIN/private model+metrics/seed; sample bots do not share AIS models.

There are **no score/strength acceptance thresholds**. Retain every round, skipped turn and failure; do not tune, omit bad games, replace seeds or rerun automatically. Skips are reported baseline outcomes, not grounds to discard completed games. An acquisition/schema/roster failure stops and preserves partial evidence for review. This does not restart automatic optimization or change the separate Retcon confirmation.

Report each official whole-game rank, score/share and first-place count, all official arrays, shooting counts/hit rate, damage dealt/taken, wall/bot collisions, skipped turns, policy/action counters and checkpoint loading/header/hash. Aggregate complete games descriptively with equal-game shares; game ranks are not per-round ranks. Seeds initialize AIS RNG, not arena starts. This small fixed sample roster is not broad melee/tournament evidence or an isolated policy/learning benefit.

Independent exact driver/workflow preclear precedes the sole60-round collection. Preserve immutable plan/layout, all actual metrics, three model endpoints, official results/logs and sample/bot source/asset/runtime provenance. Publish the full outcome and PR/raw links in `docs/improvements.md`; separate review and required routine CI precede merge. No gameplay, instrumentation or GUI changes are authorized.

## Actual descriptive result

[PR #49](https://github.com/stephanelee/robocode/pull/49), [sole run41643133213](https://github.com/stephanelee/robocode/actions/runs/41643133213), frozen execution `08739890838b614efc6bc1e6e0213e4bbe99470b`. All60 rounds completed with three cold finite20-round checkpoints and zero AIS skipped turns. **Retain the descriptive baseline evidence; no strength acceptance/rejection or gameplay promotion is made.** Source/JAR remained exactly accepted e4e4/9679; external PR47/48 and the live GUI/server were not touched. No replacements or reruns occurred.

|20-round game | Official game rank | AIS score / share | Official first places | Hits / shots | Walls / bot collisions |
| --- | --- | --- | --- | --- | --- |
| melee946-g01 | 1 | 8740 /66.783831% | 20/20 | 563/858 | 81/84 |
| melee946-g02 | 1 | 8795 /66.583390% | 20/20 | 573/900 | 75/54 |
| melee946-g03 | 1 | 8638 /65.753216% | 20/20 | 570/909 | 28/31 |

Equal-game score share is **66.373479%**, with rank1 in all three games and **60/60 official first places**. These are whole-game ranks and the runner's first-place counters, not a recorded per-round rank series. The SpinBot source folder's official display name is `Spin Bot`, as verified against its pinned descriptor.

Aggregate telemetry:2667 confirmed shots,1706 hits (**63.967004%**), damage dealt13490.153565/taken948,184 wall collisions,169 bot collisions, zero skipped turns and42391 reported turns. The observed collision counts are baseline measurements, not eliminated by a safety gate. All three model loads were false; counters run1..20 and endpoints have finite state/header20 with distinct hashes.

Action counts for strategies0..5 are `[16192,5019,5116,5168,5467,5416]`. There were11240 duel-eligible selections and11240 forced-duel selections across the games (3944/3562/3734). Thus the measured default product includes its transition toward duel behavior after eliminations; it is not a purely legacy-melee measurement. These aggregate eligibility counters cannot reconstruct the full activation history of every component.

This is a small, fixed official sample roster, including a passive Target. It does not establish strength against arbitrary melee bots, a larger field, Retcon melee, or an isolated learned-policy benefit. No gameplay tuning or automatic optimization follows this baseline. [Raw60 records, three models, all official arrays/logs and source/sample provenance](../../benchmarks/melee-default-baseline/README.md).

Validation note: the initial routine engine-probe check on preparation heada592 ([run41642979705](https://github.com/stephanelee/robocode/actions/runs/41642979705)) had a probe-startup/roster failure with empty timing traces; it produced no timing conclusion. It is separate from the completed scored melee attempt. The final documentation head requires fresh routine checks and independent review; no scored retry was used.
