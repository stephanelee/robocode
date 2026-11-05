# KeelHelm melee v2 — prospective lifecycle repair and new comparison

Registered before the lifecycle edit or any v2 games. Issue #123 / PR #124. The v1 comparison is INVALID and permanently stopped after two 20-round melee games (40 physical rounds). Control scored 15.21% in its sole valid game; the candidate engine completed 20 rounds but had no round-start telemetry and cumulative melee state. Its 392 points are retained as invalid evidence, never an accepted strength measurement. Fourteen planned v1 games remain unplayed. The earlier folder-collision setup consumed zero games.

## New hypothesis and exact repair

Pinned Bot API getEnemyCount() dereferences getCurrentTickOrThrow(), unavailable in RoundStarted before a tick. The new v1 round callback called it before resetting melee state and writing start markers. Replace only that round-callback read with an empty/inactive melee reset; activate melee from the existing live-tick/scan branches once enemyCount is available. Preserve all movement/gun/risk constants and geometry, target logic, death tombstones, duel behavior and opponent pins. Candidate becomes 0.2.1 to distinguish invalid 0.2.0. Add a real, unstarted Bot callback fixture proving the callback does not require live tick state and resets prior melee data/counters, including repeated rounds. Unit fixtures launch no server and consume no battle budget.

## Separate fixed comparison

Use the unchanged schedule, scoring gates, frozen tournament runtimes/opponents, arenas, cold starts, order rotations, own skip/timing guards, independent pre-game source/JAR/runtime approval and raw recording audit specified in [v1 protocol](keelhelm-melee-v1-protocol.md). Freeze v2 source, committed JAR, runtime, schedule and harness in a new output directory. Stage all sixteen rosters with the same acquisition function before review. Do not resume v1 or reuse/pool its valid control game or invalid candidate game.

Exactly four fresh 20-round melee games per arm with Diamond Port, Keel and Prism (160 rounds); two 10-round Retcon duels per arm (40); one five-round Walls and SpinBot duel per arm (20). New v2 maximum is 16 games / 220 physical rounds. Combined project accounting is at most 260 physical rounds including the invalid v1 attempt. The old and new comparisons remain separately reported. No tuning, replacement game, extra development game or adaptive budget.

Accept only if all games/evidence/timing guards pass, candidate equal-game mean melee share >=20%, melee delta >=5pp and each Retcon/Walls/SpinBot delta >=-5pp. Otherwise reject; no general-strength claim from this small uncontrolled-start sample. Stop this new comparison at first evidence/runtime/skip/timing failure and retain all evidence. The lifecycle repair is based on reproducible callback/API failure, not score optimization.

Java17 candidate/control builds, tests and source/JAR parity precede JDK25 games/Retcon. Production AISurvivor remains byte-unchanged; retain and disclose its existing five arm64 exact-bit golden failures and one skip out of89 tests, plus passing822-entry JAR parity, as in v1. Publish all outcomes, source/JAR/opponent/runtime hashes, failures, budget and accept/reject decision with report/raw links and PR in docs/improvements.md. Independent reviewer reviews; implementer never merges. Tournament history and frozen champion remain0.1.0.
