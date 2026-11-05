# Wave 2 L1/game 1 — explicit validation ruling

**Accept the original completed engine result: AISurvivor 598, Breakwater 1,200.**
No rerun, replacement, model change or game-setting change. This ruling corrects a
telemetry-validation assumption; it does not change the official score or ranking.
An independent review confirmed the original game and recommends this acceptance
before continuing the three unattempted fixtures in the already authorized wave.

The frozen validator rejected AISurvivor's simultaneous `won=true` and `died=true`
in rounds 9 and 18. The pinned 1.4.0 engine chooses round winners by score rank,
including dead bots. Its `TurnProcessor.computeRoundOutcome` selects rank 1;
`ModelUpdater` emits `WonRoundEvent` to those IDs without an alive check. AISurvivor
scored 75–60 and 77–60 in those rounds while dying. Official first-place counters
and observer deaths corroborate both flags. Survival and score wins are distinct.

All twenty original telemetry rows satisfy every other frozen validator check.
The untouched recording has one game-start, one game-end, twenty round-end events,
35,242 ticks and matching per-round terminal turns. Its final cumulative results
match every retained official field. Preserve the conflicting flags as recorded;
they remain part of the bot's actual within-game behavior and learning history.

| Evidence | SHA256 |
| --- | --- |
| Original failed stage report | `8637bd6fad66aed060ea1781ff438c0500c5e7923317a29483772b480d910df7` |
| Official battle.json | `de3ef09557adf97bbed4b2c205b9bdf565e53d3db70ea29bb298aa23e22674fc` |
| Original metrics | `e3fb097127af8fb193b96cacb574a2963e2e10faba53d72da7a52928be47296b` |
| Recording | `bf16743d25c8e5b9fde59b8ab88c9c17ca5b62e4b6d0f9d651299e7b2f3426c2` |
| Frozen benchmark validator | `f31a7f85d06d99ca5f19e6a2eddb6264059443d1856816a4860fe3691aa2a576` |

Apply only the hash-bound `playoff-2-ruling-1.json` with the reviewed adjudication
tool. It rechecks raw integrity, participants, all coverage and observer evidence,
and runs all other frozen telemetry checks using an in-memory projection of the
disputed flags. It never writes projected metrics. Preserve the exact failed
report as `playoff-2-validation-stop-1.json`; keep the original error in the game
record and `resolvedFailures`, with the ruling hash and observer proof. Only its
active failure is resolved; any unrelated failure remains blocking.

Promote the original output/recording to a validated game, count its original 20
rounds once, and keep the stage incomplete until explicit `--resume-unplayed`.
The unchanged runner then launches only L1/game 2 and L2/games 1–2. Preserve the
initial runner log and use a new continuation log. Another validation failure
stops again and requires its own reviewed ruling. No wave 3 is authorized.

Pinned server SHA256: `16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c`. Its nested bytes match the original pinned runner. Durable `javap -c -p` transcripts for both classes and runner/member/server custody hashes are retained in `local-evidence/tournament-2026-11-04/adjudication-1/`. The adjudication was independently approved and applied; no frozen execution source changed.
