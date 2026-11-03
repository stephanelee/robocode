# Breakwater v1 against Diamond Port 1.8.28

## Hypothesis

A power-0.1 bullet can intercept an inferred incoming bullet. At ordinary duel range Diamond often spends 1.95 energy on that shot. If Breakwater can induce a predictable incoming ray, it can spend less energy on defense, avoid damage and deny the shooter its hit bonus. It still needs survival or offensive points. While the gun is free, Breakwater takes selective attacks on targets whose stationary and constant-motion forecasts agree, or attempts to finish a low-energy target. When interception is infeasible, it evades. A miss against a stationary or slowly moving defender could be particularly costly.

The pinned Python gun attempts a previously aimed shot, gates firing on cool/aligned state and varies power with distance, energy and rammer classification. Its surfer compares stop and two orbit directions and can anticipate incoming fire before seeing an energy drop. The server checks intersecting bullet segments in the same tick. These facts permit the proposed tactic. They do not establish that inferred shots have the right origin, timing or direction for interception. [Diamond gun](https://github.com/fduch-stranger/robocode-bot/blob/67b68c22e41c18281a3fce9ef638e3c23a0b47cf/bots/ports/diamond-port/diamond_port/gun.py#L1038-L1120) · [Movement](https://github.com/fduch-stranger/robocode-bot/blob/67b68c22e41c18281a3fce9ef638e3c23a0b47cf/bots/ports/diamond-port/diamond_port/move.py#L765) · [Collision predicate](https://github.com/robocode-dev/tank-royale/blob/c8ad3a8d19a843f6258d6f6f9db7f29229963903/server/src/main/kotlin/dev/robocode/tankroyale/server/core/CollisionDetector.kt#L462)

The last valid guarded-gun pilot took 1,441.39 damage, dealt 185.2, won 1/20 rounds and earned 8.161933% share. A five-round diagnostic measured 13.386 degrees mean forecasting error against a 2.306-degree target half-width. Neither measures how predictable Diamond's own bullets are. The unrelated outgoing-context experiment's endpoint failed canonical validation. This design spends gun opportunities on eliminating incoming damage and separately records actual projectile collisions. [Pilot](../../benchmarks/diamond-guarded-ray-v6/README.md) · [Diagnostic](../../benchmarks/challenger-diagnostic-v5/README.md) · [Invalid outgoing-context attempt](../../benchmarks/diamond-outgoing-context-v7/README.md)

## Prospective rules

Use an isolated Java 17 bot, source tree, JAR, descriptors and fresh per-game processes. Only our scans, state, commands and confirmed events are available to our policy. Bounded resolved learning may persist between rounds of one game. No prior checkpoints load. Retain the final private endpoint for each game. Freeze the exact source, constants, test and driver bytes, workflow, bot JAR, opponent source, dependency and battle runtime pins before measurement. Review the frozen head independently before activating the one-shot run. CI uses ubuntu-26.04, Java 17 builds/tests and committed-JAR parity, JDK 25 for battle and the registered Python 3.13 native opponent environment. Use isolated headless ports and exclude 7654.

| Game | Challenger seed | Budget |
| --- | --- | --- |
| breakwater-v1-pilot | 531801 | 20 rounds |
| breakwater-v1-confirm-g01 | 531811 | 20 rounds, conditional |
| breakwater-v1-confirm-g02 | 531812 | 20 rounds, conditional |
| breakwater-v1-confirm-g03 | 531813 | 20 rounds, conditional |

One fresh pilot. Require valid complete evidence, zero own skips, share strictly greater than 20% and at least one official first place. A valid miss ends competition with rejection. Only a pass opens three fresh confirmation games. Require all three valid, zero own skips, equal-game mean share strictly greater than 50%, strictly more than 30 official first places across 60 rounds and at least two games strictly above 50% share. Use official totals and report each game's share. A tie at either strict threshold fails. An invalid acquisition stops the attempt. Archive all started games, partial files and logs. No replacement, favorable rerun, pooling or parameter change. The seeds specify challenger-controlled randomness only.

Report confirmed defensive/offensive shots, actual projectile collisions, hits, energy spent, damage, scans/range, skips and round/model integrity alongside official totals. Interception counts diagnose the tactic but earn no independent strength credit. A pilot pass alone does not establish repeatable wins. Retcon, AIS integration, generalization and melee require other evidence. [Tracker](https://github.com/stephanelee/robocode/issues/53)
