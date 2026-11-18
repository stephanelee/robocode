# KeelHelm Crossfire 0.1.0

Experimental fork of accepted KeelHelm 0.2.1. Crossfire keeps the incumbent melee target and uses its normal circular aim for gun turning. When the current gun ray already aligns with a nearby, freshly scanned alternate and the incumbent cannot fire, Crossfire can spend that turn's fire request on the alternate. The duel policy, melee movement and radar use the accepted implementation. The candidate has a separate entrant identity and runs with fresh process-local state.

Build with Java 17 using the repository Gradle wrapper and `-p experiments/KeelHelmCrossfire clean test assemble`. The assembled entrant is in `build/bot/KeelHelmCrossfire` within the project. The candidate launchers explicitly decode UTF-8 for the descriptor.

When a private `KEEL_HELM_METRICS` path is provided, round telemetry includes alternate request, accepted-request, matched-confirmation and unknown counts. A confirmation requires a physical own `BulletFiredEvent` on the expected next turn and agreement on power and current gun direction. Confirmations record the bullet ID and intended target. Determine actual hit victims independently from official events. Unmatched accepted requests remain unknown. The callback records do not enter targeting decisions.

The [prospective protocol](../../docs/improvements/keelhelm-crossfire-protocol.md) defines a conditional benchmark. Offline contracts check the selector, lifecycle and callback matching. Live runtime, entrant staging, receipt, schedule, raw recorder and independent clearance remain pending. No benchmark has run for this candidate.

Two clean offline builds on Microsoft Java 17.0.18+8 and Gradle 8.14.3 passed all 30 tests and reproduced the same 830-entry JAR, SHA256 `fe6498d461ebc6c92e5d2d03e047c91771166c3d339e49ed647cf47966164d96`. The initial local Gradle connection was blocked in the sandbox. The first offline test run used the original entrant's descriptor path in one inherited test and failed at file lookup. It passed after the fixture used the candidate descriptor. Neither attempt started a server or played a game.
