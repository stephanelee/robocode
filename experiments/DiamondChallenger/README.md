# DiamondChallenger0.1.0

A separate experimental1v1 bot for the [pinned Diamond counter-hypothesis](../../docs/opponents/diamond-port.md). It does not subclass or rename AISurvivor, load an AIS/DQN model, or change production sources/JARs.

The whole design combines short-range interception pressure (preferred140px), independently seeded20–45-turn flank decisions, pinned1.4 movement/wall simulation, independent radar, three-model online virtual targeting, approach-energy conservation and stronger close-range fire. It attempts to shorten Diamond's long-range surfing window; Diamond's retreat and2.95 close/rammer counterfire can defeat that hypothesis. No winning claim follows from building this prototype.

Build/test with Java17 from the repository root:

```sh
./gradlew -p experiments/DiamondChallenger --no-daemon clean build
```

The standalone Gradle project compiles its own entry/movement/gun and four unchanged pure geometry/radar/forecast/physics helpers. It does not compile/include AISurvivor, SurvivorBrain or DQN. Runnable output is `experiments/DiamondChallenger/build/bot/DiamondChallenger/`, with its own JAR/descriptor/launchers. Root Gradle build inputs remain unchanged.

Set `CHALLENGER_SEED`, `CHALLENGER_MODEL_PATH` and `CHALLENGER_METRICS_PATH` before launch. Each benchmark game requires distinct initially absent model/metrics paths and a new process. The schema1 JSON model snapshot records name/version, seed, completed rounds and three virtual-gun attempt/hit counters. It is endpoint evidence of in-process learning, **not exact process resumption**; this cold-game prototype does not load it.

Schema1 per-round JSONL records name/version, mode=train, seed, round, roundsCompleted, reported turns, checkpointLoaded=false, actual shots/hits, damage dealt/taken, `energySpent` (sum of confirmed own bullet powers), wall/bot collisions, skips and won/died flags. Range diagnostics are `rangeObservedTurns`, `rangeSum`, `closeTurns` (range<180) and `phaseDecisions`. Gun labels are geometric virtual outcomes based only on confirmed own shots and consecutive fresh legal scans, not opponent private state or proof of neural learning. Source/runnable hashes, private endpoints and every failed acquisition must accompany a prospectively reviewed fixed-budget battle.
