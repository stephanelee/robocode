# Facet

Facet is a standalone Java 17 Tank Royale experiment with contextual guns, online learning, two-wave movement and independent radar. This is a duel milestone under review. Melee activation and shadow safety credit are gated. No Facet competitive battles have run. See the [source boundary](../../docs/facet-source-separation.md) and [validation report](../../docs/facet-validation.md) for provenance and remaining gates.

From the repository root, set `JAVA_HOME` to Java 17 and run:

```sh
./gradlew -p experiments/Facet clean build verifyJarParity
```

The dependency is `dev.robocode.tankroyale:robocode-tankroyale-bot-api:1.4.0` from Maven Central. The Gradle build, semantic suite and source/JAR parity passed locally under Java 17; required CI and independent acceptance remain pending. A source change requires refreshing `Facet.jar` from `build/libs/Facet.jar` before the parity gate can pass. Optional offline development uses `python3 build.py` with `FACET_BUILD_API_JAR` pointing to the frozen official API artifact; it performs no fetches or battles.

Run `./Facet.sh` only with an explicit isolated `SERVER_URL` environment value, or Java property `-Dserver.url=ws://localhost:PORT` when invoking Java directly. Lowercase `server.url` has official precedence. The exact validated URI is passed to the API constructor. Port 7654, missing/invalid ports, non-WebSocket schemes and malformed destinations are rejected. `java -jar Facet.jar --verify-package` performs no connection. No launcher contains a clean-room temporary path.

All optional state/telemetry basenames must start `FACET_`:

- `FACET_SEED`: deterministic seed override, default 49667; no parameter search was run.
- `FACET_TRACE_PATH`: bounded JSONL legal own/event/command chronology (100,000 lines across reused games).
- `FACET_METRICS_PATH`: bounded numeric JSONL counters, observation count/sum/min/max, latency log2-nanosecond histograms, GC and memory/wave statistics (10,000 lines).
- `FACET_STATE_PATH`: export-only own-game model; `FACET_MODEL_PATH` is the compatibility alias. No cross-game loading. Existing exports are never overwritten; subsequent games use `.game-N` suffixes.

`facet.properties` is immutable independent-v1 configuration: every key/value is validated by RuntimeConfig before play; unsupported edits fail rather than being silently ignored. All initial gameplay parameters remain the supplied independent profile. Model schema 1 writes magic, API/config hash/rules, local game IDs, ratings, labeled FIFO stores, feature values/quality/sequence and end marker, capped at 128 MiB. Missing-origin masks live on frozen waves; combined observation quality is exported. Model hashes and rating cohorts are logged at round end. Histories, unresolved waves, pending shots and failure-disable flags reset per round; learned stores, resolved rate counters and ratings persist within one game. A new Core starts cold for every game.

Tests replay copied approved raw fixtures in `src/test/resources`; no external fixture directory is required. See [semantic matrix](semantic-matrix.md), [decisions](decisions.md), [correction report](../../docs/facet-correction-report.md), [protocol](../../docs/facet-protocol.md), and [evidence](evidence/correction-final.txt).
