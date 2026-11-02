# AISurvivor

A Java bot for [Robocode Tank Royale](https://robocode.dev/). This project uses Tank Royale's WebSocket bot API, rather than the original Robocode Java engine.

Project notes: [documentation index](docs/README.md), [benchmark methodology](docs/benchmarks.md), [measured improvement log](docs/improvements.md), and [the source-grounded Retcon strategy plan](docs/retcon-strategy.md).

## Build and test

Install a Java 17 JDK and set `JAVA_HOME` to that JDK. The checked-in Gradle wrapper downloads Gradle 8.14.3 on first use; a separate Gradle installation is unnecessary. Initial builds need internet access to download Gradle and Maven dependencies.

Linux/macOS:

```sh
./gradlew clean build
```

Windows PowerShell:

```powershell
.\gradlew.bat clean build
```

This compiles the source, runs JUnit 5 tests, and builds `AISurvivor/AISurvivor.jar` with its runtime dependencies. Run `./gradlew test` (Windows: `.\gradlew.bat test`) for tests alone. HTML test reports appear in `build/reports/tests/test/`. Tests cover geometry, targeting defaults, enemy fire accounting, bullet-wave matching, learning transitions, frozen evaluation, and checkpoint validation.

## Run a battle

The repository includes a ready-built [AISurvivor.jar](AISurvivor/AISurvivor.jar). Clone or download this repository and launch it from the `AISurvivor` directory with Java 17; Gradle is only needed to rebuild it. Keep the JSON descriptor and launchers beside the JAR. CI verifies that the checked-in JAR's file contents match a fresh source build.

The dependency is pinned to Tank Royale bot API **1.4.0**. Use the matching [Tank Royale 1.4.0 release](https://github.com/robocode-dev/tank-royale/releases/tag/v1.4.0). The upstream GUI/server requires Java 11 or newer; this bot is built for Java 17, so use Java 17 for both. Compatibility with other server versions has not been verified.

1. Install the Tank Royale GUI/server and sample opponents from that release.
2. Start the server from the GUI (default WebSocket address: `ws://localhost:7654`).
3. Add this repository's `AISurvivor` directory to the GUI's bot directories and boot the bot, or launch it manually from that directory:

   ```sh
   cd AISurvivor
   java -jar AISurvivor.jar
   ```

4. Select AISurvivor and sample opponents in the GUI and start a battle.

The launchers `AISurvivor.cmd` and `AISurvivor.sh` set the working directory for you. Manual launches must run inside `AISurvivor` so the descriptor is found. Learned model data is saved as `AISurvivor-qtable.dat` in that working directory; it is ignored by Git.

## GitHub Actions

[CI](https://github.com/stephanelee/robocode/actions/workflows/ci.yml) builds/tests with Java 17 on PRs and pushes to `master`, and uploads the **AISurvivor-java17** runnable artifact.

**For performance evaluation, start with the [Retcon 1v1 benchmark](https://github.com/stephanelee/robocode/actions/workflows/retcon.yml)** and dispatch `full=true`. It verifies the bot under Java 17, then uses JDK 25 for the unchanged Retcon launcher. For a local run after building with Java 17:

```sh
python tools/benchmark.py --suite retcon --java /path/to/jdk25/bin/java --rounds 5 --training-rounds 20 --timeout 900 --require-metrics --output build/retcon-evaluation
```

Use the [methods](docs/benchmarks.md), [Retcon baseline](docs/benchmarks/retcon-2026-11-02.md), and [three completed controlled experiments](docs/improvements.md) to interpret the evidence. These are in-roster adaptation trials; none of the three candidates met its full retain rule.

The older [Battle benchmark](https://github.com/stephanelee/robocode/actions/workflows/benchmark.yml) and the CLI's default sample roster remain **regression/integration checks**, with `full=true` producing the legacy sample reference comparison and standalone dashboard. They are not the primary strength benchmark. Existing commands remain supported; use `--suite retcon` explicitly for the stronger 1v1 direction.

## Current design

The bot entry point and combat strategies are in `src/main/java/AISurvivor.java`; learning, configuration, persistence, and pure combat calculations are separate classes:

- A small DQN (14 inputs, 32 hidden units, 6 action values) uses replay and a target network.
- Movement executes one selected strategy: minimum risk, clockwise/counterclockwise orbit, flee, wave surfing, or hunt. Replay credits the strategy actually executed.
- GuessFactor targeting combines per-enemy and per-class statistics with motion prediction.
- Consecutive enemy scans infer bullet waves after correcting known damage and hit-energy bonuses. Scan gaps are skipped because fire timing is uncertain. Enemy bullets are not exposed by the bot API; unobserved third-party damage in melee can still make inferred waves inaccurate.
- Surf danger learns only from a hit matching one inferred wave's shooter, power, and arrival radius. Targeting waves are added only after a `BulletFiredEvent` confirms the shot and are scored against fresh target scans.
- A versioned numeric checkpoint persists the DQN and class targeting statistics across launches.

This is an experimental bot. See [TODO.md](TODO.md) for completed repairs and the benchmark plan. Build success alone does not establish battle performance.

## Training, evaluation, and telemetry

Set these environment variables before launching the bot. Paths are relative to the bot working directory unless absolute. Invalid modes, seeds, and action IDs fail at startup.

| Variable | Default | Purpose |
| --- | --- | --- |
| AISURVIVOR_MODE | train | train, eval, or fixed |
| AISURVIVOR_SEED | 0 | Signed 64-bit seed for network initialization, exploration, and replay sampling |
| AISURVIVOR_MODEL_PATH | AISurvivor-qtable.dat | Checkpoint to load/save |
| AISURVIVOR_ACTION | 0 | Fixed strategy ID: 0 minimum risk, 1 clockwise orbit, 2 counterclockwise orbit, 3 flee, 4 wave surf, 5 hunt |
| AISURVIVOR_METRICS_PATH | unset | Optional JSONL output, one record per completed round |

Training uses Boltzmann exploration and updates model weights and targeting statistics. Evaluation chooses the highest-valued action, freezes learning and exploration temperature, and never writes the checkpoint. A missing checkpoint in evaluation creates a fresh seeded policy; check the telemetry's checkpointLoaded flag to distinguish trained from cold evaluation. Fixed mode always executes AISURVIVOR_ACTION and neither reads nor writes a checkpoint. Evaluation and fixed modes also freeze local GuessFactor and surfing statistics. Bot RNG seeding does not seed the game server or guarantee identical whole-battle outcomes.

At round completion, the final transition includes pending combat rewards plus +60 for winning or the existing death penalty of -(60 + 12 × remaining enemies). Duplicate completion notifications do not create duplicate transitions. Per-turn survival remains +0.1.

Checkpoints contain an explicit format version, dimensions, weights, counters, temperature, and class targeting statistics. Replay is deliberately not persisted; restarting begins a fresh seeded RNG stream, rather than resuming an interrupted training sequence exactly. Legacy Java-serialized or corrupt files are logged and left untouched when loading fails. On the next training save, they are backed up beside the original as .rejected-UUID before replacement. New checkpoints are validated and replaced atomically; a filesystem lacking atomic replacement leaves the previous checkpoint in place and reports an error.

For example, evaluate a trained checkpoint from PowerShell:

```powershell
$env:AISURVIVOR_MODE = 'eval'
$env:AISURVIVOR_SEED = '123'
$env:AISURVIVOR_MODEL_PATH = 'trained.dat'
$env:AISURVIVOR_METRICS_PATH = 'evaluation.jsonl'
java -jar AISurvivor.jar
```

Metrics include actual fired bullets and bullet hits, hit rate, damage dealt/taken, wall and bot collisions, skipped turns, outcome, accumulated reward, mode, seed, and checkpoint status. The benchmark runner can combine these records with official scores and placements.

