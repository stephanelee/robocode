# AISurvivor

AISurvivor is an autonomous Java combat bot for [Robocode Tank Royale](https://robocode.dev/), built to survive incoming fire, adapt its aim and win battles. It combines programmed combat tactics with a small neural network and online statistical learning. It supports both 1v1 duels and multi-opponent melee through Tank Royale's WebSocket bot API.

Project notes: [documentation index](docs/README.md), [benchmark methodology](docs/benchmarks.md), [measured improvement log](docs/improvements.md), and [the source-grounded Retcon strategy plan](docs/retcon-strategy.md).

## How it fights

- **Movement:** the default training policy uses a dedicated duel controller when one enemy remains, reacting to estimated incoming shots and predicting safe movement within the arena. With several opponents, it selects among six strategies, including minimum-risk movement, orbiting, retreat and wave surfing.
- **Radar and targeting:** independent radar control tracks a duel opponent. An adaptive gun selector compares head-on, linear and circular predictions with a GuessFactor-adjusted forecast, which learns where an opponent tends to move relative to a shot.
- **Energy management:** firing decisions account for gun alignment, bullet travel, available energy and the damage needed to finish an opponent, including enemies disabled at zero energy.

AIS uses only its legal scans, own state and confirmed combat events. Incoming enemy bullets are inferred from observations such as energy changes; their positions are not directly visible to the bot.

## Where the AI comes in

A small **Deep Q-Network** maps 14 inputs to values for six movement actions. During training, it learns from combat rewards using replay memory and a target network. The default duel hierarchy selects its controller directly; the network continues training, but does not choose a different movement strategy on those turns. Neural action selection remains part of the multi-opponent and legacy learned policies.

Simpler online learning also adapts targeting and movement danger. The virtual gun selector scores predictions against subsequent legal scans; these geometric scores are estimates, not confirmed bullet hits. GuessFactor profiles learn from virtual wave passages, while surfing danger updates from hits matched to inferred enemy waves. Checkpoints retain the neural weights and class targeting statistics between launches. Learning runs locally without an LLM or external AI service.

**Measured wins validate the complete bot, not the neural network alone.** The strongest demonstrated gains came from coordinating movement, radar, aiming and energy handling; an isolated neural-learning benefit has not been established.

## Measured performance

| Test | Result | Scope |
| --- | --- | --- |
| [Fresh Retcon confirmation](docs/improvements/final-default-confirmation-redo.md) | 187/200 official first places; 86.7% mean score share; zero skipped turns | Ten fresh default-TRAIN games against pinned Retcon 0.2.0; all registered confirmation criteria passed |
| [Four-bot sample melee](docs/improvements/melee-default-baseline.md) | 60/60 official first places; rank 1 in all three games; 66.4% mean score share; zero skipped turns | AISurvivor versus Target, SpinBot and Walls; a sample-roster baseline |

These results apply to the documented opponent versions, settings and training regime. They do not establish broad tournament or advanced melee strength. Reports preserve the raw evidence, source and model hashes, and failures, including the separate invalid first Retcon confirmation attempt.

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

Use the [methods](docs/benchmarks.md), [Retcon baseline](docs/benchmarks/retcon-2026-11-02.md), and [measured improvement log](docs/improvements.md) to interpret the evidence. The [coherent default duel comparison](docs/improvements/coherent-duel-v1.md) passed its registered gates: Retcon share14.2Ã¢â€ â€™58.4%,39/60 wins, and30/30 self-match wins. Shooting accuracy fell; this supports the bundled bot, not an isolated gun/learning improvement or broad superiority claim.

The [subsequent gun-v2 comparison](docs/improvements/coherent-duel-v2.md) also passed every gate: Retcon share49.6Ã¢â€ â€™85.8%,55/60 wins, improved shooting metrics and30/30 wins against frozen v1. These are separate fresh three-game Retcon cohorts, not a pooled confirmation or a broad tournament result.

The [first final frozen-default attempt](docs/improvements/final-default-confirmation.md) was invalidated by one skipped AIS turn after140 of200 planned physical rounds. It remains unconfirmed and its cause unknown. The user then authorized a [separate fresh redo](docs/improvements/final-default-confirmation-redo.md): all200 rounds validated, with86.7% mean share,187/200 official first places,82.3% binding game-level lower bound and zero skips. All unchanged gates passed; no old data was pooled or gameplay changed. This confirms the frozen default bot against pinned Retcon under this regime, not all tournaments. No further automatic benchmark is scheduled.

The requested [four-bot sample melee baseline](docs/improvements/melee-default-baseline.md) completed60 defaultTRAIN rounds against Target/SpinBot/Walls: rank1 in all three games,60/60 official first places,66.4% mean share and zero skips. It also recorded184 wall/169 bot collisions; this is descriptive sample-roster evidence, not broad melee strength or a gameplay change.

The older [Battle benchmark](https://github.com/stephanelee/robocode/actions/workflows/benchmark.yml) and the CLI's default sample roster remain **regression/integration checks**, with `full=true` producing the legacy sample reference comparison and standalone dashboard. They are not the primary strength benchmark. Existing commands remain supported; use `--suite retcon` explicitly for the stronger 1v1 direction.

## Current design

The bot entry point is the thin adapter `src/main/java/AISurvivor.java`; `SurvivorBrain` owns combat decisions through `BotPort`. Learning, configuration, persistence, and pure combat calculations are separate classes:

- A small DQN (14 inputs, 32 hidden units, 6 action values) uses replay and a target network.
- Movement executes one selected strategy: minimum risk, clockwise/counterclockwise orbit, flee, wave surfing, or hunt. Replay credits the strategy actually executed.
- The default TRAIN duel composition uses the accepted shot-reactive `DuelSurf` controller, independent radar and a coordinated `DuelGun` fed only legal scans/confirmed shots. The [v2 composition](docs/coherent-duel-v2.md) supplies the existing GF/circular forecast as a real adaptive gun option. [Composition and scope](docs/coherent-duel.md). With several enemies, targeting follows the older melee path; default TRAIN can switch to duel components as opponents are eliminated. Legacy learned policy and EVAL/FIXED retain their existing targeting.
- Consecutive enemy scans infer bullet waves after correcting known damage and hit-energy bonuses. Scan gaps are skipped because fire timing is uncertain. Enemy bullets are not exposed by the bot API; unobserved third-party damage in melee can still make inferred waves inaccurate.
- Surf danger learns only from a hit matching one inferred wave's shooter, power, and arrival radius. Targeting waves are added only after a `BulletFiredEvent` confirms the shot and are scored against fresh target scans.
- A versioned numeric checkpoint persists the DQN and class targeting statistics across launches.

This is an experimental bot. See [TODO.md](TODO.md) for completed repairs and the benchmark plan. Build success alone does not establish battle performance.

## Training, evaluation, and telemetry

Set these environment variables before launching the bot. Paths are relative to the bot working directory unless absolute. Invalid modes, seeds, and action IDs fail at startup.

| Variable | Default | Purpose |
| --- | --- | --- |
| AISURVIVOR_MODE | train | train, eval, or fixed |
| AISURVIVOR_DUEL_POLICY | controller | TRAIN only: coherent duel controller; learned restores legacy DQN selection, gun, radar and timing |
| AISURVIVOR_SEED | 0 | Signed 64-bit seed for network initialization, exploration, and replay sampling |
| AISURVIVOR_MODEL_PATH | AISurvivor-qtable.dat | Checkpoint to load/save |
| AISURVIVOR_ACTION | 0 | Fixed strategy ID: 0 minimum risk, 1 clockwise orbit, 2 counterclockwise orbit, 3 flee, 4 wave surf, 5 hunt |
| AISURVIVOR_METRICS_PATH | unset | Optional JSONL output, one record per completed round |

Training updates model weights and targeting statistics. With the default duel policy, one live opponent and one cached legal scan select and credit action0 directly; otherwise TRAIN uses Boltzmann exploration. `AISURVIVOR_DUEL_POLICY=learned` restores the previous selection policy. This is a hierarchical policy change, not a learning-benefit claim. Evaluation chooses the highest-valued action, freezes learning and exploration temperature, and never writes the checkpoint. A missing checkpoint in evaluation creates a fresh seeded policy; check the telemetry's checkpointLoaded flag to distinguish trained from cold evaluation. Fixed mode always executes AISURVIVOR_ACTION and neither reads nor writes a checkpoint. Evaluation and fixed modes also freeze local GuessFactor and surfing statistics. Bot RNG seeding does not seed the game server or guarantee identical whole-battle outcomes.

At round completion, the final transition includes pending combat rewards plus +60 for winning or the existing death penalty of -(60 + 12 Ãƒâ€” remaining enemies). Duplicate completion notifications do not create duplicate transitions. Per-turn survival remains +0.1.

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

