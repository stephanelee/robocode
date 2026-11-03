# Development guide

## Build and validate

Use a Java 17 JDK and the checked-in Gradle 8.14.3 wrapper:

```sh
./gradlew clean build
./gradlew test
```

On Windows use `.\gradlew.bat`. The build creates the fat JAR at `AISurvivor/AISurvivor.jar`; HTML test reports are in `build/reports/tests/test/`. The runnable JAR is intentionally tracked in Git. Rebuild and commit it whenever bot source or runtime dependencies change. CI preserves the checked-in JAR, builds from source, compares every file entry with `tools/verify_jar.py` (ignoring ZIP timestamps/order), and publishes the fresh bot directory as an Actions artifact. A stale or otherwise different checked-in JAR fails CI.

The dependency is `dev.robocode.tankroyale:robocode-tankroyale-bot-api:1.4.0`. Use the matching Tank Royale 1.4.0 GUI/server and sample bots for battle validation. From the `AISurvivor` directory run `java -jar AISurvivor.jar` after starting the server. See README.md for installation and environment variables.

## Source layout

| Class | Responsibility |
| --- | --- |
| `AISurvivor` | Bot lifecycle, observations, radar, six movement strategies, targeting and combat events |
| `DuelSurf`, `AimCorridor` | 1v1 shot-reactive movement for action 0: forward, backward or stop along an orbit line, chosen by simulating clearance from reconstructed head-on and linear aim points of inferred waves |
| `DQNAgent`, `MLP` | Seeded strategy selection, replay, target network, neural network updates |
| `BotConfig` | Validated environment configuration for train/eval/fixed modes and paths |
| `ModelStore` | Versioned numeric checkpoint validation, rejected-file backup and atomic replacement |
| `RoundLifecycle` | Pending rewards, terminal transitions and duplicate completion suppression |
| `RoundMetrics` | Optional per-round JSONL telemetry |
| `CombatMath` | East-zero geometry, circular prediction, orbit headings and bounded targeting-wave observations |
| `EnemyFireTracker`, `EnemyWave` | Observable energy bookkeeping, inferred enemy fire and hit-wave matching |

## Lifecycle and learning

`AISurvivor extends Bot`. `run()` observes features, credits prior rewards, executes one selected action, sets radar/gun commands and calls `go()`. Event handlers collect combat rewards and telemetry between turns. `onRoundStarted` clears transient combat tracking and resets episode state; `RoundLifecycle` closes each round exactly once, including pending rewards and the terminal outcome. Death/win handlers record the outcome; the round-ended handler closes the terminal transition after final-turn rewards have arrived.

The DQN uses 14 continuous features, one 32-unit ReLU hidden layer, and six action values. Features include energy, distance, relative energy, enemy count, wall proximity, gun heat, inferred approaching-wave count, radial target speed, time since inferred fire, bot speed, target bearing components, target speed and turn rate. Actions are minimum risk (0), clockwise orbit (1), counterclockwise orbit (2), flee (3), wave surf (4), and hunt (5). In a 1v1 with a scanned opponent, action 0 runs the `DuelSurf` controller instead of minimum risk (retained in `docs/improvements/duel-surf-v2.md`; its simulation follows the pinned engine's speed rule and move-then-turn order). The other actions and melee are unchanged. Replay records the action actually executed; actions are not blended.

Training uses Boltzmann exploration with temperature 1.0 decaying by 0.995 per completed episode to 0.15. Replay capacity is 5000; training begins after 64 entries, samples batches of 32, and synchronizes the target network every 100 training observations. Learning rate is 0.002 and discount is 0.95. Network initialization, exploration and replay share a seeded RNG.

Evaluation selects the greedy action and freezes network learning, temperature, targeting/surf statistics and checkpoint writes. Fixed mode executes the configured action and neither reads nor writes a model. A seed controls bot randomness, not the server or operating-system scheduling; whole battles are not guaranteed identical.

Rewards are +0.1 per turn, +0.6 times damage dealt, +30 for a lethal hit, +15 for an observed enemy death, -damage taken times `(0.6 + 0.1 * remainingEnemies)`, -3 for a wall collision, -1.5 for a bot collision, +60 for winning, or `-(60 + 12 * remainingEnemies)` for death. Metrics preserve combat rewards independently of whether learning is enabled.

## Combat geometry and statistics

Tank Royale uses Cartesian coordinates: east is 0 degrees and north is 90 degrees. Use `x += cos(angle) * distance`, `y += sin(angle) * distance`, and `atan2(dy, dx)`. Clockwise orbit relative to the inward bearing is `bearingToEnemy + 90`. `CombatMath` tests these conventions. Estimate target turn rate by dividing the normalized direction change by elapsed scan turns.

GuessFactor targeting has 47 bins and nine distance/lateral-speed segments. Empty or centre-tied profiles select the centre. Motion prediction integrates a constant turn rate continuously into the linear case, includes scan age, and iterates bullet intercept time. A targeting wave is added only after `BulletFiredEvent` confirms a shot. Score it using a fresh target scan within one bullet-turn plus collision radius of its expected arrival; discard observations that are already too late.

`TargetingGeometry` takes a frozen aiming snapshot and returns the current projected point and gun-turn geometry. It performs no API calls or state updates. `AISurvivor.aimGF` retains target/power/stat selection, the existing firing guard, command application, and pending-wave creation; the extraction preserves the accepted five-iteration arithmetic and wall-clamp behavior. Actual accepted-method command/state goldens are under `benchmarks/targeting-refactor/`.

`TickEvent.getBulletStates()` exposes this bot's bullets, not enemy bullets. Enemy threats use waves inferred only between consecutive scans. Inference runs after the turn's handlers so known damage from our bullets and the enemy's energy bonus from hitting us can be corrected without depending on handler priority. Gaps are skipped because shot time is uncertain. Unobserved third-party melee damage can still create false inferences; this is a documented limitation, not exact bullet tracking.

Surf danger learns only from one plausible hit wave matched by shooter, bullet power and arrival radius. Passing waves do not train danger. Per-enemy targeting/surf statistics are local to the process and decay by 0.92 per round in training; only class targeting statistics are persisted with the model.

## Configuration and persistence

`AISURVIVOR_MODE` is `train` (default), `eval` or `fixed`. Other settings are `AISURVIVOR_SEED` (default 0), `AISURVIVOR_ACTION` (0..5), `AISURVIVOR_MODEL_PATH` (default `AISurvivor-qtable.dat`) and optional `AISURVIVOR_METRICS_PATH`. Relative paths use the bot working directory. Model and metrics paths must differ.

Despite its legacy filename, the model is not a Q-table or Java object stream. `ModelStore` stores numeric weights, target weights, dimensions, counters, temperature and class GF statistics in an explicit versioned format. It validates sizes and finite values before using or replacing a checkpoint. Legacy/corrupt files are left intact on failed load and backed up before the next training replacement. Replay and RNG stream position are not saved; resuming a checkpoint does not exactly resume a prior process.

Tests cover geometry, targeting defaults, fire accounting, surf hit matching/frozen evaluation, learning transitions, mode configuration, metrics and checkpoint validation. Battle strength must be established separately with fixed opponents and recorded measurements; a green build is not a performance claim. Keep TODO.md aligned with verified work.
