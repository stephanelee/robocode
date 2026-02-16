# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Requires Java 17+ and Gradle 8+.

```bash
# Build fat JAR → outputs to AISurvivor/AISurvivor.jar
gradle jar

# Clean and rebuild
gradle clean jar
```

The `jar` task bundles the bot-api and all transitive dependencies into `AISurvivor/AISurvivor.jar` (configured in `build.gradle`).

## Running the bot

```bash
# From the AISurvivor/ directory (or via the Robocode booter)
cd AISurvivor
java -jar AISurvivor.jar
```

The bot connects to a running Robocode Tank Royale game server via WebSocket (default `ws://localhost:7654`). Launch the server first from the Tank Royale GUI before starting the bot.

## Project structure

```
robocode/
├── build.gradle                    # Gradle build — fat JAR → AISurvivor/
├── settings.gradle
├── src/main/java/
│   └── AISurvivor.java             # Bot source (single file, all logic here)
└── AISurvivor/                     # Deployable bot directory (Robocode format)
    ├── AISurvivor.jar              # Built output (gitignored if desired)
    ├── AISurvivor.json             # Bot descriptor (name, version, gameTypes)
    ├── AISurvivor.cmd              # Windows launcher
    └── AISurvivor.sh               # Linux/macOS launcher
```

The `AISurvivor-qtable.dat` file is written to the bot's working directory at the end of each round and reloaded at startup — this is how learning persists across battles.

## Architecture

### Bot API (Tank Royale)

`AISurvivor extends Bot` (from `dev.robocode.tankroyale.botapi`). The lifecycle is:

- `run()` — called once per round on its own thread; loops with `while (isRunning())`. Each loop iteration represents one game turn: set commands → `go()` commits them.
- Event handlers (`onScannedBot`, `onHitByBullet`, etc.) fire between `go()` calls on the same thread.
- `setAdjustGunForBodyTurn(true)` + `setAdjustRadarForGunTurn(true)` decouple the three components so each rotates independently.

### Q-Learning layer

| Component | Detail |
|---|---|
| State space | 1 920 states: energy(4) × dist(5) × bearing(8) × enemyCount(4) × wallProximity(3) |
| Action space | 6 strategies: anti-gravity, circle CW, circle CCW, flee, dodge, hunt |
| Algorithm | Tabular Q-Learning, TD(0) update, ε-greedy exploration |
| Rewards | +0.05/turn survival · +(damage×0.6) hit · +30 kill · +15 enemy death · −(damage×0.8) hit · −100 death · −3 wall |
| Persistence | `AISurvivor-qtable.dat` (Java serialisation) — survives across battles |
| Exploration | ε decays 0.40 → 0.05 at rate 0.992 per round |

### Movement strategies (actions)

| Action | Strategy |
|---|---|
| `A_ANTIGRAVITY` (0) | Anti-gravity resultant: wall repulsion + enemy repulsion + centre attraction |
| `A_CIRCLE_CW/CCW` (1,2) | Orbit nearest enemy at 90° perpendicular (maintain engagement range) |
| `A_FLEE` (3) | Anti-gravity with 2.5× enemy repulsion multiplier |
| `A_DODGE` (4) | Erratic turn + alternating forward/reverse on a prime cycle |
| `A_HUNT` (5) | Charge the lowest-energy enemy |

### Targeting

Linear predictive targeting: bullet travel time is calculated from current distance + firepower, then the enemy's position is extrapolated forward `ticks` turns using its last known velocity vector. Firepower scales from 0.5 (long range) to 3.0 (close range) with energy-conservation caps.

### Enemy tracking

Scanned enemies are stored in a `HashMap<Integer, EnemyInfo>` keyed by bot ID. Entries older than 40 turns are pruned each turn. Dead enemies are removed immediately via `onBotDeath` and `onBulletHitBot`.

## Key constants to tune

All in `AISurvivor.java` near the top:

- `LEARNING_RATE`, `DISCOUNT`, `EPSILON_*` — Q-learning hyperparameters
- `WALL_REPULSE`, `ENEMY_REPULSE`, `CENTER_ATTRACT` — anti-gravity force strengths
- `STALE_TURNS` — how long enemy scan data is trusted

## Bot API dependency

```
dev.robocode.tankroyale:robocode-tankroyale-bot-api:0.35.5
```

Check [Maven Central](https://mvnrepository.com/artifact/dev.robocode.tankroyale/robocode-tankroyale-bot-api) for newer versions and update `build.gradle` accordingly.
