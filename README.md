# AISurvivor

A Java bot for [Robocode Tank Royale](https://robocode.dev/). This project uses Tank Royale's WebSocket bot API, rather than the original Robocode Java engine.

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

This compiles the source, runs available JUnit 5 tests, and builds `AISurvivor/AISurvivor.jar` with its runtime dependencies. Run `./gradlew test` (Windows: `.\gradlew.bat test`) for tests alone. HTML test reports appear in `build/reports/tests/test/` when tests exist. The initial bootstrap has no gameplay tests; these are tracked in [TODO.md](TODO.md).

## Run a battle

The dependency is pinned to Tank Royale bot API **0.36.1**. Start with the matching [Tank Royale 0.36.1 release](https://github.com/robocode-dev/tank-royale/releases/tag/v0.36.1); compatibility with newer server versions has not been verified.

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

[CI](https://github.com/stephanelee/robocode/actions/workflows/ci.yml) runs on pull requests and pushes to `master`, and supports manual dispatch once the workflow is on the default branch. It builds with Java 17, runs available tests, and uploads the freshly compiled bot as the **AISurvivor-java17** artifact. Download it from a successful Actions run's Artifacts section and extract it into a bot directory. The pipeline does not yet run battles or measure playing strength.

## Current design

The implementation is in `src/main/java/AISurvivor.java`:

- A small DQN (14 inputs, 32 hidden units, 6 action values) uses replay and a target network.
- Movement currently blends six strategies: minimum risk, clockwise/counterclockwise orbit, flee, wave surfing, and hunt.
- GuessFactor targeting combines per-enemy and per-class statistics with motion prediction.
- Enemy energy changes infer bullet waves; Java serialization persists the DQN and class targeting statistics across launches.

This is an experimental bot with known correctness issues. See [TODO.md](TODO.md) for the prioritized repair and benchmark plan. Build success alone does not establish battle performance.
