# Meridian 0.1.0

Meridian is an original 1v1 Java 17 Tank Royale bot. From the repository root:

```sh
./gradlew -p experiments/Meridian clean build
python3 tools/verify_jar.py experiments/Meridian/Meridian.jar experiments/Meridian/build/bot/Meridian/Meridian.jar
```

Load `experiments/Meridian/build/bot/Meridian` in the official booter; it contains the descriptor, launch scripts and fat JAR. To run the committed binary without Gradle, copy the three files in `bot/` beside `Meridian.jar` in a directory named `Meridian` and select that directory. It needs no checkpoint. Optional `MERIDIAN_SEED` determines starting orbit side; `MERIDIAN_METRICS` is a write-only round/skip log.

**From observation to decision.** At startup the radar sweeps and the bot heads toward the interior. Fresh scans provide position, signed speed, heading and energy. The radar steers past the last bearing so its next sweep crosses the target. Missing scans trigger reacquisition and suppress shooting. Motion prediction uses the [pinned turn rules](../../docs/improvements/meridian-platform.md).

For each consecutive scan pair we estimate enemy expenditure, subtracting known own-hit damage and adding the enemy hit bonus to the previous turn's energy delta. A plausible residual spawns a candidate bullet front, not an observed shot. Forecast both orbital directions for 50 steps using the same wall-aware steering used to drive. Prefer paths away from walls and from head-on and lateral-lead rays near the predicted arrival time; a switching cost discourages oscillation. Neither ray presumes access to the enemy gun. Wall, ram, inactivity, missed scans, incorrect origins and more complex guns can defeat this estimate.

The gun considers stopping, straight coasting and constant observed turn rate. Every fourth fresh scan it stores three predicted bearings. A later real scan at the selected forecast horizon scores their angular misses in units of target radius. Only these past scores affect subsequent gun selection. Gun scores persist across rounds of one process; scans, paths and pending labels reset each round. The cold prior favors coasting. Power falls with distance, low energy, or a nearly disabled target. The current gun ray must fit a 12-unit tolerance before firing. The three forecasts can all miss after a reversal; the incoming planner also does not model ramming contacts or bullet shadows.

**Opponent hypotheses.** The [pinned Retcon 0.2.0 surf planner](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/movement/WaveSurfMovement.java) ranks two trajectories using incoming waves, and its gun selector compares historical hit rates. A straight extrapolation may fail at a surf reversal. The [experimental Python Diamond Port 1.8.28](https://github.com/fduch-stranger/robocode-bot/blob/67b68c22e41c18281a3fce9ef638e3c23a0b47cf/bots/ports/diamond-port/diamond-port.py) combines a learned wave gun with wave surfing; a repeated orbit need not keep evading that gun. Neither analysis establishes a counter. Opponent code, observer recordings and metrics are never read by Meridian. [Registration and results](../../docs/improvements/meridian-v1.md).
