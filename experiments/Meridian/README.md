# Meridian 0.6.0

Meridian is an original 1v1 Java 17 Tank Royale bot. From the repository root:

```sh
./gradlew -p experiments/Meridian clean build
python3 tools/verify_jar.py experiments/Meridian/Meridian.jar experiments/Meridian/build/bot/Meridian/Meridian.jar
```

Load `experiments/Meridian/build/bot/Meridian` in the official booter; it contains the descriptor, launch scripts and fat JAR. To run the committed binary without Gradle, copy the three files in `bot/` beside `Meridian.jar` in a directory named `Meridian` and select that directory. It needs no checkpoint. Optional `MERIDIAN_SEED` determines starting orbit side; `MERIDIAN_METRICS` is a write-only round/skip log.

**From observation to decision.** At startup the radar sweeps and the bot heads toward the interior. Fresh scans provide position, signed speed, heading and energy. The radar steers past the last bearing so its next sweep crosses the target. Missing scans trigger reacquisition and suppress shooting. Motion prediction uses the [pinned turn rules](../../docs/improvements/meridian-platform.md).

For each consecutive scan pair we estimate enemy expenditure, subtracting known own-hit damage and adding the enemy hit bonus to the previous turn's energy delta. A plausible residual spawns a candidate bullet front, not an observed shot. Forecast both orbital directions for 50 steps using the same wall-aware steering used to drive. Prefer paths away from walls and from head-on and lateral-lead rays near the predicted arrival time; a switching cost discourages oscillation. When a confirmed incoming bullet uniquely matches an inferred launch, v0.2 also learns its offset from the launch bearing. Later routes incur risk near previously successful firing angles, mirrored to match the current lateral direction. The v1 rays supply the cold prior. [Learning rule, legal inputs and benchmark](../../docs/improvements/meridian-hit-angles.md). Wall, ram, inactivity, missed scans, incorrect origins and more complex guns can defeat this estimate.

The v6 gun applies a fixed bank of 206 observed target crossings from a previous duel. Each example votes for a bearing according to lateral and radial speed, forward wall clearance and estimated flight time. The same bank is used from the first shot against every opponent. Firing power, alignment, radar and movement follow v3. [Training source and prospective test](../../docs/improvements/meridian-prior.md).

**Opponent hypotheses.** The [pinned Retcon 0.2.0 surf planner](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/movement/WaveSurfMovement.java) ranks two trajectories using incoming waves, and its gun selector compares historical hit rates. A straight extrapolation may fail at a surf reversal. The [experimental Python Diamond Port 1.8.28](https://github.com/fduch-stranger/robocode-bot/blob/67b68c22e41c18281a3fce9ef638e3c23a0b47cf/bots/ports/diamond-port/diamond-port.py) combines a learned wave gun with wave surfing; a repeated orbit need not keep evading that gun. Neither analysis establishes a counter. Opponent code, observer recordings and metrics are never read by Meridian. [Original v1 evidence](../../docs/improvements/meridian-v1.md) and [v2 results](../../docs/improvements/meridian-hit-angles.md).
