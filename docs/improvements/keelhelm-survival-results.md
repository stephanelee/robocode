# KeelHelm melee survival

## Where damage arrived

Of the 1,920.22 points of incoming bullet damage credited during melee, 1,038.78 came from Diamond Port and 763.99 from Prism. Keel landed nearly as many hits as Diamond but used much less damaging shots. Melee here means more than one opponent alive on KeelHelm’s immediately preceding decision tick. Lethal hits remain in the attribution.

| Shooter | Melee hits | Raw damage | Credited damage | Last-two hits | Last-two credited damage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diamond Port | 219 | 1,055.95 | 1,038.78 | 296 | 285.77 |
| Prism | 117 | 766.18 | 763.99 | 69 | 254.36 |
| Keel | 211 | 118.00 | 117.46 | 59 | 23.47 |
| **Total** | **547** | **1,940.13** | **1,920.22** | **424** | **563.60** |

All 547 melee hit events came from distinct incoming bullets. The bullet’s power gives the nominal energy subtraction before overkill. Score credit caps that value at the victim’s nonnegative energy immediately before the hit. The last-two phase yielded 570.02 nominal damage over 61,227 live decisions. Eleven of those 424 hits came from bullets fired during melee; all 547 melee impacts followed melee firing decisions. Every hit had both preceding states and a linked firing event. The live ticks include nine lethal melee bullet hits and 12 lethal last-two hits. No melee wall or ram contact was recorded for KeelHelm. There were six wall contact events and zero ram contacts in the last-two phase. Wall events omit pre-impact speed and damage. Their individual energy cost is unresolved from those events alone, and repeated wall touches can be suppressed by the server. Mirrored ram notifications are deduplicated by bot pair per tick.

| Pre-impact condition in melee | At-risk ticks | Hits | Credited damage per 100 ticks |
| --- | ---: | ---: | ---: |
| Absolute speed below 4 | 7,701 | 39 | 2.11 |
| Absolute speed 4 to below 8 | 5,112 | 28 | 2.72 |
| Full speed (8) | 47,012 | 480 | 3.44 |
| Wall clearance below 80 | 32,779 | 266 | 2.56 |
| Wall clearance at least 80 | 27,046 | 281 | 4.00 |
| Nearest opponent below 250 | 55 | 1 | 9.45 |
| Nearest opponent 250 to below 500 | 13,356 | 157 | 4.49 |
| Nearest opponent at least 500 | 46,414 | 389 | 2.83 |
| Absolute turn rate below 2° | 34,205 | 315 | 3.13 |
| Absolute turn rate at least 2° | 25,620 | 232 | 3.32 |

Firing range and distance just before impact measure different things. At launch, 452 of the incoming hits came from at least 500 units away and cost 1,654.6 credited points. The descriptive rates motivated testing a short speed change while moving straight in open space. They cannot establish how much changing speed would help. [Per-game attribution and denominators](../../benchmarks/keelhelm-survival/incoming.json)

## Did slowing down reduce incoming damage?

**Keep accepted KeelHelm 0.2.1.** The speed dip removed more contacts than it introduced inside the eight-tick windows. It failed the preregistered test once we accounted for threats that were still live when control returned to ordinary movement.

The proposal briefly reduced signed target speed from 8 to 4, then accelerated back to 8 along the same heading. It could trigger only on a fixed turn boundary, in open space, while moving straight at full speed, with a fresh own radar scan and all fresh scanned opponents at least 500 units away. Starting poses came from actual games. Decisions used only own state and own scans. Recorded opponents, live enemy rays, public contacts and deaths were used separately to score or truncate the trajectories.

| Incoming fire | Actual hits | Projected hits in window | Original nominal damage | Projected in-window damage | Conservative endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diamond Port | 29 | 26 | 135.20 | 111.22 | 151.18 |
| Prism | 7 | 5 | 41.05 | 30.05 | 71.05 |
| Keel | 20 | 17 | 8.00 | 6.80 | 8.40 |
| **Total** | **56** | **48** | **184.25** | **148.07** | **230.63** |

Across 556 nonoverlapping windows, the changed path avoided 13 original contacts and introduced five. Immediate nominal damage fell 19.64%. At the end of each window, the registered rule withheld 36.58 points of apparent savings on original bullets that had not demonstrably passed the bot. It charged 45.99 for additional rays able to reach the changed bot’s four-tick movement envelope. These charges describe unresolved exposure. Neither charge represents an observed hit. Together they put the conservative endpoint 25.17% above the baseline. No window needed a further truncation penalty.

The sample met the 30-hit minimum, and neither path encountered a wall or ram during the test. The required 5% conservative reduction, gains in at least seven games, and protection against increased Diamond damage all failed. Three games had positive conservative savings, three tied and six lost ground. No new bot was built for competition and no fresh duels or melee games were played. [Window results by game and shooter](../../benchmarks/keelhelm-survival/summary.json)

## What this comparison can measure

The 12 original games / 60 rounds were a retrospective cohort of KeelHelm Crossfire 0.1.0 with pinned Diamond Port 1.8.28, Prism 0.2.0 and Keel 0.1.0. Crossfire made zero alternate fire requests there. Its MeleePolicy and Phys sources match accepted KeelHelm 0.2.1 byte-for-byte. The study consumed zero new games and rounds.

On minimind the reader ran the measured movement selector on each original melee decision. Across all 4,424 eligible paired ticks it matched the native own position, heading, speed and original incoming physical bullet IDs. Four windows ended early at a phase or terminal boundary; all were retained. Blank or unavailable preceding states are excluded from state-specific attribution. The radius-18 contact check uses the pinned server’s post-advance current-to-next bullet segment. The changed path earns no protection from unobserved own bullets. A ray that survives an original hit or own-bullet collision continues on its fixed path, subject to independently observable terminals. A starting clearance of at least 80 units bounds an eight-tick pulse away from walls at the native speed limit. A possible ram triggers censoring and a 0.6 damage charge. There were zero such windows.

This estimates local exposure to historical fire on the same starts. Anchoring the next window restores the recorded position and state. Opponents could aim, move, fire and scan differently in a live game. The endpoint does not forecast actual round survival, and the windows are clustered within games. [Registered protocol](keelhelm-survival-protocol.md) · [Issue #147](https://github.com/stephanelee/robocode/issues/147) · [Independent pre-score clearance](https://github.com/stephanelee/robocode/issues/147#issuecomment-6991010730) · [Baseline](../../benchmarks/keelhelm-survival/baseline.json) · [Proposed path](../../benchmarks/keelhelm-survival/pulse.json) · [Original private recording custody](../../benchmarks/keelhelm-crossfire/stopped-attempt/artifact-proof.json) · [PR #148](https://github.com/stephanelee/robocode/pull/148)

A separate reviewer verified the pinned engine movement and collision semantics, reproduced the baseline before the proposed path was scored, and independently reproduced the scored output byte-for-byte. The pulse JSON SHA-256 is `ac1476d89498d764caa478d0790a94b9580737967dc80e273f512520d7b1b3ae`. The private diagnostic archive retains the exact source, compiled reader, run outputs and original hash index. [Diagnostic archive proof](../../benchmarks/keelhelm-survival/artifact-proof.json)

## Reproduction

Execution head `7789fef5d18a28a3fa63391dbf0eff598954a8bd`. The reader uses measured candidate JAR `fe6498d461ebc6c92e5d2d03e047c91771166c3d339e49ed647cf47966164d96`, server `16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c`, pinned Temurin 25.0.5+8 and Python 3.12.16. Every recording hash and the original private bot telemetry are validated against the [published independent hash index](../../benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json).

From the worktree root, set EVIDENCE to the extracted original evidence directory with games and server.jar. Use the pinned JDK and Python on PATH.

```sh
classes=$(mktemp -d)
cp="$classes:experiments/KeelHelmCrossfire/KeelHelmCrossfire.jar:$EVIDENCE/server.jar"
python3 tournament/tools/keelhelm_incoming.py "$EVIDENCE" > "$classes/incoming.json"
javac -cp "$cp" -d "$classes" tournament/tools/CrossfireReplay.java tournament/tools/SurvivalPulseReplay.java
for mode in baseline pulse; do
  java -Xmx256m -cp "$cp" SurvivalPulseReplay "$EVIDENCE" "$mode" > "$classes/$mode.json"
done
for name in incoming baseline pulse; do
  cmp "$classes/$name.json" "benchmarks/keelhelm-survival/$name.json"
done
```
