# Breakwater duel findings

All three Breakwater versions lost every round of their 20-round pilots against Diamond Port 1.8.28. Each missed both registered thresholds for confirmation, a score share above 20% and at least one first place. Against AISurvivor 1.2.1, the results ranged from zero wins for the interception controller to a sweep for the selective gun.

| Breakwater controller | Diamond wins | Diamond score share | AISurvivor wins | AISurvivor score share |
| --- | ---: | ---: | ---: | ---: |
| v1, bullet interception | 0/20 | 0.17% | 0/20 | 1.12% |
| v0.2, mobile counterfire | 0/20 | 2.05% | 5/20 | 24.98% |
| v0.3, reachable-path gun | 0/20 | 0% | 20/20 | 84.38% |

Each cell summarizes one fresh game for that controller and opponent. The six competitive games account for 120 rounds. A separate five-round v0.3 diagnostic used the v0.2 gun to fire live shots. These historical comparisons do not isolate the effect of an individual change or establish repeatability.

## What the guns actually did

The v1 controller spent 351 of its 368 shots on shields against Diamond. Thirteen intercepted a bullet. Only one of 17 deliberate attacks hit. The v0.2 controller kept moving and fired 603 attacks against Diamond, landing ten. It recorded 59.96 damage dealt and 1,396.37 taken. [v1 Diamond report](https://github.com/stephanelee/robocode/blob/e24f5d3a99ffa0e0c82bbc82e197bda1af960c1f/docs/improvements/breakwater-v1-results.md) · [v1 AISurvivor report](https://github.com/stephanelee/robocode/blob/e24f5d3a99ffa0e0c82bbc82e197bda1af960c1f/docs/improvements/breakwater-vs-ais.md) · [v0.2 report and raw game links](https://github.com/stephanelee/robocode/blob/e1f58155b31ef7d1a62ccc65a21b94b536c1325b/docs/improvements/breakwater-mobile-v2-exact-results.md)

The v0.3 diagnostic shows why an average across all aiming opportunities can mislead. On 2,556 opportunities resolved from later scans, the candidate ray intersected Diamond's observed footprint 1,263 times, versus 945 for the older ray. Every candidate intersection occurred on a turn when v0.3 would have held fire. In the competitive pilot, with v0.3 controlling the gun, it fired only three times in 31,392 opportunities, missed all three and lost 0–3,596. [Paired diagnostic and pilot report](https://github.com/stephanelee/robocode/blob/b9f61e536c0ff5860f6aabc0f6d3f4a11685a3d9/docs/improvements/breakwater-envelope-v3-exact-results.md)

The same v0.3 JAR beat AISurvivor 1,296–240 and won 17 of the 20 rounds without firing. Twelve of its 13 hits came in a single round. The official scoreboard credits Breakwater with all 1,000 survival points. The recorded events do not establish why AISurvivor lost each round. The Diamond rejection stands alongside this single-game transfer result. [Official games, round telemetry, frozen hashes and replay instructions](https://github.com/stephanelee/robocode/tree/b9f61e536c0ff5860f6aabc0f6d3f4a11685a3d9/benchmarks/breakwater-envelope-v3-exact)
