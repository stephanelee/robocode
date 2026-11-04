# Keel (experimental 1v1 bot)

An isolated experimental Tank Royale 1.4.0 duelist. It is separate from production AISurvivor and shares no code or checkpoints with it.

| File | Responsibility |
| --- | --- |
| `Phys` | Pinned-engine physics: the speed law (zero target stops instantly; +8→−8 gives +3), move on old heading then turn at the new-speed limit, wall constraint |
| `Track` | Per-round legal motion history and the 8-feature context vector (lateral/advancing speed, flight time, acceleration, time since direction change, forward/reverse wall room, 10-turn displacement) |
| `Knn` | Linear-scan nearest-neighbour log |
| `Surfer` | Enemy waves inferred from scan energy drops (with corrections for our hits, their hit refunds, ram and wall damage). Engine-exact hit segment `[(T−t+1)s, (T−t+2)s]`. Precise 18px-circle coverage. Orbit or stop-at-k options with a second-wave lookahead. KNN hit danger plus a context flattener. Optional bullet shadows (off in v1) |
| `Gun` | Per-scan virtual waves; KNN main gun and real-wave anti-surfer gun; virtual-gun score selection |
| `Keel` | Bot adapter: radar lock, energy-conserving fire policy, telemetry markers (`KEEL_METRICS`), seed (`KEEL_SEED`) |

Build with Java 17: `./gradlew -p experiments/Keel clean build`. The runnable bot is `build/bot/Keel/`. The committed `Keel.jar` must match the source (`tools/verify_jar.py`).

Registration and results: [docs/improvements/keel-v1.md](../../docs/improvements/keel-v1.md).
