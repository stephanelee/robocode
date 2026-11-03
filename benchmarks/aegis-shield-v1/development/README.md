# Aegis v1 development evidence (unregistered, exploratory)

These files are **not registered evidence and not a strength claim**. They are preserved so the development statements in the [registration](../../../docs/improvements/aegis-shield-v1.md) can be audited. All games were run locally in a Linux container (4 vCPU, JDK 21 runtime, Python 3.13) against the same pinned Diamond Port 1.8.28 source and wheels, loaded by the unchanged `tools/strong_melee_opponents.py`, via `tools/devgame.py`. They are not GitHub runner games. Nothing here is pooled into, or substitutes for, the registered pilot or confirmation.

| Path | Source state | What it shows |
| --- | --- | --- |
| `runs/stationary-probe/` | `tools/Probe.java`, stationary, never fires | 3 rounds: Diamond hit 33 of 33 shots that reached it. Line offsets from our centre were within about ±3 px, and exactly 0 once its KNN guns took over. |
| `runs/shift-probe/` | `Probe.java` with an occasional 12 px shift | Bullet lines pass exactly through the enemy scan one turn before the energy drop, which confirms the inferred origin. After our shifts, Diamond's aim moved by ±7.5–9.5 px, showing that our movement trains its labels. |
| `runs/instrumented-aim-bins-before-replay.txt` / `…-aegis22.txt` | earlier / near-final Aegis vs a **private dev-only instrumented copy** of the opponent that logs its labels and aim bins | Mean label \|GF\| 0.17 → about 0.015; centre-candidate aims rose to 938 of 977. The instrumented copy was never used in registered games or read at runtime. |
| `runs/aegis28/` | near-final policy (before buffer-reuse refactor and skip-turn logging) | 10 rounds: 0 wins, 395 of 533 inferred shots blocked (74.1%), 0 skips. `outcome-classes.txt` breaks down the failures. |
| `runs/frozen-smoke/`, `skip1/`, `skip2/` | final policy across the last performance refactors | 2 + 4 + 4 rounds, 0 wins; 1 own skipped turn in `frozen-smoke`. |
| `runs/driver-dry-run/` | exact frozen JAR `50a6b48f…` through `tools/aegis_experiment.py play()` (2 rounds, seed 1, unregistered game ID) | 0 wins, block rate 78.3%, **1 own skipped turn (turn 1339)**, maximum decision time 88 ms. |
| `runs/economics-simulation.txt` | `tools/econ.py` | Probability of winning a round as a function of block rate, using Diamond's power and power-down rules: 0.85 → 26%, 0.90 → 87%. |

Hit-line analyses used `tools/aimanal.py` and `tools/classify.py`. Development games after the label-replay planner consistently ended with Aegis dead after about 10–11 hits in short rounds, or after long low-power endgames. No development configuration won a round.
