# Aegis v1: bullet shielding against a predictable learned gun (registration)

Tracked in [issue #71](https://github.com/stephanelee/robocode/issues/71) (moved from #53 at the user's request). This document is the **immutable pre-outcome registration**. Results are appended only in the clearly separated outcome section added after execution. Nothing above that section changes once the registered run starts.

## Objective and identity

Build and measure a separate experimental 1v1 bot that beats the pinned **Diamond Port 1.8.28** (native Python Tank Royale port; not the championship Java Diamond). The bot is **Aegis 0.1.0**: descriptor `Aegis`, source and tracked JAR in [`experiments/Aegis/`](../../experiments/Aegis/README.md), and environment prefix `AEGIS_`. It does not subclass, rename, load or modify AISurvivor or DiamondChallenger. It has no persisted model and no cross-round state; every round resets all tracking. Production AIS source and `AISurvivor/AISurvivor.jar` (sha256 `b41b5de4…5914800`) stay unchanged. Other agents' concurrent outgoing-shot-context work on DiamondChallenger ([PR #65](https://github.com/stephanelee/robocode/pull/65)) is not touched or duplicated.

## Source-grounded hypothesis (what is different)

All six previous challengers (pressure, context gun, wave defence, wave-horizon correction, diagnostic, guarded learned-ray selector) tried to out-aim a strong surfer or out-dodge a strong learned gun. Each lost the damage exchange; the best registered pilot reached 8.3% share. Aegis is a different whole-bot design. It does not try to hit Diamond. Instead it makes Diamond's own gun predictable and then shoots its bullets down.

Source facts used (pinned `fduch-stranger/robocode-bot@67b68c22`, see [docs/opponents/diamond-port.md](../opponents/diamond-port.md)):

- Its 1v1 KNN guns log a wave from **every** scan (visit/virtual waves), not only firing waves. Labels are the centre of the footprint–ring intersection extent relative to the bearing to our scanned position at wave time (`wave.precise_intersection`). The query virtuality feature is 0, with weight 0.38 in TripHammer, so virtual waves dominate its neighbours.
- It aims from a classic-order self-prediction toward our position predicted one turn ahead, and selects among 59 candidate angles. With zero labels it picks the centre candidate, i.e. a deterministic head-on line.
- Its gun ignores bullet-hit-bullet events for learning (`GunDataManager.on_bullet_hit_bullet` only flags the wave).
- The pinned engine (`robocode-dev/tank-royale@c8ad3a8d`, Tank Royale 1.4) creates a bullet from the shooter's **pre-move** position with its **previous-turn** gun direction. It tests bullet–bullet collisions as exact segment intersections of `[tick, tick+1]` positions, after scans and before bullet–bot hits are applied. A bullet–bot hit computed in the same turn still applies.

Mechanism (only legal scans, own state and confirmed events at runtime):

1. Infer each enemy shot from consecutive radar-locked scan energy drops, with one-scan-late corrections (our damage, its hit bonus, shared inactivity damage). Origin = its scan one turn earlier. Line = head-on toward our predicted position, angle from an inferred classic self-prediction.
2. **Zero-label replay movement.** An exact pinned-physics dynamic programme moves along one axis and keeps us laterally where we were one wave-flight earlier, so its labels stay at GF 0 for every wave. It also asks for full speed in the four turns before each predicted enemy shot, which gives the shield crossing geometry.
3. **Interceptors.** 0.1-power bullets are solved against the exact engine contract, maximising coverage of ±4 px lateral line error. Plans lock the trajectory until intercept, schedule reload for the next shot, and dodge weakly covered shots.

Measured development facts (unregistered; preserved as exploratory only, not evidence of strength):

- Stationary probes were hit by 100% of Diamond's shots, with lines within ~3 px of centre, so the aim model is real.
- An instrumented private copy of the opponent (dev-only; never used in registered games or at runtime) reproduced our label model to 1e-6°. With replay movement it chose its centre candidate in about 97% of aims (mean |label GF| 0.013 vs 0.17 before replay).
- The last unregistered 10-round development game blocked 395/533 inferred shots (74.1%) and won **0/10** rounds; earlier variants reached about 80%.

**Prediction stated before the registered run: this pilot is expected to fail its progress gate.** Simulation of the energy arithmetic indicates about 90% blocking is needed to win most rounds. Each leak costs us 9.75 and refunds the opponent 5.85, and at the opponent's low-energy cubic power-down (0.1–0.3 power) a fixed 0.1 interceptor costs as much as its bullet. The design is nevertheless coherent and source-grounded, and it is registered unchanged rather than tuned further. Its measured mechanism (inference coverage, block rate, failure categories, aim stability) is the deliverable if it fails.

Main counters and risks: residual opponent self-prediction errors of up to about 3 px; occasional adjacent-candidate aims (≈12 px) when labels drift; reload conflicts when contacts compress below 12 turns; early-battle data before labels settle; and the low-power endgame. No gun offence. **Timing is a material gate risk.** Final-policy development rounds had own skipped turns at roughly 1 per 6 rounds (2 in 12: `frozen-smoke` and the exact-frozen-JAR driver dry run, skip at turn 1339), or 1 per 11 including the near-final 10-round game. The frozen JAR's measured maximum decision time was 88 ms against a 100 ms turn limit; typical peaks were 20–45 ms. Some skips occurred on turns whose own decision time was only about 24 ms, so the cause is likely GC, scheduling or host contention with the Python opponent rather than planner cost alone; it was not resolved. Because any own skip fails the gate, the pilot outcome may be decided by timing rather than strategy.

## Frozen inputs

| Item | Pin |
| --- | --- |
| Aegis source commit | `7fd5b8b2064dd66de0aa9f9076064e5cc70cd00e` (`experiments/Aegis/`, 11 JUnit tests) |
| Aegis tracked JAR | `experiments/Aegis/Aegis.jar` sha256 `50a6b48fc1c20c1153c02702de3c69d345f12bf02125e3eb67585e7bf0505e24`; CI rebuilds with Java 17 and requires `tools/verify_jar.py` parity, then runs the tracked bytes |
| Production AIS | unchanged vs `9b1624da`; `AISurvivor/AISurvivor.jar` sha256 `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800` |
| Opponent | Diamond Port 1.8.28, `fduch-stranger/robocode-bot@67b68c22e41c18281a3fce9ef638e3c23a0b47cf`, manifest tree sha256 `a9bf0b5d62b510ce3a405db9dcbe11cf2c1bce3f982e2a5d12b9fdc76b8fbc92`, 12-wheel Python 3.13 / bot API 1.3.1 runtime via `tools/opponents/strong-melee.lock.json` (sha256 `ddd14064…df20c3`) and the unchanged `tools/strong_melee_opponents.py` loader |
| Engine/runner | Tank Royale 1.4.0 `robocode-tankroyale-runner-1.4.0.jar` sha256 `02f6d1d8…5f327cf`, `tools/BenchmarkBattle.java` (sha256 `3646c043…77988c1`): 1v1, default 800×600 arena, turn timeout 100 ms |
| Runtime | GitHub Actions `ubuntu-26.04`; Java 17 build/tests/JAR parity, then JDK 25 to run games; Python 3.13 |
| Policy | Cold game; no model; `AEGIS_SEED` recorded but **does not randomise Aegis** (deterministic policy), so game-to-game variation comes only from the opponent, spawn positions and scheduling; trace OFF |

## Budget, seeds, game IDs

- **Pilot:** one fresh game `aegis-shield-v1-pilot`, 20 rounds, `AEGIS_SEED=20263801`.
- **Confirmation (only if the pilot passes):** three fresh games `aegis-shield-v1-confirm-g01..g03`, 20 rounds each, seeds `20263811`, `20263812`, `20263813`.
- Each game is a new process with a fresh metrics path. Nothing from development games, the pilot, or any other experiment is pooled into confirmation.

## Decision rules (fixed before outcomes)

**Pilot progress** requires all of: official score share > 20%; at least one official first place; zero own skipped turns; and valid, complete evidence (20 telemetry rows with identity/round checks, terminal flags agreeing with official first places, verified opponent source hashes and JAR hash).

**Confirmation** (fresh 60 rounds) requires all of: equal-game mean score share > 50%; more than 30/60 official round wins; at least two of three games above 50% share; zero own skipped turns; and valid evidence. A descriptive game-level t bound is reported but is not a gate.

Exactly one labelled execution counts; removing and re-adding the label or re-running jobs is forbidden, and any such execution would be reported as invalid/exploratory. Any rerun, retry or extra game is reported as invalid or exploratory and never replaces or joins the registered data. There is no seed selection, threshold change, favourable rerun or reuse of spent data. Failed and invalid attempts are preserved. No victory is claimed from proxies such as block rate.

Metrics reported regardless of outcome: shares, first places, survival/bullet-damage scores, inferred shots, shields fired, blocks, hits taken, block rate, outcome categories, mean label error, decision-time maximum and slow-turn counts, skipped turns, wall and bot collisions, and turns per round.

## Execution

`tools/aegis_experiment.py` runs the pilot and, only on pass, the confirmation, writing `experiment.json` plus all raw rows and logs. The workflow `.github/workflows/aegis-shield-v1.yml` runs on `ubuntu-26.04` only when the PR carries the label `run-aegis-v1`. It first proves the frozen source/JAR and unchanged production. Independent review of the exact implementation and this protocol precedes adding the label.

Amendment before any registered game (independent review of `dfff8336`): corrected the timing-risk statement above from development evidence, noted seed determinism, moved tracking to #71, and fixed a driver validation bound that could have invalidated a low-energy round (fire commands that the engine refuses below 0.1 energy). Bot source and JAR are unchanged (`7fd5b8b2`, `50a6b48f…`). The driver and workflow are frozen by SHA-256 in the workflow.

<!-- Outcome section is appended below this line after execution. -->
