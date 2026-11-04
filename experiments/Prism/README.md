# Prism

Prism is a standalone experimental Java 17 Tank Royale bot. It was implemented independently from [the clean-room specification](../../docs/diamond-port-clean-room-specification.md) (version 1, 2026-11-04). Wherever the specification says Facet, Prism is used instead. Prism shares no code, descriptor, JAR, configuration, model or telemetry with AISurvivor or any other experimental bot.

**Status: registered screen rejected; no confirmation.** The sole200-round screen completed: Diamond28.61% mean score share missed the >40% gate; Retcon94.33%, Adaptive69.29% and samples100% passed their gates. Zero Prism recorded skips/collisions/failures. [Full original evidence and limits](../../benchmarks/prism-screen-v1/README.md) include the missing Diamond game3 round15 telemetry line. This remains a standalone experimental reference, with no production AISurvivor promotion or fresh superiority claim.

**Current standalone reference: Prism0.2.0, profile `prism-profile-2-surfer-gun` (`Prism.jar f9c75a86...`), screen rejected.** Its13-game/200-round v2 screen missed only the Diamond>40% gate (30.34%); Retcon99.34%, Adaptive63.43% and samples100% passed their gates. Recorded skips/collisions/failures were zero. [Exact v2 evidence, maintenance provenance and limits](../../benchmarks/prism-screen-v2/README.md). No confirmation or production promotion; v1 results above describe profile1 only.

## What it does

| Area | Implementation |
| --- | --- |
| Engine boundary | `Prism` (adapter) → `Brain.turn` runs one transaction per tick after `go()` returns. Events are deduplicated. Order: lifecycle and own bullets → scans and energy ledger → wave labels → movement → validate prior firing plan → next aim and radar. |
| Physics | `core.Physics` reproduces the engine order and laws. Speed updates first. Translation uses the start-of-turn heading. Rotation is limited by the new speed. Target speed 0 stops instantly. Reversal follows the crossing law. Wall clamping is applied twice along the travel line, with first-contact damage. All of this is checked against a recorded 1.4.0 trace. |
| Waves | `wave.Wave` has the roles `OUTGOING` (engine-confirmed), `VIRTUAL`, `INCOMING` (inferred) and `FORECAST`. The shell is `[(t-T+1)b, (t-T+2)b]`. The footprint is the exact angular extent of the circular bot ∩ the shell (`core.Geometry`). |
| Guns | Analytic constant-velocity/constant-turn interception (cold start; no evolved data). Long-history contextual GF gun. Three-window recent gun. Melee displacement gun across all live targets. A virtual selector with frozen predictions picks among them (`gun.Guns`). |
| Firing | Power policy and kill cap. A plan made on turn t is validated on turn t+1 against the observed gun heading. A request stays pending until `BulletFired` confirms it, and an unconfirmed request is cancelled. Firepower is cleared explicitly whenever the gun is cool (`gun.Firing`). |
| Inference | Per-enemy energy ledger aligned to scan visibility, with gun-heat intervals. Ambiguity creates uncertainty, not shots (`track.FireInference`). Power prediction for forecasts (`track.PowerModel`). |
| Defence | Two-wave, three-action (NEG/STOP/POS) surfing with branch-and-bound and a 2,400-step quota. Learned theoretical-GF danger views and a visit flattener. A conservative fallback applies when geometry is missing. Bullet shadows come from confirmed own bullets only (`move.*`). |
| Melee | Minimum-risk destinations with 8-step predictor safety. The all-unsafe case returns a labelled emergency control (`move.MeleePlanner`). |
| Radar | Fresh-scan lock with overshoot, sweeps when scans are missed or stale, stalest-first in melee (`radar.Radar`). |
| Bounds | 256 waves per enemy, 512-tick histories, FIFO stores (8,192 long / 128, 512, 2,048 recent / 1,024 hits / 2,048 visits) and a 128 MiB global store guard. Failures in one subsystem degrade it to its analytic fallback. |

Parameters are the specification's independent initial profile in `Profile.java`, plus one registered change in profile 2: the `surfer` gun (windows 32/128/512, 5 neighbours, eligible after 8 confirmed-wave labels, chosen only by the unchanged virtual selector). Assumptions and deviations are recorded in the [implementation decision log](../../docs/prism-implementation-decisions.md).

## Build and test

Use JDK 17 and the repository's Gradle wrapper:

```sh
./gradlew -p experiments/Prism --no-daemon clean build
python tools/verify_jar.py experiments/Prism/Prism.jar experiments/Prism/build/bot/Prism/Prism.jar
```

The build writes the runnable bot directory to `build/bot/Prism/`. `Prism.jar` is committed, and CI checks source/JAR parity. Two clean builds produce byte-identical JARs. If the checkout lives in a synced folder that locks files, set `PRISM_BUILD_DIR` to a directory outside it.

There are 41 JUnit tests: `CoreTest`, `LearningTest`, `BehaviourTest` (including the PR #105 review regressions) and `EngineTraceTest`. `EngineTraceTest` replays real Tank Royale 1.4.0 recordings (see [fixtures](fixtures/README.md)).

## Run

Start any Tank Royale 1.4.0 server, then run `Prism.sh` or `Prism.cmd` from the bot directory. The launchers pass `-Dfile.encoding=UTF-8`. Without it, Java 17 on Windows misreads the accented author name in `Prism.json`, and the runner then refuses the bot identity.

| Variable | Meaning |
| --- | --- |
| `PRISM_SEED` | Bot RNG seed (melee destination radii). Default fixed. Server start positions are not seeded. |
| `PRISM_TELEMETRY_PATH` | Optional JSONL file. One line per round plus one per game, written at round/game end only. |

Every game starts cold. Learning persists across rounds within a game and resets for a new game. No warm-start mode exists.

## Known limitations

- Duel and melee are both implemented. Melee is validated only by synthetic fixtures, and no melee screen is registered. This counts as a duel-first milestone, not completion of the full specification's validation.
- The recorded engine trace covers acceleration, instant stop, reversal from high and low speed, maximum turns at several speeds, wall contact and a near-corner diagonal contact. A simultaneous two-axis corner clamp is checked only against the engine formula.
- Bot-collision prediction is a constant-velocity hypothesis. Unknown enemy commands are not modelled.
- Bullet shadows are capped at 75%. The engine's bullet bounding-circle pre-check is not modelled.
- The first smoke run (superseded candidate) showed one skipped turn in 566 and a 102.8 ms worst decision, most likely cold-start JIT. The current candidate's smoke showed none in 583, worst 19.3 ms. No warm-up has been added, and turn-budget compliance is not established for real matches.

PR112 readiness maintenance by a separate implementation agent adds surfer-store payload accounting/eviction after existing virtual priority and guarded exact-head label execution; it changes no gun parameters. The refrozen candidate pin is in the v2 plan. Original smoke3 remains evidence for its explicitly named earlier JAR, not this rebuilt readiness candidate.
