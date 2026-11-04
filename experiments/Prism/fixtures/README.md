# Prism engine evidence

These are actual recordings from the official Tank Royale 1.4.0 Battle Runner. Each was produced in an isolated embedded-server battle on an ephemeral port. The interactive server on 7654 was never contacted. None of these are gameplay measurements.

## `engine-trace-1/`: engine-contract trace

Two scripted probe bots took part (`probe/prism/probe/Probe.java`, built by `gradlew -p experiments/Prism probeJar`). The *mover* executes a fixed command script: acceleration, target-zero stop from 8, reversal from 8 and from 1, maximum turns at several speeds, a straight wall push and a diagonal near-corner push. It also fires one power-1.0 shot. The *shooter* stays still, locks its radar, and fires power-0.3 shots while requesting a simultaneous 15° gun turn. After turn 420 it fires power-3.0 shots. Each tick both bots log their own state, the exact command sent, and every delivered event.

| Item | Value |
| --- | --- |
| Runner | `robocode-tankroyale-runner-1.4.0.jar` SHA-256 `02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf` |
| Probe JAR | `PrismProbe.jar` SHA-256 `4b7eb153a44758eae23cad0f4ebd2dd85bb7eae6fc18d406a98c84974c058aac` |
| Probe source | `Probe.java` git blob `2c433d45aeb2092d33c7fed5ca74a6099bbf65e9` |
| Battle driver | `probe/PrismTraceBattle.java`: 1 round, 1v1, 100 ms turn timeout |
| Runtime | Microsoft OpenJDK 17.0.18+8 (runner and bots, `-Dfile.encoding=UTF-8`), Windows 11 10.0.26300 |
| `mover.jsonl` | 571 ticks, SHA-256 `6c95be001e1b06ea172617deccd22f383dc7ad53816744765efe20d88a9b4e0d` |
| `shooter.jsonl` | 572 ticks, SHA-256 `da8ce6b390da9102b084f61698b4cb44a85d4e1f9b454916a6ddab69bbd57985` |
| `results.json` | SHA-256 `9040d495293c192ad8d9a19576940a983bf073120784e2d848cbcbbe69b32262` |

`EngineTraceTest` replays every consecutive mover transition without a bot collision through `Physics.step`. It requires a match within `1e-6` in position, speed and degrees, and an exact energy reconciliation. It also checks that every confirmed shot used the gun heading and position observed on the request tick (not the newly requested turn), confirmed exactly one turn later. Finally it drives the shooter's recorded view through `Brain`, which must infer exactly the mover's single shot (fire turn and power 1.0) and nothing from its repeated wall damage.

Three earlier recordings in the same session are not preserved here. Each was a superset attempt that failed to reach the corner because of probe-script bugs: a wall-phase counter reset and a diagonal push that pins the bot. The first recording (362/363 ticks, mover SHA-256 `658803fa…`) passed the same tests before it was superseded. It revealed no predictor disagreement. The corner recording then exposed one: the engine applies its wall constraint twice. `Physics` was corrected and the test now covers it.

## `lifecycle-smoke-2/`: API integration smoke of the current candidate (not a measurement)

Prism `8c045f80d37a3076ced7b337b020d14bf3942c6c2a89152f11021f8ff9e9e7ca` (after the PR #105 review fixes) played three rounds against the same passive probe shooter. Results: 34 fire requests, 34 confirmations, each bound to its own frozen request wave (0 `fire.confirmedWithoutWave`), 30 hits, 0 cancelled requests, 0 subsystem failures, 0 skipped turns in 583. Worst decision 19.3 ms (p99 5.2 ms). `scan.ignoredAtOrBeforeDeath` was 3: in each round the kill-turn scan arrived in the same batch as the death, the case the review identified.

## `lifecycle-smoke-1/`: superseded candidate (not a measurement)

Prism `684a80d580619fc36666b41a3ba2827232e36fd1be59b57fa54922669691e0ed` played three rounds against the passive probe shooter. This checks the real API path: round resets, fire request → confirmation, telemetry, and no exceptions. Results: 33 fire requests, 33 engine confirmations, 30 hits, 0 unconfirmed or cancelled requests, 0 subsystem failures. There was **1 skipped turn** in 566, and the worst decision took 102.8 ms (p99 14.0 ms, p50 0.62 ms). A passive diagnostic bot is not an opponent, so this result supports no strength claim.
