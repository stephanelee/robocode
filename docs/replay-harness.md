# Command-replay harness

This is step 1 of the restructure recommended in the [foundations audit](foundations-audit.md). The goal is that any refactor or rewrite of the bot can be **proven** to make the same decisions, instead of being argued or measured with noisy battles.

## Structure

| Class | Role |
| --- | --- |
| `SurvivorBrain` | All decision logic. The bodies moved verbatim from the former `AISurvivor extends Bot`. Private one-line helpers keep the original Bot API names (`getX()`, `setFire()`, `directionTo()` …) and delegate to the port. |
| `BotPort` | The only game surface the brain uses: the current turn's own state and arena, plus the intent setters. |
| `AISurvivor` | Thin Tank Royale adapter. It forwards events and turns to the brain and implements `BotPort` by delegating to the Bot API. |
| `GameTrace` | Optional trace written by the adapter when `AISURVIVOR_TRACE_PATH` is set (`.gz` compresses it). It records, in arrival order: a header (mode, seed, duel policy), a state snapshot before every turn and at every tick, every event with all fields the brain reads, and every intent call with its argument (and `setFire`'s result) per turn. Tracing only observes. |
| `TraceReplay` | Feeds a trace into a fresh brain through a `ReplayPort` and compares every intent call bit-exactly (`Double.doubleToLongBits`, so `-0.0 ≠ 0.0`). A trace only succeeds if it holds at least one complete game: every started round ended, then a `GameEndedEvent`. It reports the first mismatch or completeness failure, giving line, round, turn and both call lists. |

The replay port reproduces the Bot API rules the brain depends on:
- `setFire` is rejected for NaN, for energy below the power, or for a hot gun.
- `directionTo` and `distanceTo` use the `IBaseBot` formulas.
- `calcBulletSpeed` clamps power to 0.1–3.

One engine detail was found while building this: **the first decision turn of a round runs before the first tick is dispatched.** At that point the live API already reports the arena and enemy count. The trace therefore snapshots state immediately before each turn.

## Golden traces

`benchmarks/replay-goldens/` holds three traces of real local games played by the current default bot. They used seed 4242, `train` mode, a fresh model, and 3 rounds each, on Ubuntu 24.04 with OpenJDK 17 and 25.

| Trace | Scenario | Command batches |
| --- | --- | ---: |
| `retcon-1v1-3r.jsonl.gz` | Pinned Retcon 1v1: duel movement, gun, radar and a forced DQN | 3,044 |
| `melee8-3r.jsonl.gz` | Default bot against seven official samples: DQN Boltzmann selection, legacy movement and gun; includes one skipped turn | 3,007 |
| `walls-1v1-3r.jsonl.gz` | Official Walls 1v1 | 939 |

`TraceReplayTest` replays all three on every build and requires identity.

It also checks that the harness actually detects changes:
- A single altered recorded command is reported.
- A different seed changes the melee decisions.
- The adapter's run-start batch contains only the two adjustments the brain makes.

To check any trace by hand: `java -cp AISurvivor/AISurvivor.jar TraceReplay path/to/trace.jsonl.gz`.

## What this does and does not establish

- **Established:**
  - The brain is deterministic given its inputs (the seeded RNG included).
  - The extraction kept every behaviour that existing tests pin: all 288 exact-bit `aimGF` command/state goldens from the targeting refactor, and all existing composition, learning and geometry tests pass unchanged except for the mechanical port to `SurvivorBrain` + `BotPort`.
  - The three live games replay with every command identical.
- **Established afterwards (`MasterParityTest`):** an unmodified copy of master `e4e4c94`'s `AISurvivor`, driven through all three complete goldens with Mockito-stubbed final Bot API getters, issues bit-identical commands (bitwise double comparison) on all 6,990 batches. The pre-extraction code and the extracted brain therefore decide identically in every recorded state. Runtime and timing behaviour, and unvisited states, are outside this proof.
- **Not established:**
  - A trace pins decisions only for the states it visited.
  - No strength claim is made.
- **Rule for future restructuring:** each step must keep `TraceReplayTest` green, or explain and separately register every intended decision change.
