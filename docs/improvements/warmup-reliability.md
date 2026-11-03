# Pre-game warm-up and turn-latency telemetry: reliability check

Status when this file is first committed: **preregistered, not yet measured.** Results are appended under the final heading in a later commit. Nothing above that heading is edited once the run has started.

## Baseline, candidate and environment

- **Baseline:** master `e4e4c94` (the confirmed default product), JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`.
- **Candidate:** the commit containing this file. Relative to the baseline it adds only `Warmup` (called in the bot constructor) and per-turn latency fields in `RoundMetrics`. The workflow records the exact SHA and JAR hash.
- **Opponents:**
  - Retcon 0.2.0 at `28981c7a469ac064194b41001cd7183ec5ecf189`, run under JDK 25.
  - The official 1.4.0 sample bots Corners, Crazy, RamFire, TrackFire, VelociBot, SpinBot and Walls, as a seven-opponent melee.
- **Infrastructure:** Tank Royale 1.4.0 Runner with a 100 ms turn timeout, Java 17 build, exactly `ubuntu-26.04`.

## Facts motivating the change

- [#44](final-default-confirmation.md) was invalidated by a single AIS skipped turn whose cause was not established.
- In local development profiling (Ubuntu 24.04, not evidence), the default bot skipped **round 1, turn 1** of a melee game.
- In the same profiling, decision-loop compute peaked at 40 ms on turn 2 and 21 ms on turn 76 of round 1. Most later turns took only a few milliseconds.
- Per-turn latency was not recorded anywhere before this change. See [the foundations audit](../foundations-audit.md).

## Mechanism (hypothesis)

Cold-JVM costs land inside early game turns: class loading, interpretation before JIT, and first-call initialization of the decision code. Running that code beforehand moves those costs before the bot connects, outside the turn timeout.

The warm-up uses only pure functions and throwaway instances with their own fixed seed. A unit test shows it does not change agent decisions.

Bot API and JSON-parsing warm-up is outside its reach, so some cold cost may remain.

## Procedure (single run, fixed in advance)

The experiment is one `Warmup reliability experiment` workflow job running:

```text
tools/reliability_experiment.py --games 15 --rounds 2 --seed-base 20261601
```

- Every arm plays default `train` mode with a fresh model.
- For each scenario (Retcon 1v1 and seven-bot melee), each arm plays 15 games of 2 rounds, alternating order.
- In total that is 60 games, 120 game starts across both arms, and 240 rounds.
- There is no early stopping, no favourable rerun and no game replacement.
- An infrastructure failure marks the run invalid. The partial evidence is preserved, and any retry is disclosed.

## Development observations before registration (disclosed; not evidence)

- **Cloud container:** Ubuntu 24.04, OpenJDK 17/25, 4 vCPU, eight bot JVMs in each melee. I ran 8 short melee games per arm on development seeds 9900–9907. The baseline skipped a turn in **3/8** games, always in round 1. The candidate skipped in **0/8**. The candidate's round-2 latency maxima in that build were inflated by a timer bug, now fixed: the timer was not reset between rounds, and one reading was 4.65 s.
- **User's Mac (`minimind`):** macOS, Homebrew OpenJDK 17.0.21 for the build and OpenJDK 27.0.1 for battles. The run used the commands of this protocol with seed base 20261701, so it was not a registered run. It completed 15 Retcon games and 8 melee games per arm, with **0** skipped turns in either arm. It then stopped at melee game 9, because the embedded server did not become ready within 5 s. That is an infrastructure failure; the partial output is kept on that machine. Fast hardware does not reproduce the skip at all, so this environment cannot measure the effect.

These observations motivated the registration. They did not change its rules.

## Primary measures and decision (unrounded, all games pooled)

Per arm, I record:
- the number of games with at least one skipped turn;
- the total skipped turns.

For the candidate only, I also record the maximum per-turn latency and the number of slow turns (20 ms or more).

| Outcome | Condition |
| --- | --- |
| **Retain with effect** | Candidate skipped turns ≤ baseline **and** candidate games-with-skips **<** baseline. |
| **Retain as neutral infrastructure** | Candidate skipped turns ≤ baseline **and** candidate games-with-skips ≤ baseline, without a strict reduction. In this case no reliability effect is claimed. |
| **Reject** | Otherwise. The warm-up is removed. The telemetry may be kept only if it adds no skipped turns. |

Scores are descriptive only. The change does not alter decisions, so no score gate applies.

## Limitations, stated in advance

- Skipped turns are rare events. With 30 game starts per arm per scenario, a modest true effect may not separate from zero.
- Runner load is not controlled.
- The baseline has no latency telemetry.
- This run cannot rule out GC pauses or scheduler stalls.

## Results

Not yet measured.
