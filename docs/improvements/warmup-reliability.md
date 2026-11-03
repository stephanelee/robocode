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

The text above this heading is the registered plan, unchanged since `f3db2d4ee9c170d719909e8cb7dd83513130266a`, which is also the measured candidate. The driver stored that commit's Git blob byte for byte.

**The run.** There was a single registered run, [41638002007](https://github.com/stephanelee/robocode/actions/runs/41638002007): job 125923991152, artifact 13233215708, zip SHA256 `96ce41f7b6774358cce00bf93fb57a0ef027f8441dacc8a0da9621ed56bdc06a`.
- Every step was green and nothing was rerun.
- It ran on actual Ubuntu 26.04.1 LTS. Java 17 JAR parity passed, and the matches ran on Temurin 25.0.5+8.
- The candidate JAR was `21c5c3d2cb4007e3e0603556af313df7f5717732e384f039b005e9e72b2daecd`. The baseline JAR hash was verified before launch.
- All 60 games (**120 rounds**: 60 battles of 2 rounds each) completed with validated telemetry.

**Decision: retain-neutral.** The candidate never skipped more than the baseline, but it did not skip less either. No reliability effect is claimed. The warm-up and the latency telemetry stay, as registered.

| Scenario (15 games × 2 rounds per arm) | Baseline: games with skips / skipped turns | Candidate: games with skips / skipped turns | Candidate worst turn latency | Candidate turns ≥ 20 ms |
| --- | ---: | ---: | ---: | ---: |
| Retcon 1v1 | 0 / 0 | 0 / 0 | 9.2 ms | 0 |
| Seven-bot sample melee | **15 / 15** | **15 / 15** | 12.3 ms | 0 |
| All | 15 / 15 | 15 / 15 | 12.3 ms | 0 |

**What the run shows (facts).**
- Every melee game, in both arms, skipped **exactly one turn, always in round 1**.
- No Retcon game skipped.
- The candidate's decision latency, from the tick to `go()`, never exceeded 12.3 ms against the 100 ms timeout.
- Mean score shares were descriptive only: Retcon 91.9% (baseline) and 96.2% (candidate); melee 30.3% and 29.6%.

**Root-cause diagnostic (exploratory, after the run, not registered).**
- *Setup.* I wrote a probe bot whose run loop does nothing but call `go()` and logs its own `SkippedTurnEvent`s. I added it as an eighth opponent in three local 2-round melees (Ubuntu 24.04, development seeds 7000–7002). Its source, descriptor, log and the three games' official results and AISurvivor telemetry are in [`benchmarks/warmup-reliability/exploratory-probe/`](../../benchmarks/warmup-reliability/exploratory-probe/).
- *Result.* The probe logged one skip at **round 1, turn 1**. Its log does not identify which of the three games that was. AISurvivor skipped once, in round 1 of the first game.
- *Interpretation (hypothesis).* The round-1 melee skip may occur at the start of a multi-bot game regardless of a bot's own work. Contention while the runner starts and source-compiles eight bot JVMs is one plausible cause. This is a single small local observation with round-level records only, not per-turn attribution. It is **not** sufficient grounds to exclude skips from future confirmation protocols. The CI fact that AISurvivor's own decision latency never exceeded 12.3 ms does show its decision code was not slow, which explains why warming that code up could not help.
- *Not explained.* The single skip that invalidated [#44](final-default-confirmation.md) occurred in a **1v1 Retcon** game at round 4, so it is a different event and remains unexplained. Future runs now record `maxTurnMicros` and `slowTurns`, so such an event can be attributed or ruled out.

**Suggestion for future confirmation protocols (needs its own evidence).** If the start-up hypothesis is confirmed with per-turn attribution, a protocol could pair each skip with the bot's own decision latency, or begin scoring after a warm-up round. Until then, every skip counts.

**Local observations, disclosed before registration (not evidence):**
- *Cloud container.* The baseline skipped in 3/8 melee games and the candidate in 0/8. That apparent effect did not reproduce under registered conditions.
- *`minimind`.* Apple **M4**, macOS 27.2, Homebrew OpenJDK 17.0.21 to build and 27.0.1 for the battles. The commit was `6c84f0e`, with a JAR (`21c5c3d2…`) byte-identical to the committed one. The run stopped with an infrastructure error at melee game 9. Over 15 Retcon and 8 melee games per arm, neither arm skipped. The candidate's worst decision latency was 2.5 ms (Retcon) and 3.5 ms (melee), with 0 slow turns. Hardware details were supplied by the user after registration.

**Limitations.**
- The arms are unpaired.
- Runner load is not controlled.
- The probe diagnostic is small and was run outside CI.

Evidence: [raw evidence](../../benchmarks/warmup-reliability/README.md).

## Review correction (2026-11-03)

The independent review of PR #47 found an arithmetic error. This correction is appended; the registered text above is left as written.

**The error.** The plan's procedure line reads "60 games, 120 game starts across both arms, and 240 rounds", and its limitations mention "30 game starts per arm per scenario". Both double the actual design. The design was 15 games per arm per scenario, and each arm played 15 Retcon and 15 melee games. That is 60 battles of 2 rounds each:
- **60** game starts in total, **15 per arm per scenario**;
- **120** rounds.

**What it changes.** The executed procedure (`--games 15 --rounds 2`) is what the plan's command line specified, and the decision rule never depended on the doubled numbers. So the registered decision, retain-neutral, is unchanged.

The results line above is corrected. The index row in `docs/improvements.md` and the PR description are corrected to match.

