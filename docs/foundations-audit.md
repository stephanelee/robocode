# Foundations audit (2026-11-03)

This audit covers master `e4e4c94` after PRs #38–#46. It separates facts, each with its source, from recommendations. No gameplay change is proposed here without its own registered experiment.

## Where the product stands (facts)

- **Retcon, default product.** The default product is `train` mode with the duel controller (`DuelSurf` movement, `DuelGun` aiming and `DuelRadar` lock in 1v1). It is confirmed against pinned Retcon 0.2.0: 200 fresh rounds, **86.74%** equal-game share, **187/200** first places, Student-t lower bound 82.27% ([#46](improvements/final-default-confirmation-redo.md)).
- **Wider roster (development probe, not evidence).** This is a local probe I ran, using new development seeds, a fresh model and 5 rounds per opponent. The default bot won every 1v1 against the whole official sample roster: Corners, Crazy, Fire, RamFire, TrackFire, VelociBot, SpinBot, Walls, Target and MyFirstBot. Its lowest score share was 76.2%, against TrackFire. In a 10-round, 8-bot melee against seven sample bots it ranked 1st, with 7 first places and a 23.5% share (a uniform split would be 12.5%).
- **Reliability is the open defect.**
  - The first default confirmation ([#44](improvements/final-default-confirmation.md)) was invalidated by **one** skipped turn, and its cause was never established.
  - In the development probe above, the melee game also recorded one skipped turn.
  - A profiled rerun located it at **round 1, turn 1**. Decision-loop compute peaked at 40 ms on turn 2 and 21 ms on turn 76 of round 1, and almost all later turns took a few milliseconds.
  - The benchmark turn timeout is 100 ms (`tools/BenchmarkBattle.java`). Before this PR, per-turn latency was not recorded at all.
  - **Update after the registered check** ([warm-up reliability](improvements/warmup-reliability.md)): the cold-start-of-our-code hypothesis is **not supported**. In CI every melee game skipped exactly one round-1 turn in both arms, while AISurvivor's own decision latency never exceeded 12.3 ms. In an exploratory local test, a do-nothing probe bot also skipped round 1, turn 1. The working hypothesis is a multi-bot game-start effect; it is not yet established. The #44 1v1 skip at round 4 is a separate, still unexplained event, now attributable through the new latency telemetry.

## Architecture findings (facts)

1. **Two parallel stacks in one class.** `AISurvivor.java` (about 930 lines) contains two complete controllers:
   - the legacy stack: DQN strategy selection over six force-vector movements, minimum-risk, one-way wall smoothing, the legacy GuessFactor gun, and the legacy radar;
   - the duel composition.

   `duelComponents()` switches between them; it is true only for `TRAIN` mode, `DuelPolicy.CONTROLLER` and exactly one enemy. Each handler branches on that flag.
2. **Mode semantics are inconsistent.**
   - In 1v1, `FIXED` and `EVAL` modes run `DuelSurf` movement for action 0, but they keep the **legacy gun and radar**.
   - The `fixed-minrisk` benchmark policy therefore no longer measures the default product, and nothing in the configuration says so.
   - The `train` mode name now also means "the confirmed default product". That conflates learning with deployment.
3. **Hidden coupling.** In the default 1v1 path, the legacy GF profile and geometry are still computed every turn, solely to supply `DuelGun`'s "legacy aim" candidate. Legacy GF statistics update only in `TRAIN`, and their class statistics persist in the model file.
4. **The DQN is in the loop mostly as bookkeeping.**
   - In default 1v1 its choice is forced to action 0, and it stores forced transitions.
   - It still steers in melee, where it can switch between six movement strategies every turn.
   - Its benefit in melee has never been measured.
   - The model file is saved at every round end.
5. **Melee movement keeps the known wall defect.** Legacy melee movement uses the one-way `wallSmooth` search whose unchecked-heading failure was diagnosed in [wall-search](improvements/wall-search.md). In the development melee probe above, the bot hit walls 35 times in 10 rounds. Duel movement has its own engine-conformant wall handling.
6. **Tests.** Unit coverage of the pure components is good: `DuelSurf`, `DuelGun`, `AimCorridor`, the composition, learning and persistence. There is no test of the full turn loop, and no harness that replays recorded observations through the controller to compare commands across refactors.

## Done in this PR (no gameplay change)

- **Turn-latency telemetry.** `RoundMetrics` now records `maxTurnMicros` and `slowTurns` (turns of 20 ms or more), measured from the turn's `TickEvent` to the `go()` that answers it. This makes skip-risk visible in every future run.
- **Pre-game warm-up.** `Warmup.run` exercises the per-turn decision code 3,000 times in the bot constructor, before it connects. On the local machine this took about 285 ms.
  - It touches only pure functions and throwaway instances with their own fixed seed.
  - A test shows an agent created after warm-up makes identical decisions to one created without it.
  - Registered result: **retain-neutral**. It neither reduced nor increased skips (15/15 skipping melee games in both arms; 0/15 for Retcon), so no reliability effect is claimed.

## Recommended order for the larger rewrite

Each of the following needs its own PR and its own evidence.

1. **A command-replay harness first.** Record legal observations together with the bot's own events and its command intents. Then replay them through the pure controllers and require identical commands. Without this harness, a rewrite of a confirmed 86.7% bot cannot show parity, and a green build is not a performance claim.
2. **Restructure behind the harness.** Split the code into:
   - a thin `Bot` adapter;
   - an observation model (enemies, own history, inferred waves);
   - independent `Movement`, `Gun` and `Radar` interfaces with `Duel*` and `Melee*` implementations;
   - telemetry.

   Remove the mode-flag branching from the handlers.
3. **Separate deployment from learning.** Make the default duel composition explicit in every mode, and make learning an orthogonal option. Re-baseline the benchmark policies so that `fixed`/`eval` measure the shipped product.
4. **Melee.** Register a coherent melee controller as an experiment: minimum-risk with path commitment and wall handling that follows the engine, plus per-target virtual guns. The baseline is the current melee rank against the sample roster.
5. **DQN.** Measure whether melee strategy selection by the DQN beats a fixed coherent controller. Retire the DQN if it does not.
