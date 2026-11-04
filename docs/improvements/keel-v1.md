# Keel v1: engine-exact surfing with energy-conserving fire vs Diamond Port

Status: **registered before any registered game**. This document freezes the hypothesis, candidate, control, opponents, runtime, budget and decision rules. Results are appended below the registration only after the registered run, without editing the rules.

## Why a new bot

About 25 earlier Diamond Port challengers (AISurvivor variants, DiamondChallenger, Aegis, Breakwater and Meridian v1–v3) reached at most a 20.77% equal-game share (Meridian v3 control, two five-round games, 1/10 round wins; [PR #92](https://github.com/stephanelee/robocode/pull/92)). None combined full multi-option precise-intersection wave surfing with KNN guns. All of them also kept firing about 1.2–2.0 power at hit rates near 3–9%.

Two pinned-source facts motivate Keel:

1. **Instant stop.** The pinned Tank Royale 1.4.0 server's `calcNewBotSpeed` returns 0 at once for a zero target speed from any forward speed. Reversal from +8 gives +3. Movement uses the new speed on the old heading, then the turn is limited by the new speed. Diamond's own movement predictor keeps the classic deceleration-2 rule ([opponent analysis](../opponents/diamond-port.md)), so it underestimates how quickly we can stop. Keel's surfer plans with the exact engine rule. This includes candidate paths that orbit for *k* turns and then stop dead.
2. **Attrition economics.** Every confirmed shot costs its power. A hit refunds 3×power and deals 4p+2(p−1). At an ~8% hit rate, one unit of fired energy costs us ~0.9 and Diamond ~0.4. Diamond's own ~10–12% hit rate at 1.95 power costs it more energy per shot (~1.3) than it removes from us (~1.0). In exploratory development games, rounds were mostly lost by our own energy drain. Diamond then fired 0.1-power finishing shots, because its power is capped by our remaining energy.

## Hypothesis

Isolated experimental bot **Keel 0.1.0** wins more score share and rounds against pinned Diamond Port than the strongest earlier challenger, Meridian v3. Keel combines:

- engine-exact precise-intersection wave surfing (orbit/stop-at-*k* options, two-wave lookahead, KNN hit danger plus a context flattener);
- KNN guess-factor main and anti-surfer guns with virtual-gun selection;
- an energy-conserving fire policy: 0.1-power token shots by default; 1.2 when our energy leads Diamond's by more than 15; 2.5 under 200px; kill-shot and low-energy caps.

Keel must not regress against Retcon or production AISurvivor. This is a whole-bot design hypothesis. Component credit is not identifiable from these games.

## Candidate, control and opponents

| Role | Identity |
| --- | --- |
| Candidate | `experiments/Keel` 0.1.0. Source and committed `Keel.jar` are frozen at the reviewed PR head. Java 17 build, contract tests and `verify_jar.py` parity run before every registered stage. |
| Contemporaneous control | Meridian 0.3.0 (v3), exact JAR bytes from commit `809b880` (sha256 `5875565a…a1a8a9`, identical to the registered meridian-aim profile), staged unchanged from `benchmarks/keel-v1/control/Meridian/`. |
| Hard target | Diamond Port 1.8.28, upstream `fduch-stranger/robocode-bot@67b68c22`, pinned source/wheel lock `tools/opponents/strong-melee.lock.json`, Python 3.13 fresh venv. |
| Guards | Retcon 0.2.0 (`tools/opponents/retcon.lock.json`). Production AISurvivor from this commit's `AISurvivor/` in cold default TRAIN mode. |
| Runtime | GitHub Actions `ubuntu-26.04`, Temurin 17 build/tests, Temurin 25 battles, pinned Tank Royale 1.4.0 runner (sha256 `02f6d1d8…5f327cf`), 1v1, 100 ms turn timeout, default 800×600 arena. A zero-game profile job records every hash. The run step requires byte equality with the committed `benchmarks/keel-v1/profile.json`. |

Keel reads only its own scans, state, commands and events. It never reads opponent files, processes, RNG or telemetry. Learning resets between games (fresh process). Within a game, Keel keeps its gun and danger data across rounds, like Diamond.

## Development disclosure

Keel was developed with **exploratory local games** against the same pinned opponents on a non-registered container runtime (OpenJDK 25.0.4 Ubuntu package, 4 vCPU). They tuned the fire policy, flattener weight and bullet power. Those games selected the candidate and are biased upward (winner's curse). They are not evidence and are not pooled. Their summary is in [benchmarks/keel-v1/development/README.md](../../benchmarks/keel-v1/development/README.md).

## Registered games ([games.json](../../benchmarks/keel-v1/games.json))

**Screen** (fixed order, one attempt each, 80 rounds):

| ID | Bot | Opponent | Rounds |
| --- | --- | --- | --- |
| s1, s3, s5, s7 | Keel | Diamond Port | 10 each |
| s2, s6 | Meridian v3 control | Diamond Port | 10 each |
| s4 | Keel | Retcon | 10 |
| s8 | Keel | AISurvivor (cold TRAIN) | 10 |

**Confirmation** (only if the screen opens it; fresh games, 120 rounds): c1–c6, Keel vs Diamond Port, 20 rounds each.

Each game uses a fresh process, a fresh staging directory, its own registered loopback port and seed, and a cold opponent. Seeds only set Keel's/Meridian's initial orbit side and AISurvivor's RNG. Engine start positions and opponent randomness are uncontrolled.

## Validity

A game is valid only if all of the following hold:

- the official recording has the registered setup and exact participants/versions;
- every round endpoint and contiguous tick is present;
- both private round markers exist for every round;
- incoming shots are nonzero;
- for AISurvivor, the cold TRAIN telemetry is complete.

An invalid game is recorded with its error. It is never replaced or rerun, and the stage decision becomes `invalid`, so no promotion is possible.

## Decision rules (fixed before any registered game)

Share = official total score / both totals, averaged equally over games.

**Screen → "progress"** requires all of:

1. all eight games valid and zero Keel skipped turns;
2. Keel's mean Diamond share > 25%;
3. Keel's mean Diamond share > the contemporaneous Meridian control mean;
4. at least 2 Keel round wins vs Diamond out of 40;
5. Keel Retcon share > 60%;
6. Keel AISurvivor share > 50%.

Otherwise the decision is **reject**. Progress opens confirmation only if Keel's mean Diamond share is also > 45%.

**Confirmation → "Keel beats Diamond Port (registered)"** requires all of:

- all six games valid;
- zero Keel skips;
- mean share > 50%;
- more than 60 of 120 round wins;
- at least 4 of 6 games above 50%.

Otherwise the result is **not confirmed**. No threshold, source, seed or opponent changes after any registered game. Failed candidates are preserved, never rerun.

## Reported measurements

Per game: share, round wins, per-round shares and winners, our and Diamond's shots, hits, hit rates, shot energy, damage dealt and taken, bullet–bullet collisions, wall and bot contacts, private skipped turns, turns per round, recording hash, command, staged hashes and runtime profile.

Limitations:

- One CI runtime, and Diamond's own turn timing is not observable to us.
- Ten-round games share learning within a game.
- Variance between games is large: exploratory development games ranged ±10–15 pp.
- A screen pass is progress, not superiority. Only the confirmation can support a "beats Diamond" statement, and then only for this pinned port, runtime and arena.
