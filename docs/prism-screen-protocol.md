# Prism prospective screen protocol

**Status: registered, not run.** No Prism gameplay battle against any opponent has happened. The first premeasurement review (PR #105, head `1fe315a`) found four gameplay defects and missing replay protection. The candidate was fixed and refrozen; that review does not clear it. Running the screen needs a **new** independent premeasurement review of the exact refrozen head, recorded in [`docs/prism/screen-plan.json`](prism/screen-plan.json) as `approvedReview` (review URL) and `approvedHead` (the exact 40-character reviewed commit), and in the tracking issue. The implementer must not approve their own protocol. This protocol is frozen as written. No value may change after any outcome is observed.

## Hypothesis

Prism profile `prism-initial-profile-1` (implementation milestone 0.1.0) passes the [specification's](diamond-port-clean-room-specification.md) section 13.3 engineering screen. All 13 scheduled games must be valid. There must be no numerical failures or unhandled exceptions, skipped turns below 0.5% of Prism turns in every opponent cohort, and combined wall and bot collisions below 1% of Prism turns. The mean equal-game score share must exceed 40% against Diamond Port and against Retcon, and must be at least 50% against Adaptive Prime, SpinBot and Walls.

The expected outcome is not stated as a claim. The repository's history (Meridian, Breakwater and AISurvivor all at or below 32% against Diamond Port) suggests the Diamond gate is unlikely to pass. Passing would permit only the statement "the candidate passed the screen".

## Pins

| Item | Pin |
| --- | --- |
| Candidate | `experiments/Prism/Prism.jar` SHA-256 frozen in `screen-plan.json` (`prismJarSHA256`). Source is frozen by the git tree ID of `experiments/Prism/src` at the freeze commit. Source/JAR parity is checked by CI |
| Profile | `prism-initial-profile-1` (`Profile.java`). Cold start every game. No model files read or written |
| Engine | Official Tank Royale 1.4.0 runner, SHA-256 `02f6d1d8…5f327cf`, embedded server on an ephemeral port, 1v1 preset (800×600), turn timeout 100,000 µs. Port 7654 is never used |
| Diamond Port, Adaptive Prime | `tools/opponents/strong-melee.lock.json` at git blob `8e2dac721b240d57330fd1afd00cf7c919bc256c`, staged unchanged by `strong_melee_opponents.prepare/stage/verify_staged` under Python 3.13 |
| Retcon | `tools/opponents/retcon.lock.json` at git blob `885137b3638f3d19d537bb9189333e747886e21a`, via `retcon.prepare`, source tree hash verified, JDK 25.0.5+8 |
| Official samples | SpinBot and Walls from `sample-bots-Java-1.4.0.zip`, SHA-256 `1707c101…a0765fe2` |
| Replay protection | A durable receipt: tag `refs/tags/prism-screen-v1-<first 16 hex of the frozen JAR SHA-256>`, created atomically through the GitHub API at the exact execution head before the first game, in a separate workflow step. Creation fails if the tag exists, so any later dispatch or re-run attempt stops before a match. The receipt is never deleted, including after a failed or invalid screen. The game step requires a receipt file from the same run ID, and the remote tag must point at the execution head |
| Exact-head clearance | The execution head must descend from `approvedHead`, and `git diff approvedHead..head` may touch only `docs/prism/screen-plan.json` (the approval record). Any other change refuses the run |
| Runtime | GitHub Actions `ubuntu-26.04`. Temurin 17 for the Gradle build and JAR parity, then Temurin 25.0.5+8 for every match after Gradle has finished. Harness: `tools/prism_screen.py`, `tools/PrismBattle.java` |

## Budget and structure (fixed)

200 rounds in 13 independent cold games:

| Opponent | Games × rounds | Seats | Prism seeds |
| --- | --- | --- | --- |
| Diamond Port | 3 × 20 | Prism first, opponent first, Prism first | 4101–4103 |
| Retcon | 3 × 20 | same alternation | 4104–4106 |
| Adaptive Prime | 3 × 20 | same alternation | 4107–4109 |
| SpinBot | 2 × 5 | Prism first, opponent first | 4110–4111 |
| Walls | 2 × 5 | Prism first, opponent first | 4112–4113 |

All rounds within a game share online learning. Games are the independent units, and rounds are never treated as independent observations. Seat order is the order of the runner's bot list. Whether it influences start positions is not established, and server start positions and scheduling are **not** seeded. That limitation is accepted before collection. Each game runs from a fresh staging copy and a fresh Prism process.

## Metrics

Per game, the harness records: official totals, first places and bullet damage for both bots; equal-game score share `prism/(prism+opponent)`; Prism shots requested, confirmed and hit; wall and bot collisions; skipped turns; subsystem failures; decision latency quantiles (turn/move/gun); and all counters, including inferred certain/ambiguous shots, forecasts, gun selections and fallbacks. Raw runner logs, results, attempt metadata and Prism telemetry are preserved per game. Round-index learning curves are derived from the per-round telemetry lines.

## Decision rules

1. Any invalid or incomplete game, identity mismatch, or missing telemetry: **reject**. It stays in the evidence and is not rerun.
2. Otherwise apply every gate in the hypothesis. Failing any gate: **reject** this profile for promotion. No favourable subset, rerun or retune replaces a failure.
3. Passing every gate: record "passed screen" only. A claim to beat Diamond Port or Retcon needs a *separately registered* fresh confirmation (initially 10 independent cold 20-round games per claimed opponent). That confirmation must show a majority of official round first places and a 95% game-cluster bootstrap lower bound above 50% (10,000 resamples, fixed analysis seed). The harness never starts a confirmation.

After the screen, `docs/improvements.md` gets the budget, the outcome, the decision, and links to the PR and raw evidence.
