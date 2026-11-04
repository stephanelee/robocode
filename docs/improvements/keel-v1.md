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

**Confirmation gate.** The workflow's confirmation job reads `benchmarks/keel-v1/raw/decision-screen.json`. That file is committed by hand from the screen's evidence branch (`evidence/keel-v1-run-keel-v1-screen-<run>`). Before the confirmation label is applied, an independent reviewer must check it byte for byte against that branch and against the `DECISION-GZ-B64` line in the job log.

**Known frozen-bot limitation (from pre-execution review).** A ram (bot-to-bot collision) changes energy before scans in the same engine turn. Keel applies its −0.6 correction to the following scan, so a collision can create a small phantom 0.6-power wave. This is retained unchanged as part of the frozen candidate. Its effect should be considered when interpreting results; collision counts are reported.

Limitations:

- One CI runtime, and Diamond's own turn timing is not observable to us.
- Ten-round games share learning within a game.
- Variance between games is large: exploratory development games ranged ±10–15 pp.
- A screen pass is progress, not superiority. Only the confirmation can support a "beats Diamond" statement, and then only for this pinned port, runtime and arena.

---

## Registered screen result (appended after the run; the rules above are unchanged)

**Decision: `progress; confirmation open`. All 8 games and 80 rounds were valid, with zero Keel skipped turns.**

- Sole run: [41792449000](https://github.com/stephanelee/robocode/actions/runs/41792449000), at reviewed head `843768f` and committed profile `0fb2d496…`.
- [Pre-execution review](https://github.com/stephanelee/robocode/pull/98#issuecomment-6625216993).
- Raw evidence (official recordings, markers, attempts/commands, runner logs, AIS model/telemetry, decision) is in [benchmarks/keel-v1/raw/](../../benchmarks/keel-v1/raw/). A second copy is on branch `evidence/keel-v1-run-keel-v1-screen-41792449000` (commit `fac3b6f`).
- The committed `decision-screen.json` is byte-identical to the evidence branch. All recording SHA-256 values match.
- An independent reducer (`benchmarks/keel-v1/development/scripts/analyze.py`, which shares no code with the harness) reproduces every share, win count, shot count and hit count below.

| Game | Bot / opponent | Share | Round wins | Hits / shots (rate) | Opponent hits / shots (rate) | Damage dealt / taken | Walls / rams | Mean round length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1 | Keel / Diamond Port | 55.33% | 8/10 | 231/2517 (9.2%) | 296/2784 (10.6%) | 232 / 458 | 11 / 0 | 3,552 turns |
| s2 | Meridian v3 control / Diamond | 14.18% | 0/10 | 35/489 (7.2%) | 89/629 (14.1%) | 197 / 505 | 3 / 0 | 908 |
| s3 | Keel / Diamond Port | 73.05% | 10/10 | 252/2782 (9.1%) | 294/3053 (9.6%) | 243 / 328 | 7 / 1 | 3,889 |
| s4 | Keel / Retcon 0.2.0 | 96.06% | 10/10 | 92/646 (14.2%) | 9/643 (1.4%) | 322 / 40 | 4 / 0 | 1,003 |
| s5 | Keel / Diamond Port | 67.10% | 9/10 | 258/2858 (9.0%) | 288/3151 (9.1%) | 186 / 402 | 5 / 0 | 4,008 |
| s6 | Meridian v3 control / Diamond | 21.60% | 1/10 | 39/483 (8.1%) | 89/610 (14.6%) | 230 / 471 | 0 / 0 | 882 |
| s7 | Keel / Diamond Port | 24.02% | 4/10 | 201/2176 (9.2%) | 284/2415 (11.8%) | 109 / 686 | 6 / 0 | 3,121 |
| s8 | Keel / AISurvivor 1.2.1 cold TRAIN | 88.28% | 10/10 | 180/560 (32.1%) | 31/483 (6.4%) | 548 / 164 | 0 / 0 | 826 |

Damage is uncapped observer bullet damage. Round lengths come from the decision file's `turnsByRound`.

**Diamond summary:**

- Keel: equal-game mean 54.88%, 31/40 round wins, games 24.0–73.0%, sample SD 21.8 pp.
- Contemporaneous Meridian v3 control: mean 17.89%, 1/20 wins.

**Gate checks:**

| Check | Result |
| --- | --- |
| Keel Diamond mean > 25% | pass |
| Keel Diamond mean > control mean | pass |
| ≥ 2 Keel Diamond wins | pass (31) |
| Retcon > 60% | pass (96.06%) |
| AISurvivor > 50% | pass (88.28%) |
| Diamond mean > 45%, so confirmation opens | pass |

**Interpretation and limits:**

- This is a **screen pass (progress)**, not a claim that Keel beats Diamond Port. Only the registered confirmation (6 fresh 20-round games) can support that.
- Between-game variation is large. Game s7 lost 6 of 10 rounds: it took 686 damage, against 328–458 in the other Keel games, and its first two rounds scored 3.5% and 7.0% share.
- The mechanism matches development. Rounds were about 3.4–4.5× longer than the control's. Keel fired mostly 0.1-power shots (≈9% hit rate) and kept energy while Diamond spent its own. Diamond's hit rate was 9.1–11.8% vs Keel and 14.1–14.6% vs the control.
- Keel still lost the bullet-damage exchange in every Diamond game. Its share comes mainly from survival scoring.
- Diamond's own skipped turns are not observable. CI ran each game sequentially on a 4-vCPU runner.

## Confirmation attempt 1: zero games (stale pull-request head)

The confirmation gate review [cleared head `eaacb2b`](https://github.com/stephanelee/robocode/pull/98#issuecomment-6629120243). When `run-keel-v1-confirm` was applied, however, GitHub still reported the PR head as `843768f`. The branch ref was `eaacb2b`, but `refs/pull/98/head` had not been updated after the 58 MB evidence push.

Run [41819088130](https://github.com/stephanelee/robocode/actions/runs/41819088130) therefore checked out `843768f`, where `benchmarks/keel-v1/raw/decision-screen.json` does not exist. The gate step raised `FileNotFoundError` before `keel_experiment.py run confirmation` started. **No confirmation game or round was played**, so nothing is consumed or replaced. The run's evidence push also failed with a GitHub `Internal Server Error`. Two artifacts were uploaded (`keel-profile-41819088130` and `keel-games-run-keel-v1-confirm-41819088130`); both contain only the zero-game profile files.

This is an infrastructure failure, not a result. Confirmation will run once, at a head that GitHub reports as the PR head, after a fresh independent stage check of the subsequent changes.

**Runtime drift found by the gate review.** Run 41819088130 used Python 3.13.18 and printed `Profile: 3ba1fcaa…`. The registered profile is `0fb2d496…`, recorded under Python 3.13.17. The harness refuses to play unless the two profiles are byte-equal, so a confirmation run would have stopped with zero games.

To keep the registered runtime rather than re-profile it, the workflow now pins the registered toolchain exactly: Python 3.13.17, Temurin 17.0.21+9 and Temurin 25.0.5+8 (previously `3.13`, `17` and `25`). A zero-game `profile-keel-v1` run must reproduce `0fb2d496…` byte for byte before confirmation. If it does not, for example because the runner image changed, confirmation stays blocked and any re-profile is disclosed and reviewed as a runtime change.

---

## Registered confirmation result (appended after the run; rules unchanged)

**Decision: `confirmed`. Under the registered rule, Keel 0.1.0 beats pinned Diamond Port 1.8.28 in this runtime.**

- Sole games-playing confirmation run: [41831109687](https://github.com/stephanelee/robocode/actions/runs/41831109687), at gate-cleared head `2127675` ([gate review](https://github.com/stephanelee/robocode/pull/98#issuecomment-6630805846)).
- Byte-identical registered profile `0fb2d496…`.
- All 6 games and 120 rounds valid, with zero Keel skipped turns.
- Raw evidence (official recordings, markers, attempts, runner logs, decision) is in [benchmarks/keel-v1/raw-confirmation/](../../benchmarks/keel-v1/raw-confirmation/). A second copy is on branch `evidence/keel-v1-run-keel-v1-confirm-41831109687` (`ee5899f`); the decision bytes match.
- The independent dev reducer reproduces every share, win count and hit count below. All recording SHA-256 values match.

| Game | Share | Round wins | Keel hits / shots | Diamond hits / shots | Damage dealt / taken | Walls / rams | Mean round length (turns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1 | 57.98% | 18/20 | 498/5480 (9.1%) | 613/6013 (10.2%) | 367 / 942 | 2 / 0 | 3,851 |
| c2 | 59.00% | 18/20 | 560/6225 (9.0%) | 735/6797 (10.8%) | 344 / 879 | 4 / 0 | 4,354 |
| c3 | 55.23% | 17/20 | 494/5601 (8.8%) | 628/6191 (10.1%) | 374 / 952 | 19 / 0 | 3,948 |
| c4 | 54.59% | 16/20 | 472/5172 (9.1%) | 599/5725 (10.5%) | 356 / 965 | 8 / 0 | 3,684 |
| c5 | 58.91% | 18/20 | 540/6042 (8.9%) | 682/6590 (10.3%) | 366 / 903 | 9 / 0 | 4,212 |
| c6 | 53.46% | 16/20 | 506/5621 (9.0%) | 664/6130 (10.8%) | 349 / 895 | 6 / 0 | 3,924 |

**Summary:**

- Equal-game mean share **56.53%**, game range 53.46–59.00%, sample SD 2.40 pp.
- **103/120** official round wins.
- **6/6** games above 50%.
- Descriptive one-sided Student-t 95% lower bound on the game-mean share: 54.56%. This was not a registered gate.

**Registered checks:**

| Check | Result |
| --- | --- |
| All games valid | pass |
| Zero Keel skips | pass |
| Mean share > 50% | pass |
| More than 60/120 round wins | pass (103) |
| At least 4/6 games > 50% | pass (6) |

**What this does and does not show:**

- **Scope.** This is a registered win against one pinned opponent build, Diamond Port 1.8.28. It ran on the Python 3.13.17 / Temurin 25.0.5 / ubuntu-26.04 runtime, in the default 800×600 arena with 100 ms turn timeouts, with cold games of 20 rounds. It is not a claim about the original Java Diamond, other arenas, other runtimes or other opponents.
- **Mechanism: energy attrition and survival scoring.** Keel lost the bullet-damage exchange in every game (≈350 dealt vs ≈900 taken). It wins by firing mostly 0.1-power shots while Diamond spends its energy in rounds of about 3,700–4,350 turns, then collecting survival points.
- **Unobserved factors.** Diamond's own skipped turns are not observable. The reviewer-noted ram correction limitation of the frozen bot remains.
- **Development history.** The candidate was selected by exploratory local games, and the screen and confirmation are the unbiased tests. Confirmation attempt 1 played zero games. Only run 41831109687 played confirmation games.
- **Guards.** These were measured only in the screen (Retcon 96.06%, AISurvivor 88.28%, one game each). Transfer to other strong opponents is untested.


## Measured implementation clarification and completed acquisition

In the frozen gun, the `real` flag means an SDK-accepted `setFire` request. It is not a `BulletFired` confirmation. The anti-surfer and virtual-gun selector learn from accepted-request wave-passage proxies, not guaranteed actual bullet creations or actual-hit labels. The historical source wording is preserved; no request/confirmation equivalence or component-level causal result is established. Source, JAR, model, registration and numerical rules are unchanged.

The [independent confirmation audit](https://github.com/stephanelee/robocode/pull/98#issuecomment-6633000801) is complete. The supported outcome is scoped to this pinned Diamond Port/runtime/800×600/100 ms setting and energy attrition/survival scoring; it does not establish broad strength. Retcon and AISurvivor guards were each measured in the screen only.

Both registered acquisition stages are consumed. The active launcher is retired; its exact [screen and confirmation snapshots and provenance](../../benchmarks/keel-v1/source/retired-workflows.json) remain archived. Ordinary Java 17 build/tests/JAR-parity CI remains enabled. The original [zero-game failure payload and compressed job log](../../benchmarks/keel-v1/zero-game-attempt-41819088130/) are retained separately; no game or outcome was replaced.
