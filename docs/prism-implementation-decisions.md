# Prism implementation decisions

This log covers Prism 0.1.0 (`experiments/Prism/`), profile `prism-initial-profile-1`. It records the implementer's provenance, the engine contracts used, and every assumption or deviation from the [clean-room specification](diamond-port-clean-room-specification.md). The specification is version 1, 2026-11-04, and its Facet naming is replaced by Prism and `PRISM_*`. The bot [README](../experiments/Prism/README.md) describes behaviour. The [registered screen](prism-screen-protocol.md) has not been run.

## 1 Implementer provenance and access boundary

- **Inputs used.** The specification and the [implementer prompt](diamond-port-implementer-prompt.md) were supplied by the user. The specification exists only on branch `facet/independent-v1-20261104-01a1a639` (PR #103), so I extracted just that one file with `git show <branch>:docs/diamond-port-clean-room-specification.md`. Other inputs: this repository's `AGENTS.md`, `CLAUDE.md`, build and runner infrastructure (`build.gradle`, `tools/verify_jar.py`, `tools/benchmark.py`, `tools/BenchmarkBattle.java`, the packaging part of `tools/strong_melee_opponents.py`, and the `play`/`main` harness functions of `tools/meridian_duels.py`), the official Tank Royale **v1.4.0** server and Java Bot API sources from `robocode-dev/tank-royale`, a `javap` disassembly of the shipped runner's `CollisionDetector`, and my own recorded engine traces.
- **Not opened.** I did not open Diamond's Python or Java implementation, any archived or embedded copy, the evolved startup dataset, private models or telemetry, source-derived expected decisions, `docs/opponents/diamond-port.md`, the analysis provenance JSON, `tools/opponents/*.lock.json`, `benchmarks/`, `experiments/DiamondChallenger/`, or any Facet implementation file from PR #103. Opponent pins in the protocol cite lock files by git blob ID only.
- **Mechanical barrier.** The checkout is a partial clone (`--filter=blob:none`) with a sparse checkout that excludes `benchmarks/`, `tools/opponents/`, `docs/opponents/`, `docs/diamond-port-analysis-provenance.json`, `experiments/DiamondChallenger/` and `docs/diamond-challenger.md`. No restricted blob was materialised in the working tree or read. **Barrier incident:** late in the session, while I was checking line endings, `git ls-files --eol` made the partial clone lazily fetch index blobs. Before I stopped it, 352 of the 3,656 blobs under the restricted paths had been downloaded into the local object store (counted afterwards with `GIT_NO_LAZY_FETCH=1 git cat-file --batch-check`). The command's output was never displayed, all gameplay code was already complete, and none of those objects were opened. The affected local clone should be deleted after this PR is pushed. File *names* in Git trees were listed. Grep hits for "adaptive" in `tools/*.py` revealed only harness constants (staging keys).
- **Disclosure.** This implementation session had no prior exposure to the restricted code. I am a language model, though, and my pretraining may include publicly available Robocode code, possibly including the original open-source Java Diamond. I did not consult or reconstruct any such material. All gameplay code here was written from the specification, the official engine/API contracts and my own fixtures. Nobody has independently verified that statement. The reviewer should weigh it accordingly.
- **Other bots.** No code from AISurvivor, Meridian, Breakwater, Aegis, Retcon, Facet or DiamondChallenger was read or reused. Prism's build file follows the repository's per-experiment Gradle convention.

## 2 Engine contracts (Tank Royale 1.4.0)

Each contract comes from the official v1.4.0 source and is confirmed by the recorded trace in `experiments/Prism/fixtures/engine-trace-1/` unless marked otherwise.

| Contract | Source | Prism use |
| --- | --- | --- |
| Turn order: fire phase → body intents (speed, translate on old heading, rotate) → wall → ram → wall constraint → scans → bullets advance → bullet hits | `TurnProcessor.kt` | Predictor; energy visibility; wave shell |
| `targetSpeed == 0` stops instantly from any speed. A sign crossing gives `(1 - v/2)·(-1)`, so 8→3→0.5→-0.75, while a same-sign target changes speed by at most +1/−2 | `math.kt calcNewBotSpeed`; trace | `Physics.newSpeed`; `Drive` picks target 0 when reversing from \|v\| ≥ 2 |
| The turn limit uses the new speed, `10 - 0.75\|v'\|` | `limitTurnRate`; trace | `Physics.step` |
| Wall constraint clamps along the travelled line (x then y) and runs **twice** (collision handling and `constrainBotPositions`). A diagonal push into a wall pins the bot. Speed becomes 0. Damage `max(0,\|v'\|/2-1)` applies only if no wall event occurred the previous turn, so events alternate while pushing | `CollisionDetector.kt`, `TurnProcessor.kt`; trace (the double pass was found by the trace) | `Physics.constrain` ×2 |
| Fire happens first in a turn, from the start-of-turn position along the start-of-turn gun direction, only if heat is exactly 0, power ≥ 0.1 and energy > power. Energy is reduced by the *requested* firepower | `GunEngine.kt`; trace | Plans aim one tick ahead; `Firing.validate`; reserve and clamp |
| A bullet fired on turn T is tested on turn T+k as segment `[(k+1)b, (k+2)b]` from its origin against the end-of-movement bot disk of radius 18 | `Bullet.kt`, `BulletLine.kt`, `CollisionDetector.kt` (source only) | `Wave.innerRadius/outerRadius`, `Geometry.footprint` |
| Scan energy at k excludes bullet damage and bonus on k (visible at k+1). It includes firing cost, wall and ram on k | `TurnProcessor.kt`; trace and the repository's earlier energy trace | `FireInference` effect visibility |
| The Bot API keeps `firepower` in the intent until a `BulletFired` arrives, and `setFire` refuses when heat > 0 | `BaseBotInternals.java` | Explicit `setFire(0)` whenever cool |
| Events for a tick are dispatched before `go()` returns. `TickEvent.getEvents()` carries the turn's batch | `BaseBotInternals.java` | One transaction per tick in `run()` |

## 3 Assumptions and deviations

| # | Topic | Specification | Prism choice and reason | Verification |
| --- | --- | --- | --- | --- |
| 1 | Footprint | "Swept circular footprint" | The engine tests a *static* end-of-move disk against a *swept bullet segment*. Prism computes the exact angular extent of disk ∩ annulus per tick (tangent, boundary-circle and source-inside cases) and takes the union over ticks. No numerical subdivision is needed, so the error bound is 0 up to floating point | Brute-force test (400 cases), tangency, wrap |
| 2 | Label of an outgoing wave | Crossing centre | The first tick at which the outer shell radius reaches the target centre. The bearing to the observed or interpolated centre becomes the label. The half-width `asin(18/d)` is used for scoring | `BehaviourTest` labels |
| 3 | Interpolated labels | Unspecified weight | Quality 0.75 (observed 1.0). A wave whose crossing falls in a gap > 8 turns is cancelled | — |
| 4 | Fire proximity | "nearestKnownFireOpportunityDistance" | Turns until our gun is cool at origin, `turnsToCool(heat)`. It is 0 on physical waves | — |
| 5 | Rated cohort | "confirmed firing opportunity" | Duel ticks with observed own heat exactly 0. Every eligible gun's prediction is frozen on that tick's virtual wave | Selector test |
| 6 | Shot hypotheses | ≤ 3 hypotheses per update | One wave with a fire-turn interval (a radial envelope). Confidence is 1.0 if certain, 0.7 for ≤ 3 feasible turns, 0.5 for an envelope. Envelope training quality inherits that confidence | `missingScansAndWallsRetainUncertainty` |
| 7 | Residual explainable without a shot | Unknown collisions create uncertainty | If the cost interval reaches below 0.1 (for example inferred wall damage `[wd(\|v\|-2), wd(\|v\|+1)]`), no wave is created. The case is counted unresolved and the heat interval widened. This can miss a real shot coinciding with a wall hit | Real trace: repeated wall damage produced 0 false shots |
| 8 | Several hidden shots | Enumerate or mark unresolved | A loss > 3 over a gap yields one max-power envelope (confidence 0.5) and widened heat | Unit test |
| 9 | Inactivity damage | — | Not modelled. Its 0.1 losses fall below legal power and are ignored | — |
| 10 | Escape reach | Precise, wall-limited | Perpendicular full-speed orbit with 30-unit inward smoothing until the shell passes, capped at 200 steps. Walls availability uses an unsmoothed orbit up to `ceil(1.5H)` | — |
| 11 | Danger kernel | "bandwidth from bot size" | `asin(18/d)/M` in GF units for the prior and learned views. Hit-label quality is multiplied by `max(0.25, confidence)` | Danger test |
| 12 | Visit labels | Reliable real-wave crossings | Emitted when the outer radius reaches our centre. Quality is 0.5 for uncertain waves. Visits never enter hit views | `storesDeclareLabelTypesThatNeverMix` |
| 13 | Orbit centre | Not fixed | The enemy's latest scan, advanced at constant velocity by scan age. The bot-collision hypothesis adds 0.6 per overlapping tick at a separation under 38 | — |
| 14 | Second-wave branch | At first-wave centre crossing | Continuations begin with `wallContact=false`, and the range penalty is evaluated at each branch end | `secondWaveContinuationChangesTheDecision` |
| 15 | Shadows | 75% / 100% caps | Always 75%: no event-supported "certain" shadow is implemented. The engine's bounding-circle pre-check is ignored (line intersection only). Turns are counted only while `r1 < d0 - 18 - 8(t-now)` | Shadow test |
| 16 | Melee radius | 100–180 capped by nearest enemy | Additionally capped at `max(40, 0.8·nearest)`. The inward safe point lies 120 units toward the centre. Shielding credit is `1 - age/9`, zero beyond 8 turns | Melee test |
| 17 | Melee projection | Fixed point, ≤ 12 iterations | Continuous travel time `max(0, dist/b - 1.5)`. On a two-cycle the later time is taken | — |
| 18 | Expensive aim | Within three cooling turns | Direct tracking otherwise. A plan is created only when heat will be 0 on the next tick | Fire chronology tests |
| 19 | Plan invalidation | Expiry, death, major update, displacement | Expires after 2 turns, on target death, on scan age > 2 turns, or when own displacement from the predicted origin exceeds 1 unit | `powerPolicyKillCapReserveAndValidation` |
| 20 | Selector initialisation | EMA 0.05 | The first score initialises the mean | Selector test |
| 21 | Global memory guard | 128 MiB, evict low-priority virtual samples | Checked every 100 turns. Evicts the oldest gun-store samples (all virtual-wave derived). Danger hit views are never evicted by the guard. Per-enemy caps keep a saturated enemy at about 4 MB | Runtime test |
| 22 | Failure handling | 3 consecutive → disable | Applies to `gun` (analytic only), `move` (no threats, analytic orbit), `scan`, `waves` and `radar` (fixed sweep on failure) | — |
| 23 | Launchers | — | `-Dfile.encoding=UTF-8`. Found while recording: Java 17 on Windows misreads the author name and the runner then rejects the bot identity | Trace runs |
| 24 | Warm start | Separately registered mode | Not implemented. Every game is cold | Lifecycle test |

## 4 Verification status (specification 13.1)

| Family | Status |
| --- | --- |
| Native angles | Pass (`CoreTest.cardinalVectorsWrapAndOrbitSign`) |
| Engine motion | Pass against the real trace: acceleration, instant stop, both reversal regimes, maximum turn at several speeds, wall and near-corner contact, energy. The two-axis corner case is checked against the engine formula only |
| Fire timing | Pass against the real trace (heading and position from the request tick, confirmation at +1) and in `Brain` (pending, cancel on absence, one promotion despite a duplicate callback) |
| Energy visibility | Pass against the real trace (isolated hits at k+1, one true shot inferred, no false shots from wall damage) and in unit tests (simultaneous hit and fire, gaps, walls, heat) |
| Wave geometry | Pass (brute force, tangency, wrap, approach/pass) |
| GF conventions | Pass (asymmetric scales, sign, theoretical vs precise, label types) |
| Neighbour maths | Pass (brute force with eviction, duplicates, weights, empty store, non-finite input) |
| Learning chronology | Pass (frozen predictions, single training, cancelled waves excluded, visits ≠ hits) |
| Lifecycle | Pass (round reset keeps learning and clears transient state; a new game is cold; three-round real smoke) |
| Surfing | Pass (instant-stop physics, a second wave changes the first action, first-step wall safety, conservative missing-geometry cost) |
| Shadows | Pass (radial zero-width, oblique, complete/partial, disjoint, death invalidation, wrap). Pending-fire cancellation is covered because shadows use confirmed bullets only |
| Melee and radar | Pass on synthetic fixtures (unknown and stale, last-survivor transition, all-unsafe emergency). No real melee trace yet |
| Runtime | Saturated 8,192-sample store over 800 turns on the development machine: p50 4.2 ms, p99 17.1 ms, worst 92.6 ms (includes JIT). The real smoke had 1 skipped turn in 566, worst 102.8 ms, p99 14.0 ms. This is not yet a measured turn-budget guarantee |

Build: 39/39 JUnit tests and 16/16 harness tests pass under Microsoft OpenJDK 17.0.18. Two clean builds produced byte-identical `Prism.jar` (`8c045f80d37a3076ced7b337b020d14bf3942c6c2a89152f11021f8ff9e9e7ca`). The first frozen candidate, `684a80d5…`, is superseded (section 5).

## 6 Profile 2: anti-surfer gun (Prism 0.2.0)

This change is registered prospectively in the [v2 protocol](prism-surfer-gun-protocol.md), and the motivation there comes from Prism-side v1 telemetry only. I read only Prism's `results.json`, `prism-telemetry.jsonl` and `decision.json` from `benchmarks/prism-screen-v1`, each fetched as a single blob with `git show`. I did not open opponent runner logs, `source/tools/opponents`, or the original artifact ZIP.

| Topic | Choice | Reason | Verification |
| --- | --- | --- | --- |
| New gun `surfer` | Three recent windows (32/128/512), at most 5 neighbours each, recent-gun features, precise-GF labels | Same kernel and decoding as the recent gun, so the only difference is which waves it learns from | `surferGunTrainsOnlyOnConfirmedBulletWaves` |
| Training data | Only waves promoted to `OUTGOING` by an engine `BulletFired` that is bound to its request wave; once per wave | A wave surfer reacts only to real bullets' waves | Same test (virtual crossings rejected, single training) |
| Activation | Eligible after 8 confirmed-wave labels; rated on the existing fire-opportunity cohort; selected only by the unchanged selector rule | Guns that don't help are never selected, which limits risk against non-surfers | `surferGunBecomesEligibleAfterEightConfirmedLabelsAndCanBeSelected` |
| Fallback chain | Unchanged (recent → long → analytic → direct) | If the surfer gun has no evidence, the old chain applies | — |
| Harness | `--plan` selects the registered plan. A plan-declared `screenId` names its receipt; v1's default name is unchanged | The spent v1 plan is not edited | `PlanSelectionTest` |

## 5 Review fixes (PR #105, review of head `1fe315a`)

Each defect reproduced against the reviewed JAR. Each fix has a regression test. I mutation-checked every regression: with only the behavioural line reverted, the test fails and reproduces the reviewer's numbers (frozen power 0.9 instead of 1.0; shadow interval `[-3.1404826, 3.1182328]`; 0 revisit samples instead of 13).

| Finding | Fix | Regression |
| --- | --- | --- |
| A confirmation promoted an independently powered virtual wave (1.0 shot vs 0.9 wave, speed 17.3 vs 17.0) | On the request tick the wave for the pending target is always created with the request's own power. The pending record stores that wave's ID. `onFired` promotes only that wave, and only if fire turn, origin and power all match. Otherwise no physical wave is invented (`fire.confirmedWithoutWave`); the bullet still casts shadows | `confirmationBindsToTheRequestPowerWhenTargetEnergyChangesOnTheRequestTick` |
| A same-batch scan resurrected a bot killed in that batch | Round-local `Enemy.deathTurn` tombstone. Scans at or before it are ignored (`scan.ignoredAtOrBeforeDeath`); round reset clears it. The real smoke recorded 3 such scans in 3 rounds, so the defect occurred in real play | `sameBatchScanNeverResurrectsAConfirmedDeath` |
| A shadow segment crossing ±π became the long arc and shadowed bearing 0 | `segmentShell` unwraps the second endpoint relative to the first (a segment not through the origin subtends < π). `Geometry.normalize` splits at the branch cut. The planner compares footprints with shadows in the same normalized frame | `shadowAcrossTheBranchCutStaysAShortArc`, `branchCutSegmentsSplitIntoShortPieces` |
| The melee revisit sampler never sampled (`turn - Integer.MIN_VALUE` overflow) | Explicit uninitialised case and long arithmetic | `meleeRevisitSamplesStartOnTheFirstTurnAndAfterReset` (also after reset) |
| Follow-up review (`bec86f0`): the path allowlist trusted the whole plan file, so a descendant could change seeds, IDs or pins alongside the approval | `verify_clearance` reads the plan at `approvedHead` with `git show` and requires equality after removing only `approvedReview` and `approvedHead` (`APPROVAL_FIELDS`). An unreadable or non-JSON reviewed plan is refused | `test_reviewers_repro_changed_seed_and_id_is_rejected`, `test_changed_seat_opponent_rounds_freeze_or_receipt_is_rejected` (9 edits), `test_approval_only_edit_of_the_plan_is_accepted`, `test_unreadable_reviewed_plan_is_refused`. Mutation-checked: removing the comparison fails both rejection tests |
| No durable replay protection for the screen | Atomic receipt tag at the execution head, consumed in its own step before any game; exact-head clearance; run-ID-bound receipt file; workflow concurrency group. See the [protocol](prism-screen-protocol.md) | `ReceiptTest` (5 tests: second dispatch refused, spent receipt kept, other run/head rejected, extra diff rejected, review mismatch rejected) |
