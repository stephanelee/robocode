# Minimal targeting decision extraction

Prospective plan, before editing gameplay source. Accepted baseline: `58d519c5e89399381760b1e7ed3fab166f60a85d`, tracked JAR SHA-256 `0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2`.

The purpose is a small testable boundary for later targeting experiments, including supervised motion prediction. This is not a targeting repair or strength experiment. Do not expand the scope into a bot rewrite or combine the rejected scan-lag correction, a narrower firing cone, radar adjustments, wall physics, power changes, or learning changes.

## Frozen boundary

Extract only the existing GuessFactor aiming geometry from `aimGF`: maximum escape angle and lateral sign; five iterations of the current motion projection; final arena clamp; base/aim direction; normalized gun-turn error; and command clamped to ±20 degrees. Independent pre-edit review specifically leaves `power > 0 && gunHeat == 0 && abs(error) < 10` and `setFire` in AISurvivor, exactly where they are.

Inputs are an immutable snapshot: shooter x/y, target x/y/direction/speed/turn rate/last-scan turn, current turn, initial target distance, bullet speed, selected GuessFactor, lateral velocity, arena width/height, and current gun heading. Outputs contain predicted point, base/aim direction, lateral sign, and raw/clamped gun turn. The helper reads no API, RNG, model, stat array, or opponent-private data and mutates no state.

Keep target selection, enemy classification, segmentation, histogram argmax/weight, fire-power calculation, and stat-array identity in AISurvivor. Keep command application, `setFire` success handling, pending-wave creation, actual bullet callback anchoring, virtual-wave advancement, and event timing in AISurvivor. No helper chooses a different target, action, power, model, or random number.

Preserve floating-point order and quirks: `Math.hypot`; the API's absolute-bearing normalization; five iterations; `ticks + currentTurn - lastScanTurn` in that order; existing clamp and ±180 loop behavior; zero-lateral fallback to +1; post-hoc wall clamp; unchanged low-energy power cutoff; and the existing ten-degree gate even when it permits a distant miss. Do not replace arithmetic with algebraically similar expressions during extraction. Snapshot reads occur within the same non-yielding turn, with gun/fire commands applied in their existing order.

## Characterization before extraction

First freeze accepted-source provenance and characterize the actual existing private `aimGF` method with test-only API mocks, including real pinned Bot API default distance/bearing methods. Capture fixed snapshot results before introducing the helper. Cover near/far targets, zero and positive/negative speed/turn rates, scan gaps, all wall/corner clamps, wraparound at 0/360 and ±180, positive/zero power, gun heat, exact ten-degree boundaries, setFire rejection, missing per-enemy stats, and the intentionally known distant-shot/wall-projection quirks. Test-only final-method mocking bypasses network startup; it is excluded from the bot JAR.

Then compare old and extracted outputs bit for bit on those golden cases and a deterministic wide input grid. Compare raw/clamped gun commands, request eligibility, and every numeric field used to create pending waves. Assert state/stat inputs remain unchanged, preserve wave-bin references and setFire success behavior, and keep the onBulletFired implementation unchanged. Existing lifecycle, model, RNG, geometry, and evaluation-freeze checks must continue passing. No new bot RNG calls are allowed.

## Bounded integration check

The primary refactor evidence is deterministic replay equivalence, not random game scores. After independent plan/code review, run **five fixed minimum-risk compatibility rounds**: two against pinned Retcon 0.2.0 and one 1v1 round each against the existing official 1.4.0 `Target`, `SpinBot`, and `Walls` samples. Use fresh isolated processes and absent model paths; require all five telemetry rows, unchanged fixed-mode model paths, no AIS skipped turns, complete native Retcon statistics, and successful completion with the intended participant identities. Scores are reported but have no gain threshold and do not establish strength or statistical equivalence. Preserve invalid attempts before any reviewed correction.

Use Java17 build/tests and tracked-JAR parity before JDK25 matches. Every job is pinned to `ubuntu-26.04`; preserve actual runtime/OS, source/JAR/opponent/release hashes, raw results and logs. The standard CI/sample/Retcon/engine-trace checks remain required. A rebuilt JAR will differ because of the extraction; source parity and replay equality are the relevant checks.

## Publication and next milestone

Publish the plan and accepted characterization before the extraction, then obtain independent preclear. The implementation PR must publish its five-round outcome and decision in `docs/improvements.md`, including the PR, report, and raw-evidence links. Independent review/merge follows green checks. Do not run tactical score A/B games as part of this refactor.

The next AI milestone is legal scan-history data feasibility: current round aggregates lack positions/scan age/action context, so they cannot train or fairly evaluate a motion predictor. A separate prospective plan will define whole-game train/development/withheld splits, horizon-aware future observed-position labels, linear/circular baselines, frozen-model hashes and fallbacks before collection or fitting. Offline accuracy alone will not justify a live gun promotion. The targeting extraction is limited to enabling this next experiment.

## Characterization milestone

Commit `74bcb6f8e79a97513c93a77425d8e631ba34cc64` froze the accepted source snapshot and **288 actual private-aimGF command/state outcomes before any production edit**: 32 named boundary/lifecycle cases and 256 wide snapshots from fixture seed 13371337. IEEE-754 output bits, gun/fire commands, pending-wave fields and array identity are recorded; stat arrays and bot RNG remain unchanged. Test-only Mockito is excluded from the runnable JAR. The extracted geometry and actual bot adapter passed those unchanged recordings, plus bitwise comparison of every pure result field against the accepted arithmetic using real API distance/bearing defaults. At that milestone, Java17 clean build and 62 Python tests passed locally; the registered five-round sanity run was still pending independent exact-workflow preclear.

## Measured result and decision

**Retain the minimal refactor, subject to final independent review and required checks.** The deterministic before/after replay passed exactly, and all **five registered compatibility rounds** completed with valid fixed-policy telemetry, zero AIS skipped turns, no model writes, and complete native Retcon statistics. No score-gain or statistical-equivalence claim is made; the accepted ten-degree firing gate, final wall clamp, scan-lag behavior, selection, learning and event handling remain unchanged.

| Opponent | Fixed rounds | AIS / opponent score | AIS wins | AIS skipped turns |
| --- | ---: | ---: | ---: | ---: |
| Retcon 0.2.0 | 2 | 48 / 270 | 0 | 0 |
| Official Target 1.4.0 | 1 | 180 / 0 | 1 | 0 |
| Official SpinBot 1.4.0 | 1 | 169 / 0 | 1 | 0 |
| Official Walls 1.4.0 | 1 | 166 / 0 | 1 | 0 |

The hard opponent remains unbeaten in this tiny sanity check. Sample wins validate launch/integration and are not evidence of broad competitive strength. This is not a paired incumbent/candidate score experiment, and five rounds cannot establish unchanged win probability. Exact accepted command/state replay is the primary refactor evidence.

- [PR #25](https://github.com/stephanelee/robocode/pull/25), [actual run 41515841364 / job 125519071370](https://github.com/stephanelee/robocode/actions/runs/41515841364/job/125519071370), exact candidate `438e16de5cef95e10fb8a8d2d6b8254a0e47a68f`.
- [Durable raw evidence](../benchmarks/targeting-refactor/README.md): original plan bytes/hash, canonical report, all four games/five AIS rows, complete native Retcon export, original accepted source/goldens/provenance. [Complete artifact](https://github.com/stephanelee/robocode/actions/runs/41515841364/artifacts/13173474829) preserves runnable bundles without duplicating their JARs in Git.
- Plan `c7f64d3be25220d4574f1b5b31bfe9d8bc6c6db7` was published and independently precleared before extraction/measurement; actual accepted characterization `74bcb6f8e79a97513c93a77425d8e631ba34cc64` predates production edits. No additional scored sanity run or rescued attempt occurred.
- Candidate JAR SHA-256 `6b5a448fa1f5edba38aae04f364e44d0943f6eeb6b4b938580085f87dd668170`; Java17 build/parity, Temurin25.0.5+8-LTS matches, actual Ubuntu26.04.1, SHA-pinned Runner/server1.4.0, Retcon0.2.0/API1.3.1, official 1.4.0 sample archive. Exact provenance appears in `sanity.json`.

The temporary sanity workflow is removed after capturing the valid outcome. The small geometry boundary, unchanged firing/state adapter, frozen behavior recordings, evidence, and reusable bounded sanity tool remain. The next AI experiment still requires legal-scan acquisition and an independently reviewed dataset/model protocol; this refactor alone adds no learned targeting feature.
