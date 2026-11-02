# Minimal targeting decision extraction

Prospective plan, before editing gameplay source. Accepted baseline: `58d519c5e89399381760b1e7ed3fab166f60a85d`, tracked JAR SHA-256 `0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2`.

The purpose is a small testable boundary for later targeting experiments, including supervised motion prediction. This is not a targeting repair or strength experiment. Do not expand the scope into a bot rewrite or combine the rejected scan-lag correction, a narrower firing cone, radar adjustments, wall physics, power changes, or learning changes.

## Frozen boundary

Extract only the existing GuessFactor aiming geometry and fire-request eligibility from `aimGF`: maximum escape angle and lateral sign; five iterations of the current motion projection; final arena clamp; base/aim direction; normalized gun-turn error; command clamped to ±20 degrees; and existing `power > 0 && gunHeat == 0 && abs(error) < 10` eligibility.

Inputs are an immutable snapshot: shooter x/y, target x/y/direction/speed/turn rate/last-scan turn, current turn, initial target distance, bullet speed, selected GuessFactor, lateral velocity, arena width/height, requested power, current gun heading and heat. Outputs contain predicted point, base/aim direction, lateral sign, raw/clamped gun turn, and fire eligibility. The helper reads no API, RNG, model, stat array, or opponent-private data and mutates no state.

Keep target selection, enemy classification, segmentation, histogram argmax/weight, fire-power calculation, and stat-array identity in AISurvivor. Keep command application, `setFire` success handling, pending-wave creation, actual bullet callback anchoring, virtual-wave advancement, and event timing in AISurvivor. No helper chooses a different target, action, power, model, or random number.

Preserve floating-point order and quirks: `Math.hypot`; the API's absolute-bearing normalization; five iterations; `ticks + currentTurn - lastScanTurn` in that order; existing clamp and ±180 loop behavior; zero-lateral fallback to +1; post-hoc wall clamp; unchanged low-energy power cutoff; and the existing ten-degree gate even when it permits a distant miss. Do not replace arithmetic with algebraically similar expressions during extraction. Snapshot reads occur within the same non-yielding turn, with gun/fire commands applied in their existing order.

## Characterization before extraction

First freeze the accepted source provenance and a reference computation copied from its current math, including pinned Bot API distance/bearing semantics. Capture fixed snapshot results before introducing the helper. Cover near/far targets, zero and positive/negative speed/turn rates, scan gaps, all wall/corner clamps, wraparound at 0/360 and ±180, positive/zero power, gun heat, exact ten-degree boundaries, and the intentionally known distant-shot/wall-projection quirks.

Then compare old and extracted outputs bit for bit on those golden cases and a deterministic wide input grid. Compare raw/clamped gun commands, request eligibility, and every numeric field used to create pending waves. Assert state/stat inputs remain unchanged, preserve wave-bin references and setFire success behavior, and keep the onBulletFired implementation unchanged. Existing lifecycle, model, RNG, geometry, and evaluation-freeze checks must continue passing. No new bot RNG calls are allowed.

## Bounded integration check

The primary refactor evidence is deterministic replay equivalence, not random game scores. After independent plan/code review, run **five fixed minimum-risk compatibility rounds**: two against pinned Retcon 0.2.0 and one 1v1 round each against the existing official 1.4.0 `Target`, `SpinBot`, and `Walls` samples. Use fresh isolated processes and absent model paths; require all five telemetry rows, unchanged fixed-mode model paths, no AIS skipped turns, complete native Retcon statistics, and successful completion with the intended participant identities. Scores are reported but have no gain threshold and do not establish strength or statistical equivalence. Preserve invalid attempts before any reviewed correction.

Use Java17 build/tests and tracked-JAR parity before JDK25 matches. Every job is pinned to `ubuntu-26.04`; preserve actual runtime/OS, source/JAR/opponent/release hashes, raw results and logs. The standard CI/sample/Retcon/engine-trace checks remain required. A rebuilt JAR will differ because of the extraction; source parity and replay equality are the relevant checks.

## Publication and next milestone

Publish the plan and accepted characterization before the extraction, then obtain independent preclear. The implementation PR must publish its five-round outcome and decision in `docs/improvements.md`, including the PR, report, and raw-evidence links. Independent review/merge follows green checks. Do not run tactical score A/B games as part of this refactor.

The next AI milestone is legal scan-history data feasibility: current round aggregates lack positions/scan age/action context, so they cannot train or fairly evaluate a motion predictor. A separate prospective plan will define whole-game train/development/withheld splits, horizon-aware future observed-position labels, linear/circular baselines, frozen-model hashes and fallbacks before collection or fitting. Offline accuracy alone will not justify a live gun promotion. The targeting extraction is limited to enabling this next experiment.
