# Independent Diamond STOP composition source audit

2026-11-04. Reviewed `/private/tmp/diamond-port-source-contract-audit.md` against the retained exact67b68c22 Diamond adapter/predictor/SurfMover, independently downloaded the pinned Python1.3.1 SDK wheel and inspected ZIP text without importing it, and read pinned engine c8ad3a8d math/TurnProcessor/CollisionDetector source. No test, geometry case, SDK module, bot or game was run. No repository source/freeze changed.

## Disposition

The audit's central STOP mismatch and ordinary reversal qualification are correct. A separate STOP-option-only forecast seam can isolate this composition correction without replacing moving forecasts, their residual-footprint approximation, gun self-projection or MEA. This is a plausible source-contract diagnostic, not yet an implemented candidate or performance hypothesis. I found no already executed Diamond-fork STOP correction in the inspected retained source/docs/82 local origin refs; prior opponent documentation already described the broader discrepancy, so it is not a newly discovered law or novel algorithm.

## Independent source chain

The wheel's SHA256 is074f28f1f62ede458801739d8f84651af8c33bb2aee3290e7841fbd9bc327df2. Independently read wheel-source hashes match the first audit: bot_internals3928eaca…, base_bot_internalsa74971b…, intent_validatordf3a025d….

1. `SurfMover` chooses STOP by calling adapter `set_max_velocity(0)`, then still computes wall-smoothed back-as-front steering and calls set_forward with signed100distance. The max setter writes SDK `bot.max_speed=0`.
2. SDK `BotInternals.set_forward` invokes the finite-distance speed controller and writes through `BaseBotInternals.target_speed`. The latter ALWAYS applies `validate_target_speed(value,max_speed)`. Thus even if intermediate classic braking asks for a nonzero speed, max_speed0 clamps the actual sent target speed to0. This final clamp cannot be omitted from a reference.
3. Pinned `calcNewBotSpeed` immediately returns0 for a moving bot receiving target0, with signed-speed symmetry. `TurnProcessor` sets new speed, translates along old heading, then updates turning. Consequently STOP's translation is zero for this command, absent another physical interaction. The original classic predictor computes residual signed speed and translation from its old-speed deceleration law.
4. Turn state also changes: SDK `set_turn_left` assigns requested turn to base turn_rate, validated against configured maximum; engine limits that command using the UPDATED speed. With STOP speed0, the engine's turn cap is the stationary cap, whereas original predictor caps turning using prior velocity. A STOP-only correction that merely sets velocity/position to0 but retains the old-speed turn cap is NOT full pose/heading composition conformance. This matters to later forecast steps even if the current stationary centre matches.
5. For ordinary reversal at max_speed8 and long finite signed distance, SDK controller can send an intermediate still-positive speed rather than requested travel sign. Therefore raw engine current8/target−8→3 does not describe the adapter's effective set_forward(−100) transition. The source example intermediate+6→engine+6 is valid. Do not generalize this equality to all distances/speeds; the future diagnostic must use the actual finite-distance controller/clamp and requested-distance sign.

The first audit's separate wall conclusion is also source-supported: original predictor wall adjustment changes coordinates but keeps next_velocity; pinned engine applied wall outcomes set speed0. Its one-pass/classic positional adjustment is not the engine's two-pass constraint. This is separate scope and must not be silently repaired alongside STOP. Likewise request-relative wave epochs can already match the creation sweep[v,2v]; a universal added tick is unjustified.

## How to isolate the seam

The active defensive call sites are two `_predict_surf_location` uses in `SurfMover._check_danger`:

- The primary next-state loop uses max_velocity0 for a nominal STOP option and8 for moving options.
- The residual footprint collection loop ALWAYS passes max_velocity0, including for moving nominal options.

Thus a global patch to `MovementPredictor` or a dispatch solely on numeric max_velocity==0 changes moving-option residual footprints and fails the proposed isolation. The caller must carry the nominal option/purpose explicitly. A STOP-option reference transition may replace STOP primary and STOP residual steps (including a STOP second-wave branch) while every call belonging to a moving nominal option remains the literal old predictor, including its max_velocity0 residual collector. Other predictor consumers—gun/radar self-location, precise_escape_angle/MEA, imaginary-wave/source convenience projections and non-SurfMover movement—must remain unchanged.

This can isolate the **direct** changes, not guarantee all downstream decisions stay the same: a changed STOP danger can change which option wins, later actual history, learned samples, caches and future behavior. Moving forecasts evaluated on identical frozen inputs should be bit-identical; later gameplay populations cannot be required identical after a selected action differs.

Preserve existing destination choice, smoothing/back-as-front sign conventions, dangerous-wave geometry, learned views, wave epochs, tie rules and all orbit options. Map engine angle conventions back to RobotState carefully; do not hide a quarter-turn/sign conversion inside expected-case numbers. The required oracle is adapter→exactSDK1.3.1→pinnedengine, not the sibling bot_core physics model and not raw engine target speed alone.

## Implications for the proposed24-case zero-game diagnostic

The stated2placements×2signedspeeds×3commandclasses×2requestedturns grid is bounded and useful as a composition check, provided all positions/headings/distances/turns and oracle inputs are frozen before outputs. Continue/reverse cells are characterization controls; STOP is the isolated seam. A near-right-wall stationary STOP can show heading correctness but cannot itself exercise translation-wall constraint, so do not claim the24rows prove general wall/corner conformance or all finite-distance regimes. The exact proposed oracle/class/source/byte pins and the complete protocol still require independent review before any case executes.

Record separately the SDK raw finite-distance intermediate, final clamped emitted target, requested/emitted body turn, engine resulting pose/speed/heading and unchanged original predictor output. Expected tolerances apply only to comparison, never dispatch. No policy/danger/action-selection/live strength conclusion follows. If a replacement is eventually implemented, source parity must explicitly demonstrate moving residual, moving primary, gun/radar and MEA paths unchanged for identical inputs. The untouched opponent remains its exact original pin, while any deployed fork needs a distinct identity/license/provenance and a later separate live protocol.

No live budget, candidate, implementation or acquisition clearance is granted. The next review is the concrete preregistered zero-game protocol; no tests/cases have been executed by this reviewer.
