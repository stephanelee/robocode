# Diamond STOP command-composition characterization: prospective protocol

Registered before implementation or evaluations. Fixed diagnostic budget: **24 one-step cases, zero games, one acquisition attempt**. No candidate fix, action selector, fork or live budget is proposed. Publish a new registration and obtain independent exact-payload review before any case is executed.

## Question and boundary

Does the frozen Diamond Port's active one-step predictor agree with the command actually produced by **adapter → pinned Python SDK controller/clamps → Tank Royale 1.4.0 motion rules**, particularly for its existing STOP option?

The classic-order/STOP finding is already documented in `docs/opponents/diamond-port.md`; it is not a novel algorithm. The important qualification is SDK composition: `set_forward(±100)` does not simply send signed max speed. Do not replace reversal with the raw engine's negative-target law. This diagnostic characterizes source contracts only and cannot establish competitive value.

## Exact inputs and original call

Arena 800×600, bot radius18. Each case is independent, representing a normal decision at turn13 with prior history available; return time14. No first-turn placement exception, other bot, collision with another bot, gun, bullet, wave, learned model or damage response is simulated.

Own body heading is Tank Royale **0 degrees east** in every case, equivalent to classic **π/2 radians** (north zero, clockwise positive). Signed speed is in pixels/turn. Classic requested turn is either `0.0` or `math.pi/6`; the latter is clockwise and maps through the adapter to `-math.degrees(math.pi/6)` Tank Royale degrees. Do not round that conversion to a different input. The classic turn is the frozen input; its Tank Royale degree conversion is recorded as an acquisition intermediate under the pinned Python runtime, not precomputed as a separate platform-dependent input.

Two layout templates:

- O: own `(400.0,300.0)` for either signed speed.
- W: own `(780.0,300.0)` for speed+8, and `(20.0,300.0)` for speed−8. This mirrors the outward wall approach, making both starting states compatible with a preceding legal constant-speed step. It replaces the earlier informal “near right wall” sketch before any outputs; no cases were evaluated to choose it.

Commands are fresh, reissued adapter commands: CONTINUE uses `set_max_velocity(8)` and signed distance `100*sign(speed)`; REVERSE uses `set_max_velocity(8)` and the opposite distance; STOP uses `set_max_velocity(0)` and the same signed distance as CONTINUE. The signed distance maps exactly to SDK `set_forward(distance)` (negative means backward). Requested body turn is independent, as in the explicit command pair. Equivalently, back-as-front can issue that pair using classic desired travel angle `heading+turn+(π if distance<0 else 0)`.

Each original forecast is exactly:

`MovementPredictor(BattleField(800.0,600.0)).predict(RobotState(Point(x,y), math.pi/2, speed, 13), distance, turn, max_velocity, 1, False)`

The source is the unchanged staged `diamond_port/movement_predictor.py` and its exact transitive `dia_utils` dependencies. **Do not import `diamond-port.py`**, whose bootstrap/SDK bot entry point is outside this diagnostic. Active SurfMover reaches `next_location_toward`, which passes signed distance±1000, whereas these direct calls match the adapter's actual±100 command. Source-level first-step equivalence holds for this fixed grid because both magnitudes≥20 use the same original max-velocity shortcut. Normalize the equivalent back-as-front desired angles exactly as the pinned adapter, including negative-distance cases. This does not execute the full SurfMover decision pipeline or reconstruct its destination, danger or private state.

| ID | Layout | Speed | Command | Signed distance | Max velocity | Classic turn |
|---|---|---:|---|---:|---:|---|
|01|O|+8|CONTINUE|+100|8|0|
|02|O|+8|CONTINUE|+100|8|π/6|
|03|O|+8|REVERSE|−100|8|0|
|04|O|+8|REVERSE|−100|8|π/6|
|05|O|+8|STOP|+100|0|0|
|06|O|+8|STOP|+100|0|π/6|
|07|O|−8|CONTINUE|−100|8|0|
|08|O|−8|CONTINUE|−100|8|π/6|
|09|O|−8|REVERSE|+100|8|0|
|10|O|−8|REVERSE|+100|8|π/6|
|11|O|−8|STOP|−100|0|0|
|12|O|−8|STOP|−100|0|π/6|
|13|W|+8|CONTINUE|+100|8|0|
|14|W|+8|CONTINUE|+100|8|π/6|
|15|W|+8|REVERSE|−100|8|0|
|16|W|+8|REVERSE|−100|8|π/6|
|17|W|+8|STOP|+100|0|0|
|18|W|+8|STOP|+100|0|π/6|
|19|W|−8|CONTINUE|−100|8|0|
|20|W|−8|CONTINUE|−100|8|π/6|
|21|W|−8|REVERSE|+100|8|0|
|22|W|−8|REVERSE|+100|8|π/6|
|23|W|−8|STOP|−100|0|0|
|24|W|−8|STOP|−100|0|π/6|

## Reference, using actual pure helpers where feasible

Use the actual pinned SDK class methods, not reimplemented approximations:

1. `IntentValidator.get_new_target_speed(speed, distance, max_velocity)`.
2. `IntentValidator.validate_target_speed(result, max_velocity)` — the max-speed clamp is essential for STOP.
3. `IntentValidator.validate_turn_rate(-math.degrees(turn), 10.0)` — the SDK's default maximum body-turn rate.

These correspond to wheel1.3.1 `internal/bot_internals.py:168–173,184–187,353–359`, `internal/base_bot_internals.py:768–780`, and `internal/intent_validator.py:26–29,44–51,66–111`. Repeated pre-send movement preparation sees the same current speed/fresh distance for this one-step input. Record the intermediate controller result and final emitted SDK speed/turn request separately.

The pinned engine JAR exposes server-free public static methods:

- `dev.robocode.tankroyale.server.rules.MathKt.calcNewBotSpeed(double,double)`
- `dev.robocode.tankroyale.server.rules.MathKt.limitTurnRate(double,double)`

Use those actual methods in a minimal Java17 bridge if the independently reviewed build can bind them exactly. ZIP/class-signature inspection only has established their availability; **no physics method has been invoked**. The classes are major55, and the nested JAR includes Kotlin runtime classes.

Invoke the helpers with exact argument order: `newSpeedBeforeWall = MathKt.calcNewBotSpeed(oldSignedSpeed, sdkFinalTargetSpeed)` and `limitedTurn = MathKt.limitTurnRate(sdkFinalRequestedTurnRate, newSpeedBeforeWall)`. Translate along the OLD Tank heading and apply `limitedTurn` before wall stopping. Record SDK requested turn and actual limited turn separately. The first wall pass is explicitly the normal-turn path where `lastRound?.lastTurn` is non-null, not the initial-turn suppression path. An applied constraint means exact `rawX != firstX || rawY != firstY`; set final speed0 only on that first-pass constraint. The second positional pass uses original oldX/oldY and first-passX/Y as its candidate, with no second speed/heading update. Within each literal constraint call, preserve the original candidate x/y in its branch expressions rather than feeding an X-branch intermediate into the Y branch. Normalize the final heading to [0,360); preserve unconstrained, first-pass and second-pass positions. This composition does not prove general corner handling.

The translation/wall composition is an **independently reviewed source-derived reference**, not a complete running server. Port the literal pinned `TurnProcessor.kt:145–150,422–423` and `CollisionDetector.kt:240–266,374–389,406–459` operations, including both wall passes. Direct whole-turn engine invocation is deliberately avoided: its GameSetup/bot/round/event context would enlarge this diagnostic. If actual pure MathKt binding is unavailable, stop before cases; a fully source-derived speed/turn alternative would need a prospective amendment and new review, not a silent fallback.

The SDK import/loader closure must also be reviewed as server-free before execution: no Bot construction, entry-point bootstrap, `start`, `run`, `go`, server, WebSocket, socket or opponent process. Actual controller method bodies, MathUtil and constants must remain unmodified. Use the exact owned private Python environment with `-I -B` and only the reviewed explicit module closure; sanitize Python import/injection variables and retain every imported module's resolved path/hash. The Java bridge's isolated, explicit classpath contains only its frozen artifact and the pinned nested server JAR (plus prospectively declared pinned dependencies, if any). Invoke bridge main only. Resolve the exact public static signatures; record the actually loaded MathKt `CodeSource`, class bytes and containing JAR hash and reject any unexpected origin or duplicate. Clear Java classpath/injection options including `CLASSPATH`, `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS` and `JDK_JAVA_OPTIONS`; use absolute pinned executables. No ambient class or package may satisfy the contract. Import feasibility has been inspected as source/metadata, not exercised here.

## Outputs, tolerances and interpretation

Retain all24 rows in ID order, with exact input bits/hex floats, original output, SDK intermediate/final intents, reference speed/turn/position before and after both wall passes, final output and errors. Report absolute x/y/signed-speed error and wrapped heading error in **degrees**, plus wall involvement and all exceptions. Convert the original classic heading back using `(90-degrees(h)) mod360` before comparison.

An agreement means each x/y/speed error≤`1e-9` and absolute wrapped heading error≤`1e-9` degrees. Time must equal14 exactly. Keep signed-zero representations in raw output; ±0 compare as equal. No tolerance tuning, favorable subsets or replacement cases. Summarize all cases and the eight STOP cases separately; these are predetermined descriptive groups, not performance populations.

Validity requires exact pins/runtime,24 unique inputs/outputs, finite required values, no skipped/omitted case and no forbidden execution path. A forecast discrepancy is a measured characterization, not an invalid run. If STOP does not exhibit the source-predicted discrepancy, report that result rather than repairing/re-running. No required mismatch count or agreement result opens a live budget.

## Freeze, review and one attempt

Freeze before exact-payload review: port67b68c22/source manifest a9bf…; SDK wheel SHA `074f28f1f62ede458801739d8f84651af8c33bb2aee3290e7841fbd9bc327df2`; all imported SDK/port source bytes and dependencies; engine c8ad3a8d source/blob hashes; Runner1.4 SHA `02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf`; its unmodified nested server JAR SHA `16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c`; bridge source/classes/JAR/classpath; the24-card manifest; reference/loader/reducer/wrapper code; exact commands, environment, paths and300-second process cap.

Hosted-only `ubuntu-26.04`, x86_64, exact Temurin17.0.21+9 and CPython3.13.18. Only the existing image revisions `20261101.186.1` and `20261026.180.1` are permitted, as defined by `benchmarks/meridian-aim/profile.json` and `profile-20261026.json`; freeze both defining files. Use their pinned wheel graph and installed-package manifest `52b7bfcf10f8226711798f27784be64583ea4c8fc676452d27f6e16b906bf191`. Zero-case preparation captures the actual selected image/OS/architecture, Java release/executable/javac/modules/JVM hashes, Python binary/version, package/source/file identities and boot/workspace provenance. Independently freeze that actual profile and bridge/artifact bytes before clearance. No third image, Java/Python fallback or retrospective profile acceptance. No Java25/Retcon launcher is needed.

Branch `codex/diamond-stop-contract24`; exact-head independent clearance token `DIAMOND-STOP24-CLEARED <head>`. Both hosted jobs have60-minute caps. Serialize acquisition concurrency with cancellation disabled and require `run_attempt == 1`. The zero-case profile job downloads/hashes/compiles/inspects metadata only and never creates the acquisition receipt. It makes no SDK helper smoke call, Java numeric sanity call, physics import/evaluation or fixture call.

Before acquisition, verify the exact reviewed head/tool/source/artifact/runtime/package/profile closure and current boot/workspace/run identity. Admission requires at least10minutes remaining under the job cap. The entire24-case process, including Java bridge execution, has a single300-second cap; reserve another5minutes for evidence handling. No300-second per-case interpretation. Atomically create the fixed durable annotated receipt `refs/tags/experiment-receipts/diamond-stop-contract24-v1` before the first case, binding payload head, freeze/profile/package hashes, actual bridge/classpath provenance, boot ID, resolved workspace, hosted run ID and attempt1. Existing receipt means permanent abort; do not delete/reset or mint another ref.

Require a previously absent acquisition output root; any existing or partial root permanently aborts this attempt, never resumed, replaced or repaired. Preserve the spent receipt after any failure, attempted IDs, partial rows and tracebacks. Always upload commands/logs/inputs/outputs/hashes and failure evidence. Pre-execution ordinary source/build failures remain zero-case preparation evidence. Registration authorizes implementation and zero-case compile/metadata preparation only. Freeze and obtain independent exact-payload review before any numeric controller/helper invocation or the sole24-case acquisition. No extra controller/helper evaluations are permitted outside its24 registered cards.

## Future scope is not authorized here

If this source characterization warrants another proposal, an actual STOP-option-only change would need to account for **heading as well as velocity**: zero post-stop speed changes the permitted turn rate. It must leave moving-option zero-max residual footprint collection (`move.py:903–906`), gun/self-origin prediction, MEA normalization, learner data, wave epochs and global MovementPredictor behavior untouched, or explicitly register a larger intervention. Existing two-wave continuation and four-expert guard families remain prior-art rejections. This diagnostic supplies no wins, damage, firing/hit-rate or strength evidence and opens no fork, game or confirmation budget.
