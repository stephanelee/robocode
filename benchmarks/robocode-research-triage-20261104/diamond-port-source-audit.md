# Diamond Port source/engine contract audit

Read-only research, 2026-11-04. No bot, test, diagnostic case or game was executed, and no repository source was changed. The only new artifact is this report. The SDK wheel was read as ZIP data in memory; SDK modules were not imported or run.

## Finding

There are concrete, active own-motion forecast mismatches that could motivate a separately registered fork of the strong port. The most precisely isolated one is **STOP prediction versus the actual SDK command**. The port also retains nonzero predicted speed after its wall-position adjustment. Neither finding establishes a performance problem, a measured opportunity frequency, or a strength improvement from fixing it.

STOP and movement-order differences are already described in `docs/opponents/diamond-port.md`; they are not new discoveries or new algorithms. I found no completed experiment that repairs the actual Diamond Port implementation in a distinct fork. The proposed first question below is a source-contract question, not permission to run a new benchmark or repeat a rejected Meridian candidate.

An important qualification: **the Python SDK sits between this adapter and the engine**. Replacing every classic velocity forecast with the server's raw target-speed formula would be unjustified. In particular, the often-quoted raw-engine reversal 8→3 is not generally the adapter's effective reversal when `set_forward(±100)` is used.

## Pins and source scope

- Port: `fduch-stranger/robocode-bot@67b68c22e41c18281a3fce9ef638e3c23a0b47cf`, selected-source SHA256 `a9bf0b5d62b510ce3a405db9dcbe11cf2c1bce3f982e2a5d12b9fdc76b8fbc92`. Inspected the retained executed source at `/private/tmp/robocode-meridian-peak/benchmarks/meridian-peak-v1/screen-evidence/stages/mp1-02-peak-diamond/opponents/diamond-port/` and its sibling `bot_core/`.
- Engine: `robocode-dev/tank-royale@c8ad3a8d19a843f6258d6f6f9db7f29229963903`, using the pinned original Kotlin sources retained under `benchmarks/meridian-wave-support/engine-source/`, plus existing contract documentation.
- Actual adapter SDK: `robocode_tank_royale-1.3.1-py3-none-any.whl`, SHA256 `074f28f1f62ede458801739d8f84651af8c33bb2aee3290e7841fbd9bc327df2`, literal URL/size from `tools/opponents/strong-melee.lock.json`. Its inspected source hashes agree with the previously frozen installed-package manifest:
  - `internal/bot_internals.py`: `3928eaca5f8742af01a9df0fda4fdb0d093e2a44f069da7dbbdfda2812653281`
  - `internal/base_bot_internals.py`: `a74971b17d1c828e5641ca7837561041a779970bf1b7c619f7dbd5b9b4fc6946`
  - `internal/intent_validator.py`: `df3a025da92f09a9508bb7c4ab5f53444e92b584ed79bdef010ae6e2e7918cf0`

Below, `D/` means the executed `diamond-port/` directory, `C/` its sibling `bot_core/`, and `E/` the pinned engine's `server/src/main/kotlin/dev/robocode/tankroyale/server/`. SDK line numbers refer to the exact wheel above.

## Active motion contracts

### 1. STOP: a real mismatch after composing SDK and engine

The live choice sets `robot.set_max_velocity(0.0)` at `D/diamond_port/move.py:821–836`, then still calls back-as-front movement. The adapter writes `bot.max_speed=0` at `D/diamond-port.py:224–226` and issues `set_forward(±100)` through `:218–242`.

The pinned SDK's `bot_internals.py:168–173,353–359` calls its finite-distance speed controller. `intent_validator.py:66–84` can initially calculate a braking intermediate speed. Crucially, `base_bot_internals.py:768–772` passes that result through `validate_target_speed`, and `intent_validator.py:44–47` clamps it to `[-max_speed,+max_speed]`. With max speed zero, the **sent target speed is zero**. The engine then returns zero immediately for positive speed and target zero (`E/rules/math.kt:24–45`), with negative-speed symmetry at `:25–26`.

But the STOP forecast uses `max_velocity=0` at `D/diamond_port/move.py:887–896,914–915`, through `_predict_surf_location` at `:962–964`, `MovementPredictor.next_location_toward` at `D/diamond_port/movement_predictor.py:174–181`, and `_get_new_velocity` at `:103–114`. Its classic deceleration clamp retains nonzero speed when the current magnitude exceeds two. For current +8 and long forward distance, the source expressions give forecast +6 while the composed actual STOP command gives zero. These are algebraic source implications, not newly executed fixture results.

This can change a predicted STOP pose even with zero body turn in open space, so it can be separated from the turn-order and wall issues. It does **not** show that STOP is chosen incorrectly in real games. Also note `move.py:903–906`: even moving options use a separate zero-max-velocity forecast while collecting the wave's residual footprint. A global change to every predictor call with zero max velocity would alter that approximation too; it would not be a clean STOP-option-only intervention.

### 2. Wall adjustment retains speed: a real state-contract mismatch

At `D/diamond_port/movement_predictor.py:62–72`, the predictor computes `next_velocity`, adjusts an out-of-field position at `:70–71`, and stores the original `next_velocity` in the resulting `RobotState` at `:72`. `_adjust_for_walls` (`:75–99`) changes only the point. Thus a nonzero-speed adjusted wall contact does not zero the predicted speed.

The pinned engine detects a constrained translation (`E/core/CollisionDetector.kt:240–266`) and sets `bot.speed=0.0` on every applied wall outcome (`:374–389`). `E/core/TurnProcessor.kt:68,75` then performs its second position constraint using the original pre-turn copy. Diamond's `SurfMover` actively uses the walls-enabled predictor (`D/diamond_port/move.py:962–964`); this is not merely unused library code.

Position adjustment is also a different algorithm from the engine's two-pass interpolation, so a velocity-only repair must not be called complete wall conformance. No frequency or route-choice effect was measured here. Earlier dual-aim and Meridian/Peak work covered wall physics in our own forecasts, not this port's predictor.

### 3. Turn/move order: confirmed, already known, broader than a STOP repair

`D/diamond_port/movement_predictor.py:52–69` caps turning from **old** speed, changes heading, computes new velocity, and moves along the **new** heading. The file's header explicitly states this classic-order choice and that its danger/escape calculations were tuned around it (`:1–7`).

The engine sets new speed, moves along the old heading, then turns (`E/core/TurnProcessor.kt:145–150`); its turn cap uses that updated speed (`:422–423`). This is a definite ordering difference. It is already documented, and correcting it is not presumed beneficial: learned normalization, gun self-projection, escape locations and surfing can all be coupled to this predictor. The predictor's own-location convenience path is used by gun and radar too (`D/diamond_port/gun.py:1045,1059,1131–1133`; `D/diamond_port/radar.py:55`). A global replacement would be a whole-model change rather than an isolated defensive stepping repair.

### 4. Do not infer a reversal bug from the raw engine formula alone

For the ordinary `set_forward(-100)` reversal while max speed remains eight, the SDK's recursive finite-distance controller (`intent_validator.py:66–84`) can request an intermediate **positive** speed. For the +8 example, that intermediate is +6 and is not changed by the ±8 clamp. The engine therefore receives +6, not -8, and also produces +6. The classic predictor can agree in this case.

The raw engine's +8 with negative target→+3 law remains true, but comparing that directly with the port's desired direction omits the SDK layer. A future conformance reference must compose **adapter → exact SDK controller/clamp → pinned engine**, not assume the adapter sends signed max speed directly.

## Bullet timing and geometry: avoid a false universal fix

The adapter accepts a fire request for the next turn (`D/diamond-port.py:244–250`) and confirmation merely clears its pending flag (`:412–413`). Listener bookkeeping is created at request time (`D/diamond_port/gun.py:1174–1179`). This is already documented prior art.

However, request-time wave epochs can be consistent with this engine's unusual sweep. `Wave.distance_traveled(time)` is `(time-fire_time)*v` (`D/diamond_port/wave.py:282–283`), and the intersection annulus uses that radius through the next one (`:462–463`). With request epoch s and creation event s+1, those radii at creation are `[v,2v]`, which agrees with the engine's advance-before-collision pipeline (`E/core/TurnProcessor.kt:77–82`; `E/model/BulletLine.kt:9–15`; `E/model/Bullet.kt:51–58`). Incoming firing waves are also assigned a request-relative epoch (`D/diamond_port/move.py:430–450`). **A universal “add one tick” correction is not established by this source audit.** Gun-origin forecast error and confirmation reconciliation are separate questions.

A secondary boundary difference exists in bullet-shadow search: `D/diamond_port/move.py:311–322` terminates using `BattleField.contains_point`, whose bounds are the bot-centre box `[18,width−18)×[18,height−18)` (`D/diamond_port/dia_utils.py:277–295`). Actual bullet removal uses the updated near point at the full arena edges, inclusive removal at x/y≤0 or x≥width/y≥height (`E/core/CollisionDetector.kt:269–271,469–473`). This is a real boundary-model difference, but the crossing test occurs before the model's boundary break, and shadows are a predictive geometric heuristic. I have not established a lost valid shadow, its decision effect, or a complete timing correction; do not market this as an exploit or combine it into a motion patch.

The square wave footprint (`D/diamond_port/wave.py:464–474,506–509`) versus radius-18 engine circle is another known approximation. A risk model may deliberately be conservative; different footprints alone do not establish a correctness defect in a probabilistic danger score.

## Shared-core reachability and prior-art check

Diamond's Python entry imports `bot_core.debug` and `bot_core.telemetry.timing` (`D/diamond-port.py:106–107`). Its gun, mover and radar import the port's own `MovementPredictor`. The staged `C/physics/movement.py` is therefore **not the active predictor for this Diamond port**. Its apparently more engine-like implementation is not a drop-in proof of Diamond behavior, and repairing that shared file alone would not address the findings above.

- Existing opponent documentation already identifies classic order, stop/raw reversal differences, footprints and request-time fire bookkeeping. I have qualified the reversal comparison with the actual pinned SDK composition.
- PR94's mask diagnostic and PR96's Peak defense used our own Meridian forecasting/risk policies; they did not repair or benchmark a Diamond fork. Their failed advancement/screen outcomes stand and cannot be recycled as evidence for this proposal.
- Meridian v5 continuation and PR64's four-expert guard are unrelated prior-art families that should remain rejected here; no extra-wave search or expert selector is proposed.
- The ledger/docs searched contain no completed strong-port fork that changes this STOP/wall stepping seam. That is an audit of the retained repository evidence, not a claim that the underlying physics idea is novel.

## Bounded prospective question, if selected

First register a **zero-game, controller-composition conformance diagnostic**, not a live candidate: 24 predetermined one-step cases, crossing two placements (open and near right wall), signed speeds ±8, command classes continue/reverse/STOP, and two requested body turns (zero and a fixed nonzero turn). Freeze explicit positions/headings/±100-distance commands, the exact SDK wheel/source hashes and engine reference before outputs. Record emitted SDK target speed, resulting pose/speed/heading, original predictor error, wall involvement and all exclusions; include no danger-model, gun or action-selection outcomes. Require exact rule agreement within a prospective floating tolerance for any proposed replacement; do not tune case selection after inspection.

The immediate question is whether a distinct port fork's **own defensive STOP step** can match the effective SDK/engine command while leaving its gun, learned views, virtual-gun selection, wave epochs, existing two-wave search, wall/footprint approximations and opponent unchanged. If changing the shared predictor would also change the zero-speed footprint collector or gun/MEA models, disclose that larger scope or keep the implementation scoped to the STOP-option transition; do not describe a global rewrite as one isolated fix.

Only a separately registered, independently reviewed fresh live comparison could determine competitive value. Any fork must have its own name/version/output/model paths, plainly mark the alteration, retain the complete original Zlib notice at `D/LICENSE` and shared Apache-2.0 notices, and leave the opponent pin/source/wheels unchanged. No game budget, implementation or strength claim is authorized by this report.
