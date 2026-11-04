# Specification for an independently implemented Diamond style bot

Version 1, 2026-11-04. Status: design and source analysis; no implementation or new battle results.

Build a separate Tank Royale bot combining contextual nearest-neighbor targeting, an adaptive virtual-gun selector, two-wave surfing, bullet shadows, independent radar, and minimum-risk melee movement. The objective is similar capabilities and learning behavior, with independently written code and independently obtained startup behavior. This specification does not promise identical commands or equal competitive strength.

## 1 Scope and provenance

The reference is **Python Diamond Port 1.8.28**, from `fduch-stranger/robocode-bot` commit `67b68c22e41c18281a3fce9ef638e3c23a0b47cf`. It is an altered Tank Royale adaptation of Voidious's Java Diamond, not the original tournament binary. The repository's [opponent lock](../tools/opponents/strong-melee.lock.json) identifies Python 3.13, Python Bot API 1.3.1, and the selected upstream files. Its aggregate source digest includes shared code and another opponent; it is not a digest of Diamond alone.

At the user's request, the analyst inspected the locally retained, pinned boundary, gun, movement, radar, wave, predictor, enemy, neighbor-view, spatial-index, and geometry code. Individual files were checked against the lock; see the [analysis provenance](diamond-port-analysis-provenance.json). The evolved startup data file was not opened or decoded. No reference code, startup data, trained tree, checkpoint, or source translation is part of this handoff.

This is a **source-informed specification for a separated implementation**. The analyst has seen the reference code and must not be described as a source-unexposed implementer. For the implementation boundary, provide the implementer this document and the [implementation prompt](diamond-port-implementer-prompt.md), with original source and source-derived datasets inaccessible. Provenance remains with the analyst/reviewer. A claim of a stricter process in which even the specification author never saw source would be inaccurate.

Throughout this document:

- **Reference behavior** means a fact observed in the pinned port, stated in new prose.
- **Required** means behavior the new implementation must provide.
- **Initial profile** means an independently chosen, unmeasured parameter set for the new bot. It is not Diamond's tuning and may change only through a separately registered experiment once frozen for measurement.

The deliverable is a distinct experimental Java 17 bot, tentatively `Facet`, under `experiments/Facet/`. Use a distinct descriptor, entry point, JAR, model paths, and telemetry identity. Do not modify, subclass, rename, or use checkpoints from production AISurvivor. Both duel and melee are specified, but implement and validate duel before melee.

## 2 What the reference actually does

### 2.1 Overall organization

The reference retains separate movement, gun, and radar components for a battle. Its turn loop applies deferred enemy-energy corrections, executes movement, gun, radar, then submits commands and receives events. A scan updates radar, movement, and gun in that order. Own body, gun, and radar rotation are decoupled. Round resets clear trajectories and unresolved waves but preserve per-enemy learning views and virtual-gun ratings. Starting a new game discards the components.

This produces online adaptation within a multi-round game. Twenty rounds in one process are not twenty independent cold starts. The reference has separate gun and movement enemy records; the new bot may share immutable observations while keeping the two learners and label types separate.

### 2.2 Targeting

An initial duel starts with a fixed, evolved nearest-neighbor startup model containing 1,000 points. At nine recorded duel wave breaks, the selector can retire this model and choose between a longer-history contextual guess-factor gun and a short-history anti-surfer gun. The switch counts wave breaks, not hits. Virtual waves generate training observations even when no physical bullet was fired.

The longer-history gun uses power, flight time, movement direction and relative heading, acceleration, forward/reverse wall availability, time since velocity change, and how close a virtual wave was to a real firing opportunity. Its feature weights change with observation count. It selects up to 225 neighbors, approximately growing with sample count divided by nine, and searches 59 candidate angles using a compact angular kernel and distance weighting.

The anti-surfer gun combines four recency windows with capacities 125, 400, 1,500, and 4,000 and up to three neighbors per view. It uses a Gaussian angular kernel. In battles that began as duels, labels are direction-normalized guess factors with separate wall-aware escape scales. In battles that began as melee, the survivor duel uses displacement-based targeting learned in that context. The reference has a separate displacement gun for that transition and a melee gun that combines projected targets across opponents.

Virtual ratings use smooth angular accuracy, adjusted for target angular width and reachable escape range. They are not physical hit percentages. Predictions are recorded before outcomes. Round reset discards pending predictions while retaining accumulated ratings.

The gun tries the previously prepared shot before aiming the next one. Expensive duel aiming occurs near gun readiness; otherwise it tracks the target directly and reevaluates gun selection. Firing requires zero heat, zero remaining gun turn, and energy strictly greater than the prepared power. The boundary treats accepted fire requests as firing immediately for internal waves and shadows, although actual creation is confirmed later by the engine.

### 2.3 Defense

The reference chooses among two orbit directions and stopping. It predicts trajectories toward wall-aware escape destinations, evaluates the angular footprint of the bot as the wave crosses it, and can add the best continuation against a second wave. It prunes continuations whose current danger already exceeds a cutoff. Stopping retains an orbit orientation and destination; it is not an arbitrary direction reset.

Danger combines a simple hit view, a contextual hit view, several strongly weighted recent-hit views, and conditional visit-based flatteners. A flattener penalizes repeatedly visiting learnable escape regions, even if they have not yet caused many hits. View activation depends on normalized enemy accuracy and, for flatteners, an uncertainty margin. Hit labels and visit labels remain distinct. Incoming movement-wave guess factors use the theoretical escape scale, while duel gun labels use precise asymmetric scales.

Danger also includes a baseline enemy hit rate, bullet damage, inverse time to impact, a distance penalty, and bullet-shadow coverage. With no enabled data, its prior emphasizes head-on and a positive high guess factor. Imaginary waves fill missing surf slots when estimated enemy heat is nearly cool. Their power comes from a contextual model of observed enemy shots. A real firing inference replaces the forecast.

Its distance controller prefers 650 arena units, with different gains while orbiting and surfing, and a 160-unit wall-smoothing lookahead. These are reference facts, not recommended universal optima.

### 2.4 Radar and melee

The reference locks a fresh duel scan with a 22.5-degree overshoot. Otherwise it sweeps by up to 45 degrees toward unseen or stalest opponents, with center-directed and sweep-continuity rules. A participant-count mismatch persisting beyond 25 turns clears radar bookkeeping.

Melee movement samples 64 destinations around the bot, usually 100–200 units away, limited by the closest enemy. Risk increases with enemy energy, observed damage threat, proximity, movement along the enemy bearing, and repeated visits to recent locations. It decreases when other opponents are closer to that enemy. The current destination receives a 0.8 risk multiplier. A five-step forecast rejects destinations likely to cause a wall collision, but the final fallback can still be unsafe. Randomized recent-location repulsion discourages repetition.

The melee gun projects nearest-neighbor displacement outcomes for each live target, rejects projections outside the field, normalizes sample weight per opponent, and scores candidate angles across all opponents, favoring nearer targets through inverse distance. It is not simply a duel gun pointed at the nearest bot.

### 2.5 Important adaptation defects and boundaries

The predictor uses classic Robocode ordering: rotate, update velocity, move. The local Tank Royale evidence instead describes speed update, translation on the previous heading, then rotation. Its stop/reversal law also differs from the pinned engine. The wave footprint is a 36-unit square, while Tank Royale collision geometry uses a circular bot. The adapter delays known damage/bonus corrections to align with scan-energy visibility. One alternative melee-origin projection supplies heading and velocity in reversed argument positions. Some zero-distance neighbor weights become infinite, and missing precise intersections can return zero danger.

These are reasons to verify a new engine-native design, not requirements to duplicate defects. Do not import these defects for superficial behavioral similarity. They do not by themselves establish a counterstrategy or explain all measured outcomes.

The existing [opponent analysis](opponents/diamond-port.md) reports AISurvivor's historical 0/60 first places and 3.886% mean score share against this pin. That is whole-bot negative evidence for a specific old AIS version. It does not isolate Diamond's gun, surfing, startup model, or radar as the cause.

## 3 Required runtime and information boundary

### 3.1 Legal observations

The decision system may use own state, own submitted commands, legal scans, public battle rules, and combat events delivered to this bot by the official API. Persist only this bot's derived data. Opponent source knowledge informs this specification, not live access to opponent state.

Never use opponent model files, private timing, selected gun, intended movement, future state, random seed, hidden bullets, debug telemetry, server spectator data, or reference process memory. An incoming wave is an inference, never a claim that an unseen physical bullet was observed. Own confirmed bullet trajectories are permitted for shadows.

### 3.2 Coordinates and rules

Use one internal convention: radians, east zero, counterclockwise positive, Cartesian positions. Direction vector is `(cos(theta), sin(theta))`; bearing is `atan2(dy, dx)`. Wrap differences into `[-pi, pi)` and headings into `[0, 2*pi)`. Forward/reverse movement uses signed speed. Tests must cover crossing zero and changing velocity sign.

The reference's classic angle conversion must not leak into this convention. If needed for analyst comparison, classic angle equals `pi/2 - nativeAngle`, wrapped appropriately. Use explicit `PORT_CW`, `PORT_CCW` labels only in reference discussion; implementer actions should use unambiguous signed mathematical orbit direction.

The general official [physics reference](https://robocode.dev/articles/physics.html) gives speed limit 8, body rotation limit `10 - 0.75*abs(v)` degrees per turn, gun limit 20 degrees, radar limit 45 degrees, power range 0.1–3, bullet speed `20 - 3*p`, bullet damage `4*p + 2*max(p-1,0)`, hit energy bonus `3*p`, and firing heat `1+p/5`. These summarize general rules; exact update order, clamping, cooling and boundary behavior must come from the frozen engine/API profile and measured contracts.

The official [coordinate reference](https://robocode.dev/articles/coordinates-and-angles.html) confirms the native convention. Current documentation pages can change and disagree on details such as wall damage. Neither current web prose nor classic Diamond helpers may override the pinned engine. Freeze the exact server, API, runner, rules and hashes before implementation validation. The existing local [scan-energy trace](engine-energy-trace.md) concerns Tank Royale 1.4.0; it does not automatically validate another engine release.

### 3.3 Command and event timing

Required turn transaction:

1. Collect the delivered event batch with event turn, callback order, round and own-state version. Deduplicate by event identity where available.
2. Apply lifecycle and death events. Reconcile own fired, hit, bullet-collision and termination events using bullet IDs.
3. Update scans and the timestamped energy ledger using the pinned visibility contract. Do not infer a shot from a partially applied same-turn event batch.
4. Resolve wave crossings and labels; update learned views and model versions exactly once per observation.
5. Predict candidate movement commands and choose one. Publish its predicted next own state to the aiming and radar stages.
6. Validate the prior firing plan against actual heading, heat, energy, target freshness and rule timing. Submit it only if valid. Create a pending request record, not a confirmed bullet.
7. Prepare the next aim and radar commands, submit one complete intent, and retain its sequence ID.

Event handler APIs may force a different call arrangement; preserve these logical dependencies. Never silently reorder timestamps to make a trace match. Distinguish `decisionTurn`, `requestedFireTurn`, `confirmedFireTurn`, and `firstBulletMotionTurn` explicitly. The engine fixture must establish when a shot uses the current gun heading versus a newly requested rotation.

On accepted fire, virtual predictions may be provisional, but a physical wave and any full-confidence bullet shadow require confirmed creation. If a pending request is absent from the expected confirmation batch, cancel it and record the discrepancy. No double shot accounting when acceptance and confirmation both arrive.

For a known hit at turn `k`, the local pinned visibility contract has unchanged scanned enemy energy at `k` and the damage/bonus reflected at `k+1`. Record those effects with their expected visible turn. Do not subtract the same damage at callback time and again at the next scan. Unknown wall/bot collisions or missing scans create uncertainty rather than certain fire detections.

## 4 Module contracts and state

Implementation names are free; responsibility boundaries are required.

| Component | Inputs | Output and ownership |
| --- | --- | --- |
| Engine boundary | API events and own state | Immutable observations, command conversion, round/game lifecycle |
| Observation history | Scans and own states | Timestamped states with observation/interpolation quality |
| Energy accountant | Scans and known effects | Residual energy-change interval and candidate shot intervals |
| Engine predictor | State, command, rules | Next state and collision facts; no model learning |
| Wave service | Confirmed shots, inferred shots, virtual origins | Wave lifecycle, crossings, associations and uncertainty |
| Neighbor store | Frozen features and labels | Bounded exact nearest-neighbor queries and data version |
| Gun ensemble | Target context and wave labels | Candidate angles, ratings, chosen firing plan |
| Duel planner | Incoming waves and learned danger | Movement command plus path, score and reason |
| Shadow service | Confirmed own bullets and waves | Time-qualified merged angular intervals |
| Melee planner | All known live opponents | Safe candidate destination and movement command |
| Radar | Scan ages, own predicted state | Decoupled sweep/lock command |
| Diagnostics | Decisions, events, counters | Observation-only telemetry and end-of-game evidence |

Required records:

- **State sample:** game/round/turn, position, body heading, signed speed, energy if known, provenance (`observed`, `interpolated`, `predicted`), and uncertainty. Own samples also hold gun/radar headings, heat, remaining command values and command sequence.
- **Enemy record:** stable battle bot ID, alive status, latest scan, observation history, last nonzero movement direction, acceleration estimate, velocity-change and reversal timers, 8/20/40-turn displacement features, energy ledger, inferred heat interval, power history, damage counters, per-enemy gun and movement views.
- **Wave:** unique ID, owner/target, role (`outgoing`, `incoming`, `virtual`, `forecast`), fire-time interval, origin interval, power interval, reference bearing, target context at origin, orbit sign, escape convention, confidence, actual-bullet association if any, shadow intervals, crossing history, outcome flags, and label-emitted flag.
- **Training observation:** feature-schema version, feature vector, label type/value, source wave ID, event time, provenance, quality weight and insertion sequence. Never interchange precise-GF, theoretical-GF and displacement labels without an explicit conversion.
- **Gun prediction:** target, gun ID, wave ID, prediction timestamp, angle, power and context version, then eventual score or cancellation reason. It must be immutable before observing the outcome.
- **Firing plan:** angle, power, target, creation/expiry turns, required heading tolerance, expected origin and state version. A stale plan cannot create a ghost bullet.

Round reset clears waves, unresolved shots/predictions, trajectories, radar locks, transient commands and timing estimators. Preserve per-enemy training views, resolved ratings, cumulative counters and feature schema for the game. A new game is cold unless a separately registered warm-start mode explicitly loads this bot's own compatible checkpoint. Never persist opponent IDs across games as if identity were guaranteed.

## 5 Physics and geometry

### 5.1 Predictor

Implement a pure one-turn function from own state, a complete command, and the pinned rule profile to next state. Include target-speed clamping, forward/reverse command behavior, movement/rotation order, remaining distance/turn bookkeeping, wall limits, and collision outcomes supported by legal information. Unknown enemy commands must remain unknown; predicted bot collisions use bounded opponent hypotheses.

The predictor must reproduce recorded engine transitions for acceleration, zero target speed, reverse from both signs, finite-distance stopping, maximum turn at multiple speeds, wall/corner contact and gun/radar decoupling. Do not approximate reversal as a constant subtraction of two or assume stop always means gradual braking. A requested inward heading does not guarantee the first translation is safe.

Initial profile uses radius 18 and a 1-unit safety margin inside the engine's legal center bounds; record these separately. Endpoint wall smoothing and next-step safety are separate checks. A candidate must pass the next actual translation before it can win a future-path comparison.

### 5.2 Histories

Interpolate only between observations from the same round. Position and velocity may be linear; heading uses the shortest angular difference. Mark interpolated samples. Do not interpolate across death, round reset or a gap over eight turns for training. Predictions beyond the latest scan are predictions, never retrospective observations. For displacement windows lacking history, use the oldest available sample and record the available horizon.

### 5.3 Waves and intersections

For an exact-origin wave with speed `b` and origin time `t0`, radius is `b*max(0,t-t0)`, subject to the validated first-motion phase. The wave represents possible bullet bearings, not an actual expanding circular projectile. Classify it as approaching, overlapping the bot footprint, center passed, and fully passed. For uncertain fire times and origins, compute a radial envelope; never collapse it to a precise observation without evidence.

Use the engine's circular footprint and swept motion over the tick. Compute the angular interval from the source whose bullet paths could intersect that swept footprint during the wave's radial shell. Handle tangent contact, angle wrap, source-inside-footprint and zero-width intervals. Union intervals across crossing ticks; retain multiple intervals if their union is disconnected. A conservative numerical subdivision with a declared error bound is acceptable initially. A square approximation is allowed only as an explicitly named diagnostic profile, not the default.

If a reliable interval cannot be computed, use a conservative positive-risk fallback and log it. Missing geometry must never make an option look perfectly safe.

### 5.4 Guess factors and displacement labels

The theoretical escape angle is `M=asin(min(1,8/b))`. Let `beta` be source-to-target bearing at origin and `s` the target's signed orbit direction at origin. Define `u=s*wrap(hitBearing-beta)`. Theoretical guess factor is `u/M`.

For precise gun labels, predict reachable escape magnitudes `Epositive` and `Enegative` in the two orbit directions from the actual target origin state, using the pinned predictor and walls. Then `g=u/Epositive` for nonnegative `u`, otherwise `g=u/Enegative`. Decode with the corresponding current-wave scale. Apply a minimum denominator of `1e-6` radians. Values outside `[-1,1]` are retained as outlier diagnostics; the default aim candidate domain remains physically reachable.

When lateral motion is effectively zero, retain the last nonzero orbit sign; use +1 only before any direction evidence. The sign is frozen at wave origin, never taken from the outcome state. Gun and movement models use the correct convention recorded on each wave.

A displacement label is average velocity from target origin to crossing center, rotated into the target's effective movement frame and reflected by the origin orbit sign. Projection reverses that transformation in the new context and iterates travel time to a fixed point. Initial cap: 12 iterations; detect two-cycles and choose the conservative later interception time. Reject invalid/out-of-field projections; fall back to another gun if all are invalid.

## 6 Enemy fire inference and power prediction

Let `Eold-Enew` be the scanned loss over an observation interval. Subtract known damage, add known energy gains, and subtract an interval for possible wall/ram damage, all aligned to when scans show each effect. The result is the possible firing-cost interval. A single certain shot requires consecutive reliable observations, a residual consistent with legal power, and a compatible inferred heat interval. Missing scans can hide multiple shots; use the heat constraint to enumerate feasible schedules or classify the interval unresolved.

For the initial profile, allow at most three shot-time hypotheses per enemy per observation update. More ambiguity creates a conservative radial envelope with reduced training quality; it does not invent many exact shots. Unmatched bullet events remain unmatched diagnostics. Association must agree on owner, power tolerance and radial/time consistency; initial radial tolerance is two bullet travel steps plus geometry uncertainty. Break ties by lower residual and then earlier wave ID. Never associate one physical event to two training hits.

Maintain estimated heat as an interval, initialized from the game rules. Cool it by the actual cooling rate; update it after a plausible shot. Missed or ambiguous observations widen the interval. Do not hardcode Diamond's 0.1 cooling assumption.

Power prediction uses normalized range, enemy energy and own energy, with equal coordinate weights in the initial independent profile. Store at most 512 reliable shot observations per enemy; average the closest `min(16,max(1,floor(sqrt(N))))` powers, with a cold default of 2.0. Clamp to legal and affordable power. This predicts power for forecast waves; it does not change an observed shot's power.

A near-cool heat interval with fewer than two credible incoming threats may create one forecast wave at the earliest feasible firing turn. Initial confidence is 0.25. Forecast waves affect movement but never count as confirmed shots, hits or visit training. Replace/expire them when actual evidence arrives or when the hypothesized firing window closes. Do not score both the forecast and its confirmed replacement.

## 7 Contextual nearest-neighbor learning

The reference uses a bucketed exact kd-tree with FIFO eviction, weighted squared Euclidean distances and dynamic weight changes. Required behavior is equivalent nearest-neighbor mathematics, not that data structure. Start with bounded arrays and exact linear search if timing permits; optimize behind the same contract.

For coordinates `x` and `q`, use `D=sum((wi*(xi-qi))^2)`. Note that coordinate scale `wi` is squared in the distance. Reject nonfinite inputs. Inverse-distance weight is `1/sqrt(max(D,1e-6))`; zero-distance duplicates remain finite. Neighbors are ordered by increasing distance and insertion sequence. Changing weights or feature schema invalidates all cached neighbors. FIFO eviction is based on observation sequence, independent of nearest-neighbor ordering.

### 7.1 Initial independent feature profile

All values below are proposed normalization choices, not recovered Diamond tuning. `F=range/bulletSpeed`, `H=max(1,F)`. Every wave freezes these at origin. Missing features carry a quality flag and use zero, with the observation's training weight halved; do not silently recompute them at outcome time.

| Feature | Definition |
| --- | --- |
| Power | `p/3` |
| Flight time | `min(F,100)/100` |
| Lateral speed | signed tangential velocity divided by 8 |
| Advancing speed | signed velocity toward the source divided by 8 |
| Acceleration | change in speed magnitude per elapsed turn, clipped to `[-2,1]`, mapped by `(a+2)/3` |
| Heading | sine and cosine of effective movement heading relative to source bearing |
| Forward wall availability | predicted steps until wall along current orbit divided by `H`, clipped to `[0,1.5]` |
| Reverse wall availability | same for opposite orbit |
| Velocity change age | `min(age/H,1)`; reset at speed change exceeding 0.5 |
| Reversal age | `min(age/H,1)`; reset when nonzero movement direction changes |
| Recent displacement | distances from states 8, 20 and 40 turns earlier divided by 64, 160 and 320 |
| Fire proximity | zero on physical firing waves; otherwise `min(nearestKnownFireOpportunityDistance/8,1)` |
| Population | live opponent count divided by initial opponent count |

Define forward/reverse wall availability with the same engine-native orbit simulator used for escape reachability, capped at `ceil(1.5*H)` steps. Initial heading weights are 1; flight, lateral, acceleration and wall weights are 2; all other coordinates are 1. Keep weights fixed in version 1 to avoid reproducing the reference's tuned schedules. Each gun/view selects an explicit ordered subset and records it in its schema.

### 7.2 Training and recency

Outgoing virtual waves learn target movement at reliable crossings whether or not a bullet was fired. Incoming real-wave crossings learn self visits; confirmed enemy bullet impacts and bullet-bullet contacts can supply incoming aim labels. Keep contact evidence distinct from actual hits. Skip duplicate, impossible, unobserved or terminally unresolved outcomes.

Store long-history gun data up to 8,192 observations per enemy. Recency gun views retain the latest 128, 512 and 2,048 observations. Movement stores hit history up to 1,024 and visit history up to 2,048. Sample timestamps use `(round,turn,sequence)`. Initial recency weights are `2^(-rank/32)` over queried neighbors, newest rank zero. No offline data or evolved priors ship in version 1.

Cache keys include wave ID, feature version, store version, weights version, role and neighbor count. An update to hits, shadows or wave identity invalidates the affected planner result; a surf-slot index alone is not a stable identity.

## 8 Gun ensemble and firing policy

### 8.1 Required guns

1. **Analytic startup gun:** engine-native constant velocity interception with wall-limited alternatives; cold fallback to direct bearing if projection fails. It is independently derived and replaces the excluded evolved dataset.
2. **Long-history contextual gun:** features power, flight, lateral/advancing speed, heading sine/cosine, acceleration, both wall availabilities, velocity-change age and fire proximity. Query `min(160,max(1,floor(N/12)))` neighbors; decode precise-GF labels in duel and displacement labels for melee-origin survivor data.
3. **Recent contextual gun:** same movement context without power or fire proximity; query up to five neighbors from each independent recency window and combine equal view contributions after per-view normalization.
4. **Melee displacement gun:** flight, movement, wall and recent-displacement context; project outcomes for every live opponent and combine angular densities across targets.

For duel guns, evaluate 61 evenly spaced signed offsets across the reachable escape interval, plus direct bearing and valid analytic intercept angles. Use Gaussian kernels with angular bandwidth `max(1e-4,2*asin(min(1,18/range)))`. Neighbor weights are quality times finite inverse distance times recency. Normalize per view before combining. Tie-break by smaller gun rotation, then smaller absolute offset, then signed offset. Fall back through recent, long-history, analytic and direct bearing when a chosen model has no valid evidence.

The reference switches at nine wave breaks. The initial independent profile permits learned guns after 16 reliable target-wave crossings. It continues to rate the analytic gun and does not permanently delete it.

### 8.2 Virtual selection

At a confirmed firing opportunity, freeze every eligible gun's prediction before any crossing outcome is consumed. Rate all guns on the same resolved wave cohort; exclude waves cancelled by death or inadequate observation from every gun's denominator.

For prediction error `e` and crossing half-width `h`, score `exp(-0.5*(e/max(h,1e-4))^2)`. Record both the raw score and a difficulty-adjusted score using the current wave's escape range divided by full angular target width, with that factor capped at 10. Initial selector uses the raw score's exponential mean with coefficient 0.05; the adjusted score is diagnostic only. A challenger gun replaces the current gun only when its mean exceeds the current mean by 0.02 after at least eight resolved predictions for both. Ties preserve the current gun. These are proposed selector rules, not the reference formula.

Do not recompute an old prediction with newly learned data, score virtual waves as actual shots, or update models on a wave before recording that wave's prediction.

### 8.3 Power and fire

Initial independent duel policy starts at power 2.0, uses 2.8 below range 180, and caps at the power needed to kill the latest observed target. Invert the actual bullet-damage rule for that cap: `E/4` for target energy up to 4, otherwise `(E+2)/6`. At range above 400 with own energy below 30, multiply nominal power by `max(0.1,(ownEnergy/30)^2)`. Clamp to legal power and reserve 0.2 own energy. If less than 0.1 remains available, do not fire. A zero-energy target receives minimum legal power if the plan is otherwise valid.

Melee starts at 2.5, drops to 1.5 when the selected projected target is beyond 500, and to 1.0 when own energy is below 20. Apply the same reserve and kill cap to an identified primary target. Score the chosen angle across all targets; record the primary target for energy accounting without claiming every bullet was aimed exclusively at it.

Expensive aim is needed within three cooling turns; otherwise direct-track cheaply. In melee, recompute expensive projections every other turn until four turns from readiness, then every turn. A firing plan expires after two turns or a target death, major scan update or incompatible own displacement. Fire only with legal heat, sufficient energy, fresh target evidence and validated heading error no larger than half the target's angular half-width, capped at 0.5 degrees. Use a 0.05-degree lower tolerance to avoid numerical deadlock. Never rely on a same-turn rotation applying before fire without the engine contract.

## 9 Duel movement and learned danger

### 9.1 Movement actions

Evaluate negative orbit, stop and positive orbit. Use back-as-front control: choose forward/reverse to reach the desired travel direction with at most a quarter-turn of body rotation, using the actual signed-speed predictor. STOP requests the engine-validated zero-speed command while preserving the chosen travel orientation for later continuation.

Initial desired range is 600, reduced to fit small arenas: `min(600,0.65*min(width,height))`. Let `a=clamp((range-desiredRange)/desiredRange,-0.8,0.8)`. For mathematical orbit sign `s`, desired heading is outward source-to-self bearing plus `s*(pi/2+gain*a)`, with gain 0.5 when surfing and 1.2 when no wave is credible. Wall smooth by searching in that orbit direction for an endpoint 140 units ahead inside safe bounds, at most 180 steps of one degree. Then independently test the actual next translation. If no safe heading exists, compare braking/reversal controls and minimize unavoidable collision damage.

### 9.2 Danger views

| View | Labels and features | Capacity and neighbors | Initial weight |
| --- | --- | --- | --- |
| Simple hits | Actual incoming aim evidence; flight, lateral speed, acceleration | 1,024; `min(24,max(1,N/5))` floored | 1 |
| Contextual hits | Same labels; add heading, walls, change age, displacement 8 | 1,024; `min(24,max(1,N/5))` floored | 2 |
| Recent hits | Latest same-label observations, contextual features | 64; up to 5 | 2 |
| Visit flattener | Reliable real incoming-wave self crossings; add reversal age and displacement 20/40 | 2,048; up to 32 | 0.5 when enabled |

Use theoretical-GF labels for all these movement views. Neighbor kernels estimate density over the candidate crossing interval. Integrate with nine equally spaced quadrature points including endpoints, using normalized trapezoidal weights; subdivide intervals wider than 0.2 radians until each segment is at most 0.2. Cap total quadrature points at 81 and use a conservative maximum-density fallback above that cap.

Enable simple hits after the first label; contextual and recent views after eight labels. Enable the flattener only after 50 resolved real incoming waves and a 95% Wilson lower bound on raw enemy hit rate above 0.08. Exclude bullet-bullet contacts from the raw shots-that-could-hit denominator, but permit their aim evidence in hit views with half weight. This is an independent activation policy, not Diamond's normalized thresholds.

With no enabled view use a normalized mixture: 50% density around head-on, 30% around a legal constant-velocity aim, and 20% uniform density over theoretical escape range. Use angular bandwidth from bot size. Do not import the reference's evolved or hand-tuned distributions. Blend a prior weight `max(0.1,1-Nhit/32)` with learned density during early learning.

### 9.3 Two-wave search

Select the earliest two credible threats by earliest possible footprint contact, with stable wave-ID ties. For each first action, simulate until the first wave clears the swept footprint, recording path and crossing intervals. At its center crossing, evaluate three continuations against the second wave using the already predicted history if that wave is already overlapping. Second-wave score is the best valid continuation; replanning occurs every real turn.

Initial per-wave cost is `confidence * damage(power) * integratedDensity * unshadowedFraction / max(1,timeToContact)`. For uncertain power use the maximum plausible damage. Add `0.05*max(0,desiredRange/max(predictedRange,1)-1)^2` as a dimensionless close-range penalty. Predicted wall/bot collisions add expected damage and invalidate an avoidable unsafe first step. Accumulated nonnegative cost allows branch-and-bound pruning against the best fully evaluated action. A bound must never be used as if it were an exact score.

Cap each simulated trajectory at 160 ticks, second-wave depth at one continuation, and total predictor steps at 2,400 per turn. On exhaustion use the best fully evaluated safe action; if none exists, use a safe analytic orbit. Cache only immutable context and preserve deterministic tie behavior: retain the previous safe orbit action within `1e-6` cost; otherwise prefer a moving action over stop, then negative orbit. Log budget truncation. Those limits are an initial profile, not a measured latency guarantee.

### 9.4 Bullet shadows

For each confirmed own bullet, intersect its time-indexed swept segment with the incoming wave's radial shell. Convert overlap to angular intervals, merge wrapping intervals, and retain validity times and bullet identity. Only coverage that occurs before the candidate enemy bullet could reach us can reduce danger. Do not use a bullet after its confirmed death; invalidate affected future intervals when it hits a bot, another bullet or a wall.

Initially cap shadow reduction at 75% for inferred incoming shots and 100% only for geometrically certain, event-supported conditions. Compute the unshadowed fraction over the candidate footprint interval. Forecast outgoing bullets may not produce full-confidence shadows. Tests must cover partial overlap, complete coverage, disjoint intervals, wraparound, and collision-before-shadow invalidation.

## 10 Melee behavior

Activate melee when more than one enemy remains; transition to duel without deleting learned per-target data. Preserve a feature indicating whether training originated in melee. Do not treat stale melee scans as current duel observations.

Generate 48 destinations uniformly around self at a seeded radius between 100 and 180, capped by nearest-enemy distance, plus the current destination and an inward safe point. Clamp to safe bounds and deduplicate. For each destination use:

`risk = sum(enemyThreat * energyRatio * alignment / ((distanceSquared+36^2)*(1+closerOpponents))) * revisitPenalty`.

Here `enemyThreat=(damageReceivedFromEnemy+5)/max(1,turnsObservedTogether)`, `energyRatio=clamp(enemyEnergy/max(ownEnergy,0.1),0.25,4)`, and `alignment=1+abs(cos(enemyBearing-travelHeading))`. Count closer opponents using their latest legal scans, explicitly discounted for scan age; beyond eight turns use no shielding credit. Revisit penalty is `1+sum(100/(distanceToRecentPointSquared+100))` over at most 32 own samples taken every eight turns. Apply a 0.85 multiplier to the retained destination. These are independent initial choices.

Rank candidates, simulate at least eight actual control steps for walls and bounded opponent collision risk, and choose the least-risk safe candidate. Never return the last unsafe candidate simply because the list is exhausted. If none is safe, choose the emergency control minimizing imminent damage and label that fallback.

For melee targeting, query up to 80 total displacement samples split equally among reliable living targets; each target receives at least one query if available. Normalize weights per target before adding Gaussian angular density divided by projected range. Candidate angles are valid projected bearings. Consider shortest gun turn on ties. If no model has valid projections, use analytic interception on the nearest fresh target.

## 11 Radar

Decouple radar from gun and body. With one live target and a fresh scan, compute bearing from predicted own next position to the target's latest legal location. Overshoot by the smaller of 20 degrees and the radar limit after accounting for the required turn, with a minimum two-degree overshoot when feasible. On missed scans, sweep in the prior direction toward the uncertainty sector; after four turns without a scan perform full sweeps until reacquired.

In melee prioritize unseen participants, then greatest scan age; ties use minimum feasible scan time and bot ID. Keep sweep direction if reversing would repeat a sector just covered. A known participant-count mismatch for 24 turns clears only radar bookkeeping and forces acquisition; it does not delete enemy learning. Unknown participant locations require an acquisition sweep, not an invented coordinate. The radar stage may use the movement predictor's proposed own location, but never a future enemy observation.

## 12 Bounded execution and failure handling

The new bot must run deterministically for the same event stream, seed and rule profile. Split random streams by subsystem and record seeds. Timing-based budget exits can reduce reproducibility; prefer deterministic work quotas, with a wall-clock emergency guard recorded as such.

Initial hard limits per enemy are in sections 6–9. Also cap transient waves at 256 per enemy, retaining unresolved real threats before virtual/forecast waves; remove expired resolved waves. Own and enemy histories retain 512 ticks plus samples referenced by unresolved waves. Global neighbor storage limit is 128 MiB estimated payload; if that cap would be exceeded, evict oldest low-priority virtual samples first and record the event. Set and report a JVM memory limit separately.

Target p99 decision time below half the configured turn budget on the frozen runner. Measure movement, gun, radar, inference, storage and GC separately. Turn-budget success is empirical; work quotas alone do not prove it. Run a bounded worst-case fixture with full stores and long-lived waves before battles.

Reject NaN/infinite features and scores before command submission. Use explicit finite denominator floors. Each subsystem failure increments telemetry and produces a safe fallback; it may not reuse a stale fire request. Three consecutive failures in the same subsystem disable that learned subsystem for the round and use its analytic fallback. A lost/skipped turn reconciles actual own state and pending commands before prediction resumes. Keep legal model data but never pretend a missed observation was seen.

## 13 Verification and acceptance

### 13.1 Required semantic tests

| Test family | Acceptance condition |
| --- | --- |
| Native angles | Cardinal vectors, wraparound, effective reverse heading and orbit sign round-trip correctly |
| Engine motion | Recorded one-step acceleration, stop, reverse, turn, wall and corner transitions match within `1e-6` position/speed and `1e-6` degrees, or explicitly documented engine precision |
| Fire timing | Accepted/rejected/confirmed requests account for exactly the real shots and correct origins/headings |
| Energy visibility | Isolated damage and hit bonus do not become false shots; simultaneous hit/fire and missing-scan cases retain uncertainty |
| Wave geometry | Approaching/crossing/passed states and swept-circle tangencies work across angular wrap |
| GF conventions | Sign and asymmetric scale encode/decode correctly; movement and gun labels never mix |
| Neighbor mathematics | Exact queries match brute force, including changed weights, duplicates, eviction and empty stores |
| Learning chronology | Predictions precede labels; duplicate callbacks do not train twice; virtual visits are not physical hits |
| Lifecycle | Round reset retains learning but clears unresolved state; new game is cold |
| Surfing | Two-wave continuation changes a controlled decision; stop uses actual engine physics; missing intersections are costly |
| Shadows | Death times, partial coverage, circular wrap and pending-fire cancellation are respected |
| Melee and radar | Unknown/stale targets, last-survivor transition and all-unsafe destination cases have defined behavior |
| Runtime | Worst-case stores and waves respect quotas, remain finite, and expose every fallback/skip |

Use independently constructed fixtures and actual engine event traces. Do not use Diamond private decisions, exported training data or reference source execution as a training oracle. The analyst may investigate reference behavior separately, but source-derived fixtures must not be mislabeled as independently observed black-box behavior. Do not execute or consume unrelated already registered diagnostics such as STOP24 as part of this specification task.

### 13.2 Staged delivery

1. Engine boundary, rules profile, chronology recorder and lifecycle; prove observation and command timing.
2. Geometry, histories, finite exact neighbors, wave service and analytic gun/radar; pass semantics before learned behavior.
3. Long/recent guns, virtual selector, independent startup, energy policy and persistence tests.
4. Fire inference, learned danger, two-wave search and shadows; freeze duel candidate.
5. Register and run bounded duel performance screen; keep a rejected candidate's evidence.
6. Add melee only after duel correctness and review; register its own screen and transition checks.

Every stage requires a buildable standalone artifact and a short report of what passed, what remains approximate, and exact source/JAR/rules hashes. Implementation agents open PRs; a separate agent independently reviews and merges only after required CI. The implementer must not merge its own PR.

### 13.3 Proposed prospective battle protocol

This is an implementation-ready proposed budget, not an executed experiment or permission to reuse historical confirmation games. Before the first gameplay run, copy the final frozen protocol into `docs/` and the relevant issue, identify the exact source/JAR and engine profile, and obtain the repository's independent premeasurement review. Lock seed availability and seat-order handling; if the runner cannot control either, document the limitation before collection.

Initial screen: **200 rounds**, comprising three independent cold 20-round games each against pinned Diamond Port, Retcon and Adaptive Prime (180 rounds), plus two 5-round games against each of two already pinned official sample bots (20 rounds). Pair seat orders where supported. All rounds in a game share online learning; games are independent cold populations. Report both whole-game metrics and round-index learning curves. Do not adapt parameters midway.

Predeclared screen gates for advancing to confirmation: every scheduled game completes with valid identity/artifact provenance; no numerical failures or unhandled exceptions; skipped turns below 0.5% of scheduled bot turns per opponent cohort; combined wall/bot collisions below 1% of bot turns; mean equal-game score share above 40% against both Diamond and Retcon; at least 50% against Adaptive and each sample cohort. These are proposed engineering gates, not validated equivalence thresholds. Failing any gate rejects this profile for promotion; no favorable subset or rerun replaces the failure.

A screen passing those gates permits only the claim that the candidate passed the screen. To claim it beats Diamond or Retcon, preregister a separate fresh confirmation for that opponent, initially ten independent cold 20-round games (200 rounds per claimed opponent). Require both majority official round first places and the lower bound of a 95% game-cluster bootstrap interval for mean equal-game score share above 50%. Use 10,000 resamples of whole games, fixed analysis seed, and preserve every game. Ten clusters still limit precision; report individual game shares and the interval. Confirmation against one opponent does not establish general strength or original Java Diamond equivalence.

If a change is later transferred to AISurvivor, run a new paired candidate/baseline protocol with Retcon preservation and the pinned official sample regressions. These standalone-bot screen results cannot validate production integration.

### 13.4 Evidence bundle

Retain all raw official results and legal event traces, launch commands, game/round counts, wall-clock budget, timeout/skip/exception logs, invalid attempts and reasons, source and committed-JAR hashes, reproducible build details, source/JAR parity outcome, runtime/OS/JDK/engine/API versions and hashes, opponent lock and models policy, seeds and seat order, initial/final private model hashes, schema/config hash, latency quantiles, shots requested/confirmed/hit, damage given/taken, energy economy, collision counts, achieved range, scan ages, inferred/ambiguous waves, gun ratings and fallback counts.

For every experiment update [improvements.md](improvements.md) with measured budget/results, accept/reject decision, PR and detailed report/raw links before declaring completion. Keep offline fixture correctness separate from competitive strength. All GitHub Actions jobs use exactly `ubuntu-26.04`. Build/test AISurvivor and preserve source/committed-JAR parity under Java 17. Complete Gradle builds before switching to JDK 25 for Retcon's source launcher. Preserve Stéphane Lee's author identity and existing evidence/history; add no AI authorship trailers.

## 14 Explicit departures and remaining limits

| Topic | Reference | Required new design |
| --- | --- | --- |
| Cold gun | Fixed evolved 1,000-point dataset | Independently derived analytic startup; no copied model |
| Predictor | Classic order and speed law | Frozen Tank Royale engine-native contracts |
| Footprint | Classic square in waves | Engine-native circular swept geometry |
| Shot bookkeeping | Request-time internal firing | Pending/confirmed distinction |
| Neighbor zero distance | Can yield infinite weight | Finite floor and deterministic ties |
| Memory | Some uncapped views | Explicit FIFO capacities and global cap |
| Danger failure | Empty intersection can score zero | Conservative positive-risk fallback |
| Tuning | Original learned weights and schedules | Declared independent initial profile |
| Melee alternate origin | Swapped heading/speed projection arguments | Typed, engine-native projection |

Consequently the new bot reproduces a strategy family rather than Diamond's exact decisions. It cannot claim startup equivalence without independently trained and separately evaluated data. Source inspection supports the architectural analysis; only new, properly registered battles can establish the new bot's strength. No source/JAR gameplay changes or battles were made to prepare this specification.
