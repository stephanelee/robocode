# Opponent analysis and challenger design

This study covers the **exact opponents used in our retained games**, not similarly named bots or general tournament reputations. It combines source analysis with measured outcomes and distinguishes those from untested counterstrategies. The next build is a separate [DiamondChallenger](../diamond-challenger.md); production AIS remains unchanged.

## Scope and evidence

| Opponent | Frozen source | Our AIS 1v1 evidence | Detailed analysis |
| --- | --- | --- | --- |
| Diamond Port1.8.28 | `fduch-stranger/robocode-bot@67b68c22e41c18281a3fce9ef638e3c23a0b47cf` | 0/60 first places;3.886% mean share | [Diamond: guns, surfing, adaptation, physics and counters](diamond-port.md) |
| Adaptive Prime1.0 | Same Python repository/pin, including shared bot_core | 3/60 first places;17.683% mean share | [Adaptive: effective configuration and shared-system behavior](adaptive-prime.md) |
| Retcon0.2.0 | `robocode-dev/rumble-bots@28981c7a469ac064194b41001cd7183ec5ecf189` | Fresh confirmation187/200 first places;86.745% mean share | [Detailed mechanics, resets and counter-tradeoffs](https://github.com/stephanelee/robocode/pull/57); [historical strategy study](../retcon-strategy.md) |

The Python [source/runtime lock](../../tools/opponents/strong-melee.lock.json) and [Retcon lock](../../tools/opponents/retcon.lock.json) fix file identities, dependencies and licenses. AIS results concern their documented source/runtime/arena/training regimes, not a universal ranking. Diamond Port is an experimental Python adaptation; its original Java namesake's championship rating is not inherited. Upstream local validation claims are identified as author reports in the individual analyses.

The valid baseline losses are preserved in [PR51](https://github.com/stephanelee/robocode/pull/51). Damage dealt/taken was383.6/5050.5 against Diamond and1464.9/3851.8 against Adaptive. Wall contacts were9 and2; AIS skips were zero. That identifies severe unfavorable combat exchanges, without attributing them to one gun or movement component. The [stronger melee](../improvements/strong-melee-baseline.md) is separate: Diamond51, Adaptive4, Retcon4 and AIS1 first places out of60. It does not determine a universal 1v1 order.

## The important differences

| Mechanism | Diamond Port | Adaptive Prime | Pinned Retcon |
| --- | --- | --- | --- |
| Offensive model family | Pre-populated Perceptual gun, then context-conditioned TripHammer/AntiSurfer learned guns | Linear fallback, learned dynamic-cluster GF, segmented traditional GF and historical displacement | Head-on and linear virtual-gun selection |
| What repeated motion exposes | Signed velocity, heading, acceleration, wall limits, time since velocity change, recency-specific views | Velocity/acceleration/change-age/walls plus contextual and historical path evidence | Mainly current location and linear velocity |
| Defensive planning | STOP/CW/CCW, two-wave lookahead, footprint-integrated learned danger and bullet shadows | STOP/CW/CCW, two-wave continuation, segmented danger/flattening and estimated shadows | Simpler nearest-wave/global-hit-profile surfing with orbit fallback |
| Preferred duel spacing |650px |480px preference, minimum380px |450px orbit preference |
| Main close-pressure cost |2.95 power below150px or on rammer classification |Strong close power, repulsion and retained older waves |Higher close-range exposure to its simpler guns and surfer |

The individual reports contain immutable line-level citations and the effective runtime defaults. These are mechanisms, not proof that an apparent limitation can be exploited successfully.

All three adapt within a multi-round game. Fresh-process games reset opponent learning differently from another round in the same game. Confirmation therefore needs several fresh games, not a claim that every adaptive round is an independent cold duel. Gun-selection scores, virtual wave passages and inferred danger distributions are estimates; they are not automatically actual-hit probabilities or private engine truth.

## What the failed AIS changes taught us

AIS's successful Retcon controller was designed around clearance from head-on and linear aim. That model family is a good match to Retcon's limited guns but does not represent Diamond's or Adaptive's contextual targeting. AIS did collect legal matched-hit danger bins, yet the duel controller originally ignored them. This justified a concrete experiment, not a guaranteed fix.

[PR54](../improvements/learned-duel-danger.md) replaced the analytic score after sparse hit evidence. Advanced screen shares improved modestly, but Retcon lost16.01points and the candidate was rejected. [PR55](../improvements/blended-duel-danger.md) kept at least75% analytic weighting. Its screen passed, but fresh confirmation gave only+0.288points against Diamond and−0.889points against Adaptive; it too was rejected. Retcon guards passed in that confirmation. No favorable screen is pooled into confirmation, and no unmeasured blend is promoted.

These results do not establish that all statistical movement learning fails, or explain the causal effect of every profile update. The profiles were sparse, global matched-hit distributions; the opponents condition on much richer contexts. The available telemetry lacks profile-activation and opponent-selected-gun traces. More code complexity by itself is not the next objective.

## First coherent challenger and alternatives

The first standalone design tests **controlled interception pressure** with seeded flank phases, physically consistent wall avoidance, independent radar and coordinated close-range gun/energy handling. It aims to shorten reaction time and change the repeated long-range movement context. The prototype is a new official-API entry point, not an AIS subclass or renamed checkpoint.

The hypothesis has real costs. Diamond tries to maintain650px and escalates close/rammer power, so approaching may be difficult or lose the damage exchange. Adaptive filters some new movement waves outside150–700px, but older waves can remain and its close gun/repulsion still operate. Neither fact is a guaranteed blind spot. Seeded phase changes also do not make movement unlearnable: both advanced guns explicitly use acceleration and time-since-change context.

The prototype must measure **achieved range**, confirmed shots/hits, damage economy, collisions, skipped turns, official score and first places. Low score with little time near the intended range would weaken the pressure mechanism; reaching close range but losing damage exchanges would suggest a different failure. These are interpretation aids, not retroactive changes to acceptance gates.

If the bounded first design fails, the next source-grounded alternatives are a compact context-conditioned displacement/GF gun, richer legal danger conditioning, or forecasting a family of feasible opponent movement options with the real engine. Diamond's private learned views, wave shadows and selected option remain hidden; source knowledge does not permit importing them into live decisions. There is no automatic model-family or parameter search, and no claim that firing less, fixed offsets, always ramming or radar blinding is already a winning counter.

## Measurement and transfer

Freeze every prototype's source/JAR, opponent/runtime pins, model policy, budget, seeds and rules before play. Use a bounded pilot to reject weak ideas, then genuinely fresh multi-game confirmation for a promising one. Publish all outcomes, failures, private endpoint models and hashes in [improvements.md](../improvements.md). Independent exact review precedes games and an independent agent reviews/merges results. A development gain is not a Diamond-beating claim.

The challenger itself has no Retcon-preservation gate: it is intentionally free to specialize. Once it demonstrates repeatable Diamond wins, transfer an attributable idea into a separately frozen AIS candidate and run fresh Diamond/Adaptive comparisons plus absolute/relative Retcon and sample guards. Challenger results cannot validate the integrated AIS source. Melee remains later; its known unsafe-first-step wall problem is documented as backlog rather than silently mixed into this 1v1 experiment.
