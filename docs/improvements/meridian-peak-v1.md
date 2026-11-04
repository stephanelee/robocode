# MeridianPeak0.1: prospective coherent defense prototype

Registered before implementation or new games. Follows the rejected held-gun
comparison(PR92) and rejected mask-only advancement(PR94). Preserve both outcomes.
No prototype source/JAR or runtime is considered frozen until a later exact-payload
comment and independent review; no game may precede those prerequisites.

## Hypothesis, novelty and isolation

One inferred bullet should contribute its closest modeled finite-sweep exposure
once per ray hypothesis, instead of accumulating a penalty for every nearby
forecast tick. An engine-faithful own-route forecast may then choose more useful
escapes. This is a heuristic whole-defense hypothesis, not calibrated hit probability.

Create a distinct Java17 bot under experiments/MeridianPeak, nameMeridianPeak,
version0.1.0, own launcher/JAR/model/metrics paths. Fork actual Meridianv3 source
809b880cdcee98d7946d05e6bcd30f623f604348, not the rejected held-gun policy. Keep
its gun selection/fallback, power, radar, energy/scan corrections, incoming-hit
learning and outgoing crossing learning. No prior training bank or replay rows
are deployed. Existing Meridian/Breakwater/Challenger and production AISurvivor
source/JARs remain untouched.

Keep original Flight.step unchanged for gun forecasts. Own defensive routes use
a separate pinned-engine step with both wall-constraint passes, old-heading
translation/new-speed turn limit and exact signed-speed rules. This distinction
is deliberate; the v3 gun's forecast approximation is not silently changed.
Keep both orbit signs,50-tick horizon, preferred range380,110px wall probe,
switch cost2, wall cost0.1*max(0,40-edge), old tie choice(-1), and current wave
retirement F>range+36. No STOP option, speculative shot or second-wave action tree.

For each actual legally inferred wave, take the existing HitAngles mixture as
weighted ray hypotheses. Cold/unoriented fallback uses head/lead with weight1each.
With oriented learned support N, head/lead each weigh4/(4+N); bin i weighs
2*n_i/(4+N), using unchanged GF offsets/orientation/escape and existing decays.
Weights total2; keep the same learned-state update rules. Read-only ray/snapshot
accessors may be added without changing the learner.

For each of the two50-step routes, each modeled ray advances at v=20-3p. At future
turn t+i its near radius is(t+i-waveTurn+1)*v and far radius near+v. Remove a ray
when its NEAR endpoint is x<=0,y<=0,x>=width or y>=height before that tick's sweep.
Use target CENTRE to FINITE segment squared distance d2, not infinite-line miss
and not max(0,d-radius18). Its kernel is40/(1+d2/500). Take max over future ticks
separately for EACH ray, then weight/sum rays per wave and sum waves. Add unchanged
wall/switch costs. Each wave is bounded by80 apart from floating-point tolerance.
A modeled contact is not simulated as damage or used as a learning label.

This is sum_ray(weight*max_tick(kernel)), not max_tick(sum_ray(weight*kernel)).
Breakwater0.2/0.3 already uses the latter per-wave mixture peak, and its Diamond
pilots failed(PR67/69). AIS DuelSurf evaluates a crossing/worst-wave clearance;
Challenger WaveSurfMove unions footprints then integrates density. Peak danger is
not novel by itself. The present finite-segment/per-ray aggregation, ray-wall death
and Meridian two-route combination is a new coherent prototype, not evidence of
success or an isolated causal test of MAX. Removing dwell-time summation changes
danger's relative scale despite retaining numerical wall/switch coefficients.

## Required contract checks before source/JAR freeze

Build/test under actual Java17 and preserve committed/source JAR parity for AIS,
existing Meridian and new Peak; separately verify exact control809b880 source/JAR.
Use existing pinned engine c8ad3a8d19a843f6258d6f6f9db7f29229963903.
Contract cases: finite-segment endpoints/interior and first-sweep phase; near- vs
far-endpoint wall death; max-before-mixture ordering; unchanged learned ray weights
and <=80 bound; no oracle/private input; source-equal v3 gun/orbit/learners; movement
only two-pass physics including the existing golden corner case translated to
800x600: own(780,580,atan2(4,7),8),target8,turn0 ->(782,581.1428571428571),speed0.
Do not claim these tests establish gameplay strength or use the spent PR94 matrix
as fresh confirmation. Any preparation/test failure is retained before final freeze.

## Prospective game schedule

All games are1v1,800x600, cooling0.1,inactivity450,turn timeout100000us,ready
10000000us,defaultTPS-1; connect80s,battle1500s,final callback flush10s,process1680s.
Never access7654; occupied registered ports invalidate instead of choosing another.
Each game starts a fresh server/runner,both bot processes,stage/home,private outputs
and models. Learning persists within one5- or20-round game only. No model/state or
observations transfer across games or from the screen to confirmation. Own seeds
do not seed/pair engine starts, native Diamond/Retcon randomness or scheduling.

Screen: complete all five VALID games regardless of scores; stop at first invalidity.

| GameID | Own bot | Opponent | Own/AIS seed | Rounds | Port |
|---|---|---|---:|---:|---:|
|mp1-01-control-diamond|Meridianv3|Diamond Port|61000|5|32900|
|mp1-02-peak-diamond|MeridianPeak|Diamond Port|61000|5|32901|
|mp1-03-peak-retcon|MeridianPeak|Retcon|61000|5|32902|
|mp1-04-peak-ais|MeridianPeak|AISurvivor TRAIN|61000/61000|5|32903|
|mp1-05-peak-walls|MeridianPeak|official Walls|61000|5|32904|

Screen passes only if ALL five games are complete/valid, all required own/enemy
confirmed-shot denominators nonzero and own skipped turns0; candidate Diamond
score share strictly exceeds contemporaneous control and uncapped incoming damage
per100 enemy confirmed creations is strictly lower; EACH of the four candidate
matchups has share strictly>35% and at least1 official first place. A failed guard
rejects this candidate for advancement. No favorable rerun or threshold waiver.
One control game and uncontrolled starts limit inference; transfer guards are
absolute, not comparisons to contemporaneous transfer controls.

Only a fully valid screen PASS permits the unchanged candidate's fresh confirmation:
Diamond3games,seeds61011/61012/61013,ports32910/32911/32912;
Retcon3games,seeds61021/61022/61023,ports32913/32914/32915;
AIS TRAIN3games,own/AIS matching61031/61032/61033,ports32916/32917/32918.
IDs mp1-c-diamond-g01..03,mp1-c-retcon-g01..03,mp1-c-ais-g01..03. Run in that order,
20rounds each, all valid games despite scores; stop first invalidity. Each opponent
separately must have equal-game mean share>50%,more than30/60 official first places,
and at least2/3 games individually>50%share, with all validity/skip/shot guards.
No screen pooling, candidate/source tuning or extra game opens this gate.
Confirmation is a bounded in-roster result with variation/uncertainty limits,
not an unqualified claim against the original championship Java Diamond or all bots.

Budget25 screen rounds +conditional180 confirmation =maximum205 physical rounds.
Separate hosted stage jobs: screen cap180min/admission>=150min remaining;
confirmation cap300min/admission>=262min remaining. These margins cover the fixed
process caps plus10min verification/reduction/artifact overhead. An incomplete
attempt is not replaced, resumed or called a completed comparison.

## Opponent and runtime pins

Control source809b880, JAR5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9.
Diamond Port1.8.28: upstream67b68c22e41c18281a3fce9ef638e3c23a0b47cf, selected
sourcea9bf0b5d62b510ce3a405db9dcbe11cf2c1bce3f982e2a5d12b9fdc76b8fbc92;
Retcon0.2.0:28981c7a469ac064194b41001cd7183ec5ecf189, tree
5f7dfaaedefc39bd93d86440a2996342583d57b75b861942dc1e1899b7164bb2,
API1.3.1 acb1858a7b83b45ddd153db52fcea4d855b95cc442f643f94f299f0ee2258ce2.
AIS TRAIN1.2.1 JARb41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800.
Official Walls comes unmodified from sample-bots-Java-1.4.0.zip SHA256
1707c1011a37c6b854fc79d68549ab204007bac6d33d803fa254152ac0765fe2; freeze its actual
descriptor version/all staged hashes before collection rather than inventing them.
Runner/server1.4.0 JAR02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf.
Preserve all source/licenses/lock files; no live opponent-private input.

All Actions jobs use ubuntu-26.04. Build/test/parity on exact Temurin17.0.21+9
BEFORE Temurin25.0.5+8 battle/source launch. Native Python3.13.18 and existing pinned
source/wheel environment with exact installed-package manifest52b7bfcf10f8226711798f27784be64583ea4c8fc676452d27f6e16b906bf191.
Accept only the two already registered Ubuntu image/runtime profiles; explicitly
freeze participant asset substitutions/additions before games. Runtime/profile
or hash mismatch stops; no silent OS/JDK/Python fallback. Zero-game preflight may
collect provenance but cannot approve an unregistered runtime by measurement.

## Telemetry, validity, review and once-only execution

Interface: MERIDIAN_PEAK_SEED; MERIDIAN_PEAK_METRICS marker path;
MERIDIAN_PEAK_STATE_DIR absent fresh output directory (no model input/loading).
Preserve per-round final/revised own model snapshots and an endpoint, own legal
hit/crossing counters, complete event markers, planner wave/ray/kernel counts,
peak per-wave risk and decision-time totals/maxima. Endpoint/round counters must
not advance twice when callbacks arrive after round-end; preserve terminal model
mutations during the closing drain before final sealing. Contract tests must cover
this known SDK phase. Use
explicit own-state DTOs, not bot/SDK object serialization or opponent files.

Official scores, first places, confirmed shots/hits, uncapped damage dealt/taken,
shot energy, hit rates,firing rates,wall/bot contacts and skips come from retained
observer events with exact setup/participant identities/contiguous ticks/round
endpoints. Private metadata/models must be complete, finite, match seeds/cold
starts/rounds and reconcile declared counters; no missing counter is reported0.
Keep all unresolved/masked labels and required denominator failures. Groundtruth
observer/enemy state is for post-game validation only, never deployed decisions.

Freeze candidate artifact/source, baseline/opponents, payload/tool/launcher hashes,
full runtime/package verification and all stage manifests before independent
pre-execution review. Each stage needs a fresh exact-head clearance token and
an atomic durable annotated receipt ref, binding stage,payload,GitHub run/attempt,
actual boot/workspace/profile/package/tool manifests. Fixed refs:
refs/tags/experiment-receipts/meridian-peak-v1-screen and
refs/tags/experiment-receipts/meridian-peak-v1-confirmation.
Existing ref aborts,never delete/reset it; run_attempt1 and serial concurrency
supplement the gate. Confirmation also verifies archived independently audited
screen PASS for the same candidate; publishing a result-only commit cannot alter
candidate/tool/dependency bytes. No implementation agent merges its own PR.

Archive all physical rounds, raw official recordings, private traces/endpoints,
commands,logs,source/JAR/runtime/package hashes,CPU cost and any failure. Stop on
first crash,timeout,skip,invalid metadata/hash/setup/endpoint/denominator and keep
partial evidence. Update improvements.md with measured budget/results,decision,
PR/report/raw links before completing each stage. Source/tests/whole-bot claims
remain separate from measured competitive outcomes; no improvement is presumed.

## Prospective numeric/runtime clarification before game outputs

Original mixture fallback is exactly N==0 OR abs(lead-center)<1degree, using the
existing arithmetic angular difference. It is not merely orientation==0. For
positive oriented support use the stated4/(4+N),2*n_i/(4+N) coefficients without
renormalizing/clamping. Floating validation allows abs(sumWeights-2)<=1e-12 and
per-wave score<=80+1e-9; this is validation tolerance, not a policy clamp or a
tunable margin. All coefficients,500kernel scale and target-centre distance stay
fixed before any game results.

Screen and confirmation may each use EITHER of the two already preallowed exact
Ubuntu image profiles. Their actual image revisions may differ; candidate/tool/
dependency/package bytes must not. This is a preregistered profile family, never
an unrecognized runtime fallback. Each stage revalidates and binds its own full
profile/boot/workspace before its one receipt; preserve both actual profiles.
No screen/confirmation pooling or causal claim is justified by the family. The
confirmation verifies screenPASS for the exact same candidate/artifact/tool
closure and the frozen profile family, not a dynamically approved third image.

## Prospective callback and skip-boundary clarification

Before acquisition, distinguish immediately visible own private skip markers from
observer-only skips: poll private markers while processes run; validate observer
skips at each completed game's reduction before starting another game. Observer
recordings do not provide a safe live polling endpoint. Any detected skip rejects
validity and stops the schedule; this changes no budget or success threshold.

The pinned SDK invokes public onRoundEnded before stopping the decision thread
and flushing queued terminal events. Enter a closing phase there; stop decisions
and accept terminal callbacks. Seal the prior round at the next public round-start
or final game-end after the prior round-end drain, under the same adapter monitor
as callbacks and decision calculations. Never hold that monitor across go/waits;
use a round epoch and interrupted-entry guard to exclude stale decision threads.
Keep sealing idempotent, accept queued terminal mutations while CLOSING, and reject
mutations once finally SEALED. The source audit establishes the ordered drain
boundary; do not claim protection against arbitrary injected stale callbacks or
externally cleared interrupts. A provisional
round-end snapshot cannot satisfy the harness's final completion contract.

## Prospective observer/private delivery cohorts

Before acquisition, pinned SDK source review identifies an observability boundary:
TurnToTickEventForBotMapper supplies no private tick when the owner's bot state is
absent from that observer tick. Official observer outgoing hits and damage remain
fully counted for competitive results. Private outgoing hit/damage counters must
instead reconcile exactly with the observer events in ticks that contain own bot
state. Retain the complement explicitly as observer-only outgoing hits/damage,
per round and game; never drop it from score, hit-rate or damage comparisons.
All own confirmed creations still reconcile with private fire counters. Same-tick
terminal events with own state present belong to the deliverable cohort and must
survive the final SDK drain. Incoming hit callbacks keep their existing strict
observer reconciliation. This is a prospective evidence-delivery definition,
not a waiver after seeing a discrepancy; no games have run. In 1v1 the engine
ends a round when alive count is at most one, limiting later posthumous ticks;
the source-defined cohort is retained without presuming such events occur.

## Zero-game preparation failure and prospective harness repair

Screen-label run41780412075 at head b3e50b07b19ab46c938b72fa24c20136425dfa92
failed before activation or any game. Its first profile collection succeeded;
the admission command then called the fresh-venv constructor a second time and
correctly refused the existing environment. The sole screen receipt was not
created. Preserve the original full ZIP20619a387320f7e88549b96404cf7b8a66d49cebd96ffdd170e97a1b44c20fc2
and log. Measured game budget remains0/25; there is no comparison result.

Repair only the collection/verification boundary: construct a fresh opponent
environment once per new hosted workspace, then strictly verify that same
prepared environment, actual runtime/package/source bytes and host identity at
admission, execution and before each game without reconstructing it. Preserve the
prepared-environment proof inside each acquisition and bind its hash into the
host/receipt and any archived-screen confirmation audit. Never reuse an ambient or
unbound environment. Review and refreeze the changed tooling/protocol before a
new job at run_attempt1; preserve the old head/freeze/failure. Do not rerun the
failed GitHub run, reset any receipt, change candidate JAR/source, alter game
population/budget/criteria, or open confirmation. The fixed screen receipt remains
once-only; any consumed physical round cannot be replaced. A zero-game hosted
check of the repaired repeated-verification path must precede screen clearance.
