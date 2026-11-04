# Meridian future-wave support: prospective zero-game diagnostic

Written before implementation or matrix execution. Follows the rejected held-plan
comparison in PR92. This is a component experiment, not a gameplay promotion.

## Hypothesis and scope

Meridian.route scores a wave only when abs(r-F)<=22 at a future post-motion pose,
where F is that future sweep's far edge. A segment of length v=20-3p against a
radius18 circle has necessary radial support F-v-18 <= r <= F+18. The symmetric
window can omit a possible first future contact. Test this one support correction,
keeping Flight.orbit/step, HitAngles.risk,50-turn horizon, wall/switch costs,
current side and tie-breaking unchanged. No stopping option, speculative wave,
second-wave policy, gun, radar, power or learning change.

Keep the existing current-decision retirement F>r+36 unchanged. The next sweep
starts at current F, and next r cannot exceed current r+8. F>r+26 therefore
already prevents future contact; bullet speed exceeds8 thereafter. Current-sweep
support is not the same as future retirement. Audit retirement against FUTURE
contacts; do not incorrectly call it a low-power defect.

Use unchanged shared Flight/HitAngles/route source at4a02dfd (held candidate) and
verify the route method is identical to exactv3 at809b880. The held gun has failed
its competitive comparison and is not promoted or exercised here. Its unchanged
JAR7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe supplies
the actual source policy for parity checks; exactv3 JAR5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9
remains the prospective live baseline unless separately registered. No bot/server
or network game is started, and no production source/JAR changes.

## Frozen population

Arena800x600; radius18. Turns are relative to a legal cooled-down launch, not a
round start. Decision turn13, wave turn0. Thus current sweep is
[14v,15v], first future sweep [15v,16v]. Source and enemy scan stay stationary.
Generate the following2940 states as a full Cartesian product, with no search:

- powers:0.1,0.4,1,1.65,1.95,2.8,3;
- current radial error e=r-currentF: -(v+18)-1e-6, -(v+18)+1e-6,
  -22-1e-6,-22+1e-6,18-1e-6,18+1e-6,22+1e-6;
- body headings relative to radial bearing:0,90,180,270 degrees;
- signed speeds:-8,-3,0,3,8;
- placement: open/source(400,300),own=(400+r,300),bearing0;
  wall/own(780,300),source=(780-r,300),bearing0;
  corner/own(780,580),source=(780-r/sqrt(2),580-r/sqrt(2)),bearing45.

For each, evaluate both orbit signs through all50 ticks, and compute full route
scores/argmins for each committed side(-1,+1). Equality chooses-1 exactly as source.
This is2940 poses,5880 scoring contexts and5880 shared trajectories, not games.
Five hypothetical actual bullet headings are radial bearing plus{-30,-15,0,15,30}.
Those headings are oracle hypotheses only; never pass them to the risk scorer.

Add one frozen source-derived case: source(400,300), own(600.4,318.1), heading270,
speed8,power1.95,wave0,decision13, actual bullet heading0. Current sweep misses
because perpendicular separation18.1>18. The continuing first step is
(600.4,310.1); distance to next sweep is sqrt(11.85^2+10.1^2)<18, but old radial
error is below-22. The reversing first speed is3, giving y315.1 and a first-step
miss. This is an analytic possible geometry, not an observed Diamond state or a
computed full-route action result. Test it unchanged; do not tune it after output.
Both committed sides are included. Total2941 poses/5882 contexts/5882 trajectories,
294100 position rows, at most1470100 hypothetical segment/circle checks.

## Legal history, oracle and scorer

For each pose, reconstruct turns-1..13 by constant signed speed/body heading,
backwards from the current pose. This is a hypothetical legal motion history,
not enemy telemetry. Require centres in [18,782]x[18,582] at every history tick.
For each hypothetical bullet heading independently, replay sweeps at turns0..13;
if a previous/current contact or bullet-wall removal occurred, mark that ray
ineligible for FUTURE-contact comparisons, with reason. Preserve every state and
ray, including exclusions. Source parity is still evaluated for every current
legal state; report action changes in all states and separately eligible states.

The oracle must independently implement the pinned1.4.0 segment/post-motion-circle
and bullet-wall-removal order documented in meridian-platform.md. Fetch/verify the
exact pinned engine source as needed before source freeze; no alternate engine.
Advance actual ray segments until first contact or wall removal, then stop that
ray's truth stream. A future damage/energy response or changed next decision is
not simulated. Oracle contact checks use squared segment distance<=18^2; preserve
all geometry and any near-boundary cases, with floating-point disclosures.

Cold analytic HitAngles prior only. Wave centre is bearing to hypothetical own
pose at turn-1; lead uses the same launch-state lateral-speed/asine expression as
Meridian.onScannedBot. No trained histogram, hidden gun heading or oracle input.
Source enemy scan is stationary at the wave source, energy100-power (one legal
shot from100 energy, no later events), direction0,speed0,
omega0. Both policies use exactly this same model. Current-wave retirement36 and
all scalar risk weights stay fixed. New support is inclusive F-v-18<=r<=F+18;
old is abs(r-F)<=22. No per-wave normalization or risk retuning.

Independently reproduce the unmodified50-tick route scores in the helper and
compare bit-for-bit with actual private route method invocations for both sides
and both signs, using the unchanged source/JAR. Fail on any parity mismatch.
Mockito may instantiate/spy the existing bot as prior fixtures do; never invoke
start,run,go or create a socket/server. Main production classes stay untouched.

## Metrics, guards and decision

Save every state/ray eligibility reason, all50 poses per route, exact ray contacts,
old/new admission, old/new per-wave contribution/multiplicity, wall/switch costs,
full route scores and argmins. Report oracle first contacts omitted by each mask,
retired-wave future contacts, action changes, and selected-route contact exposure
on common eligible state/ray/committed-side cohorts. Equal weight each eligible
hypothetical ray/context, without choosing a favorable heading or subset.

Diagnostic passes only if source parity is exact, corrected support omits0 oracle
first contacts, unchanged retirement removes0 waves with a future contact, and the
frozen case reproduces its predeclared first-step contact/miss and old omission
through all50 future ticks. Invalid source/runtime/row counts stop and preserve all
partial evidence; no repair-and-rerun of the measured matrix under this registration.

To justify proposing a separately registered live candidate, additionally require
at least1 action change in an eligible context, strictly fewer total first-contact
exposures on common eligible cohorts and no increase in selected-route wall-contact
count. Otherwise retain geometry evidence only and reject advancement. These are
in-matrix feasibility guards, not generalization or competitive-strength criteria.
No live games or confirmation budget is opened by a fixture pass.

## Runtime, freeze and review

Budget:0 competitive rounds, one full frozen matrix plus the one stated analytic
case; one acquisition attempt. No parameter sweeps, outcome-selected subsets or
repeat. Implementation may compile before review but must not execute this matrix
or inspect its full-route choices before independent exact-payload clearance.
Freeze diagnostic source/runnable artifact, unchanged baseline source/JAR, all
classpath JAR hashes, engine-source oracle pins, commands and full runtime before
execution, in this document's companion freeze JSON and issue comment. An
independent agent reviews and then separately audits results. Implementation
agents do not self-merge.

Local runtime is actual Homebrew OpenJDK17.0.21.0.3+0,Gradle8.14.3,Python3.15.1,
macOS/Darwin27.2.0 arm64. Java executable SHA256
1e82f9fb8ac162eea33c3e5b36896458f0b58785e93effc20b78ed0d32f5e9da,
javac2868d8e548eec1a68c8caaf88e0a203992f002b47d2af260fd97326a44e054f7,
modules87215c58e1b671f093bd097b4a641b74d919091a77fc4d269e6ff7a6e065b31a.
Record full Python binary/version,Java release/JVM hashes and OS metadata before
freeze and require an exact match at execution. No Java25 or Retcon launch is
involved. Any GitHub Actions jobs remain pinned to ubuntu-26.04; automatic battle
jobs must exclude this zero-game branch. Standard Java17 source/JAR CI stays active.

Archive raw rows, summaries, all commands/logs, source/artifact hashes, exclusions,
runtime and every failure. Update improvements.md with measured budget,decision,
PR/report/raw links before marking complete. The original held-plan rejection and
all previous failed Diamond experiments remain unchanged and are not pooled.

## Prospective clarification before numerical outputs

The anchor assertion is: continuing firstfuturetick contacts; old window omits
that contact (and stays false through all50 projected ticks); corrected window
admits it. Reversing misses the FIRST future tick only. Its later truth is measured
and may contact; no50-tick avoidance result is assumed.

All2941 poses/5882 trajectories remain in output. Mark current-decision retirement
separately. Actual source parity and both policy scores use waves active AFTER
the unchanged current retirement36. Raw hypothetical window/contribution fields
may be reported separately but never enter policy scores for retired waves.

A forecast wall-constraint step is one whose unconstrained new-speed/old-heading
translation differs in x or y from the final clamped/interpolated Flight.step
position. Count every such step, including repeated outward constraints; a mere
edge pose without a constraint counts0. This is a forecast metric, not an actual
engine wall-event count. For the advancement wall guard, sum selected-route
constraint steps ONCE per(stateID,committedSide) having at least1 eligible actual
ray. Also report equal-eligible-ray-weighted counts, but those do not define the
guard. Contact exposure itself remains equally weighted per eligible ray/context.

Pinned bullet-wall removal acts on the UPDATED NEAR endpoint: x<=0,y<=0,x>=800
or y>=600 removes the bullet BEFORE the same-tick segment/circle check. A far
endpoint outside alone does not remove it. Capture pinned engine file hashes
with the final freeze. These clarifications precede all matrix/full-route outputs.

## Additive source-fidelity guard before any matrix outputs

Source inspection confirmed Meridian's existing Flight.step applies one wall
constraint pass, while pinned TurnProcessor applies a first wall outcome and
then a second constraint using the same pre-turn origin. This requirement is
already documented by the repository's dual-aim-v1 conformance correction. No
new numerical fixture was executed to discover this source-level issue.

Keep both scored policies and their argmins on the UNCHANGED one-pass forecast.
Do not silently repair motion in this mask candidate. At each of294100 stored
forecast transitions, independently apply the pinned single-bot motion rules,
including both wall-constraint passes, to the SAME incoming forecast pose and
SAME orbit command. Compare returned x,y,signedspeed and wrapped heading with
Flight.step using absolute tolerance1e-9. Save both results, errors, incoming
pose/command and any out-of-bounds input. No separate propagated engine rollout
or recomputed engine-orbit policy is introduced. This isolates local step
conformance; it does not model bot-to-bot collision, damage or enemy behavior.

Record motion mismatch rows/contexts and first divergence for each route as a
MEASURED baseline conformance outcome, not an artifact/source-parity invalidity.
Keep all frozen rows even when mismatches exist. Contact/exposure results are
explicitly conditional on the existing forecast and cannot then be treated as
engine-faithful future paths. In addition to every original advancement guard,
require ZERO motion mismatches for ANY proposal to advance this mask candidate
to live testing. A nonzero count vetoes advancement and motivates a separately
registered physics repair; it never justifies bundling a repair or rerunning here.

For each checked segment, retain both the independent closest-segment squared
distance<=324 oracle and the literal pinned isLineIntersectingCircle boolean.
Record any disagreement with full geometry; diagnostic validity requires0
classification disagreements. No epsilon/tolerance retuning follows outputs.
All additions are prospective, before source/artifact freeze or matrix execution.
The population,294100 position rows and1470100 maximum ray-segment checks remain
unchanged; add294100 same-input motion-reference comparisons.
