# Independent melee activation review

2026-11-04. **The duel prerequisite has passed** the reviewed semantic, source/JAR and bounded live-adapter checks. This authorizes completing/reviewing the melee activation path. It does **not** approve flipping the existing flag unchanged: the currently gated integration contains functional defects. No gameplay files edited and no competitive games run in this review.

Reviewed baseline JAR73be04d7386a60474743a78fcebf3fd0c143e37120b410b18aa668bd5ed50c16. All findings below concern newly authored Facet only, plus supplied specification/official contracts. No prohibited inputs used.

## Required implementation corrections before activation

1. **Mode-correct firing policy.** FireControl.valid hardcodes `Energy.power(...,false)`. A legal melee plan power2.5 at range200 is always rejected by duel nominal2. Carry the plan's mode/policy identity and validate against the current compatible policy, including reserve, latest primary energy, and population transition. Pending accepted requests remain immutable. Independently reproduced by `revision4/MeleeActivationChecks.java`: `requiredMeleePower=2.5 currentPriorPlanAllows=false`.

2. **Melee immediate bot safety.** Melee.move still tests static enemy coordinates with a small distance buffer, rather than the corrected legal first-translation velocity image. Independent example with own(400,300),heading0,v8 and enemy(445,300),heading180,v8, plus a far second enemy: it chooses reversal, next real gap34, while STOP gives37safe; unsafeFallbacks stays0. Use the same independently written Facet first-step collision-risk contract as duel. For later forecast steps keep opponent commands uncertain with bounded legal envelopes; never present a static position as known next state. Retain eight actual own steps for selected candidates and explicit safest emergency if none is safe.

3. **Exclusive subsystem dispatch, quotas and fallback.** Core currently spends the full duel planner budget first, then runs melee movement outside movement timing/failure handling. Dispatch the correct movement stage once, within the shared deterministic budget and subsystem failure policy; preserve bounded emergency work. Record melee predictor/quota/unsafe fallbacks and include their time in movement latency. A disabled learned movement path must not silently rerun melee outside the guard.

4. **Identified projected primary and power.** Melee.aim returns only an angle, while Core chooses primary/power/kill-cap from its unrelated freshness/range target. Return immutable angle, identified primary, selected projected range and context metadata. Apply the specified melee power/range/energy policy to that primary, then validate the final solution under the selected legal power with a bounded deterministic resolution/fallback. Do not claim bullets exclusively target that primary; combined angular scoring still includes all reliable targets. Physical accounting and pending plans must retain the actual chosen primary.

5. **Survivor selector must control the actual aim.** Core overrides its selected gun prediction with `Melee.aim(...)` whenever beganMelee=true, bypassing the already corrected typed long/recent displacement gun predictions. In survivor duel, use the selected ensemble result; Learner.gun already selects the proper label convention for melee-origin data. Retain per-target displacement views/ratings, invalidate stale multi-target plans on transition, acquire a fresh survivor scan before treating it as current duel evidence, and keep prior melee provenance flags.

6. **Specified expensive-aim schedule.** Core currently runs full melee projections every turn. Implement alternate-turn expensive projections while more than four cooling turns from readiness, then every turn near readiness; keep plan expiry/target death/incompatible-scan rules. A bounded analytic cheap track can run between full projections. No policy tuning is requested.

7. **Activation configuration/descriptor/package consistency.** After reviewed corrected active-path tests pass, update both descriptor copies to include melee, immutable RuntimeConfig/property gate, package guard, README/matrix and tests consistently. Replace old assertions that deliberately require disabled melee with active semantic invariants; do not remove unrelated checks. Record activation as completion of the original independently chosen profile, not a measured strength improvement. Export schema/version must still match any new decision-state fields.

The `observedTogether` normalizer currently counts scans, not elapsed jointly observed turns. The risk formula's intended count should be explicit and match the specification's turnsObservedTogether. The alignment bearing convention is underspecified in prose; state the chosen native convention in the decision log and test it. These are semantic/model definitions, not invitations to search parameters.

## Required active-path offline acceptance

Use deterministic fixtures and the existing official contracts only; no original source/data. Require:

- More-than-one-enemy Core dispatch actually uses minimum-risk movement and combined displacement gun; legal2.5/1.5/1.0 policies submit/confirm on compatible fresh scans, with selected-primary kill cap and reserve.
- The reproduced avoidable collision rejects; both velocity signs, stopped opponents, unknown/stale opponents and all-unsafe destinations have defined bounded outcomes. Selected destination receives at least eight predictor steps unless an explicitly counted quota fallback occurs.
- Total predictor steps<=2400 and bounded80-sample projections,12-iteration displacement solve,49–50deduplicated destinations and existing memory/shadow bounds remain. Worst-case filled displacement stores, multiple waves and several live targets must expose truncation/fallback rather than throwing an uncaught boundary exception.
- Multi-target angle choice returns deterministic primary/range metadata, normalizes each target's evidence and respects gun-turn ties. A stale/dead primary cancels an unsubmitted plan, not a confirmed physical shot.
- Controlled population2->1 transition retains survivor model bytes/counts, preserves melee-origin labels, invalidates stale multi-target plan and uses the actual selected typed long/recent displacement gun after a fresh scan. Round reset retains models; a cold new game does not.
- Three consecutive melee movement/gun failures obey the same round disable/fallback policy. Terminal metrics/shot accounting remain exactly once. Existing10034 assertions/866 actual transitions and all existing counterexamples remain passing.

## Bounded own-probe live melee/transition diagnostic

Register a separate prospective semantic diagnostic before acquisition. No competitive melee screen is authorized by this review.

Budget: **two fresh cold games of two rounds each**, frozen corrected active Facet versus two separately named own stationary ContractPartner probes (three participants). Seeds49667/49668; alternate Facet's launch-order position. Official embedded runner800x600/.1cooling/100000us/ready10000000us/inactivity450, Java17 for this local semantic acquisition;240seconds per game,600seconds total. Empty private directories, exclusive attempt path, preserve every failure and stop without replacements/tuning.

Each probe has a distinct descriptor/trace path, uses only legal scans, remains stopped and fires minimum.1 no more than once/60turns. Both should be weak enough for Facet to eliminate one and expose survivor duel. Log round-start notification fields without querying own state before first tick; this avoids the prior probe's logging limitation. Actual own state begins with the first legal tick. No hidden/spectator state.

Require exact three-bot identities and complete two official rounds per game, two unique Facet rows per game, no numerical/subsystem/boundary failures or unattributed shots, and actual population2->1->0 progression with Facet alive. Require real confirmed Facet fire while population2, a retained survivor model before/after2->1, a fresh survivor scan and subsequent confirmed shot using the selected displacement ensemble path. Bound and record latency/predictor/shadow work, collision/skip/unsafe fallbacks. Independently reconcile probe hit IDs/damage and terminal deaths with Facet counters, including final hits. Diagnostic wins/scores establish no strength.

If a transition/active required behavior is not exercised, report that gap rather than a pass; do not automatically add replacement rounds. Existing actual skipped/rejected/multicontact cases can remain explicit fixture-only limitations. Successful semantic activation does not claim competitive melee performance; a future competitive melee screen still needs its own registered roster/budget.

## Acquisition infrastructure documentary disposition

Independently reviewed current `tools/facet_benchmark.py`, workflow restoration and draft freeze `/tmp/facet-publish/freeze-draft.json` SHA256f74b56c7e6c486739b5ac2fa290ec958a515ea4e31696e8776e17bf95a76fc98. File/source manifest validation passed17 frozen files and40 Facet source/input files; acquisition tests passed3 tests. Restore of both committed JAR copies after source/payload parity is present. Runtime checks enforce Temurin25.0.1+8,Python3.13.9,Ubuntu26.04/x86_64 before games; embedded official server hash is verified, per-game file/source pins and launch argv/state environment retained, opponent versions checked, cold processes and no retries enforced. Documentary infrastructure is acceptable for the requested descriptive300 rounds.

The draft correctly keeps premeasurementApproved=false. It pins the old duel-only artifact and must be regenerated after corrected melee activation/semantic review and CI. No existing approval silently transfers to changed source/JAR/config/descriptor bytes. Final prospective scope remains five cold20-round games per AISurvivor/Retcon/Diamond Port, descriptive only; no additional competitive melee games or confirmation automatically follow.
