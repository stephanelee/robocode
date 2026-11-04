# Independent next-gun prior-art and evidence audit

Read-only source/document review, 2026-11-04. No fit, diagnostic, fixture, replay, game, constant selection or source change. Reviewed current Peak `MeridianPeak.java`, `CrossingGun.java` and passive journal accessors; Challenger `PressureGun.java`/`ContextGun.java`; guarded-ray-v6, outgoing-context-v7 and context-v2 reports; coherent-duel gun source/docs; off-mainline Meridian v5 and frozen-prior v6 docs/source; and Breakwater's inherited gun on `origin/experiment/diamond-breakwater-v1`.

## Main finding: the proposed controller mechanism already exists

Adding four ex-ante predictions to each common confirmed-shot cohort and allowing the learned ray only after it outperforms the three kinematic alternatives is direct prior art, not an orthogonal new selector. Challenger guarded-ray v6 / PR64 implements that exact conceptual guard. V7 / PR65 and Breakwater inherit it. Copying it into Peak is a different host/predictor implementation, but scientifically it is another application of a previously rejected mechanism.

PR64's protocol explicitly freezes all four headings before the pending shot is confirmed, irrespective of which model is selected; matches actual turn/target/power/origin; tracks all four from the same confirmed creation through consecutive fresh legal scans; completes the shared denominator only when all four resolve; and discards an unresolved/gapped cohort as unknown rather than a miss. `PressureGun.onFired`, `pairedFlights` and `score` implement this. `learnedQualified` compares equal four-way denominators against the best of the three legacy predictors. Its existing minimum/margin are historical facts, not suggested new constants. This is already prequential, common-cohort evidence with current learned-support validation.

That valid20-round PR64 screen was rejected:8.161933% share,1/20win; actual78/1176hits,185.2damage dealt/1441.392437taken; zero skips. The guard was active, with701confirmed learned shots,1093joint completed cohorts and endpoint virtual hits[97,81,80,117]. A failed whole-bot score does not prove the guard is intrinsically useless, but does preclude presenting it as an untested solution.

PR65 changed outgoing-wave context features while preserving that guard. Its20physical rounds were canonically INVALID because persistent paired totals531 differed from summed round completions530; descriptive2.856265%share/0wins is not a validated competitive comparison. The later lifecycle source probe and fix establish a structural callback/snapshot hazard, not a retroactive repair of that attempt. Retain both the invalidity and the guard's prior existence. Breakwater's inherited `PressureGun` still contains `pairedFlights`, equal shared counters and `learnedQualified`; a bot-name change does not make this mechanism new.

Other pertinent prior work:

- Challenger context-v2 / PR60 did unconditionally prefer a supported learned ray; its227passage samples and226legacy virtual cohorts had different definitions. It was rejected at20rounds/10.795607%share/0wins. PR64 was specifically motivated by this unsupported-comparison problem.
- Existing AIS `DuelGun` already stores four frozen headings on confirmed own shots and jointly resolves virtual-ray outcomes before updating attempts/hits. Its coordinated gun has an explicit current/GF versus head-on/linear/circular selector. Coherent-duel v2 passed its bundled Retcon test, but neither that result nor v1 isolates virtual-selection benefit; v1's actual accuracy fell sharply.
- Meridian v5 retained the v3 gun; its failed20-round continuation screen is not a new gun-selector test. Meridian frozen-prior v6 / PR87 replaced the online choice with a fixed206-example bank after retrospective proximity gains; its valid25-round fresh screen failed Diamond (12.80% versus14.53%control,29.93versus42.04outgoingdamage/100shots). It is a direct warning against promoting replay geometry into live strength.
- Retcon itself selects virtual-gun performance on legal scan-based cohorts. General expert selection and common virtual bullets are established techniques in this repository and opponents.

## The narrower correctness question in Peak is real

Peak/v3 chooses the highest of three exponentially smoothed kinematic skills as fallback, but *any finite supported CrossingGun estimate overrides it*. Its virtual trials are created only at fresh-scan turns divisible by4, whether or not a real shot is requested or confirmed, and all three angles are graded at the selected kinematic expert's forecast arrival turn. CrossingGun instead learns from confirmed-shot first-far-front crossings with bracket/support masks. Those two streams have different launch populations, label times and eligibility; their scores are not directly comparable.

A common-shot instrumentation audit could answer: **On the same actually confirmed launch and the same prospectively defined legal observation cohort, how do the frozen ex-ante CrossingGun and three kinematic headings compare, and what fraction of real selections has such comparable evidence available before selection?** This is a measurement/availability question, not a new gun controller, probability calibration or proof that another selector will win. Finite support is not predictive calibration. If outputs are only headings and geometric loss, call it prequential forecast evaluation rather than calibration of probabilities.

## What current Peak evidence can and cannot establish

The saved Peak journals preserve actual launch contexts, legal crossing labels/reasons and virtual-trial labels/angles, with cold and per-round bank/skill snapshots. The independent outcome audit replayed the outgoing bank updates and three skill updates. Those facts show legal learning and sealed-counter conservation; they do not establish a four-predictor head-to-head cohort.

Missing for a direct ex-ante claim:

- The exact CrossingGun forecast angle/support, model revision and all three kinematic angles/arrival times at EACH actual shot decision are not stored together. A CrossingGun launch stores context and reference bearing, not its predicted GF, bank/query support or frozen selected aim.
- The every-fourth-turn virtual trials do not generally match confirmed-shot identities, origins or creation times. Their chosen-expert due time is not a shared physical passage/segment-resolution time for all confirmed shots.
- Actual bullet heading cannot substitute for exact selected forecast: barrel-turn/alignment tolerance and timing permit a different realized heading. It also cannot reveal unselected ex-ante alternatives.
- Re-evaluating CrossingGun after its crossing label was admitted is leakage. Re-evaluating from a final192-example bank is both leakage and censoring older states. Chronological legal-bank replay may reconstruct an algorithmic query under source assumptions, but absent logged decision revision/callback ordering and query output it is not direct evidence of the actual ex-ante forecast. Label order alone must not be treated as a complete per-decision execution trace.
- Same future target coordinates are an observed-trajectory proxy under the actual fired bullet/controller. Different hypothetical aim/fire timing can change surfing response; proxy superiority is not a counterfactual live treatment effect. Known actual-bullet removal must not silently censor only one virtual alternative.
- Terminal/gapped/unmatched cohorts, unsupported learned queries and late callbacks need explicit masks and counts. Dropping difficult or unresolved rows after observing loss would bias the comparison. Cold-game boundaries and adaptive within-game dependence remain.

## Recommendation

Reject a fresh gameplay proposal framed merely as “add common four-predictor evidence/guard”; PR64 already did it, and both its valid failure and the later invalid cohort must be acknowledged. Do not choose another minimum/margin, recency factor, score or fallback using spent Peak outcomes.

If the user wants a measurement-first next step, the strongest concrete question is the common-shot forecast-availability/evaluation question above. Register a passive, source-parity-preserving observation protocol first, with exact ex-ante headings/query support/model revision, actual-request/confirmation identities, common legal grading time/segment convention, deterministic callback order, complete exclusions and final-drain readiness. Preserve all original behavior, including unconditional supported GF selection; do not select a new controller from the same measurement cohort. Any fresh collection still needs a separately bounded and independently reviewed budget, pinned environment and once-only receipt. The protocol should explicitly distinguish whether it is observing existing frozen behavior or introducing a new scorer; no automatic live promotion follows.

Alternatively, stop this family and inspect a different concrete correctness bottleneck before spending more games. Current evidence supports saying Peak's scoring streams are incomparable; it does not establish that its GF override caused the losses, that the fallback is stronger, or that copying a rejected guard will beat Diamond.

Disposition: no new selector, fit, constants, implementation or acquisition cleared. A passive measurement design is potentially useful but not yet registered, and does not rehabilitate any prior failed candidate.
