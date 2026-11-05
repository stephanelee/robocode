# Independent review of Facet revision 2

2026-11-04. **Decision: changes required; no premeasurement or merge approval.**

Reviewed stable `experiments/Facet/`, not the moving filtered third-pass workspace. JAR SHA256 independently checked: `378ebc23a589c8107eb381924b53736e19bf0a4247969257a301ad2a0434bec5`. Root's source manifest identifies sourceSHA256 `64060f9f54a8e0f1994fbfc18dbc6f31623a08063e413de2c6c21e73946ca5f8`. No Facet gameplay files edited. No restricted implementation, opponent dataset, archive, private model or source-derived expected decision inspected. No network or fresh matches used for this review.

The reviewer independently compiled the revision2 test sources against its committed JAR and ran the supplied suite under Java17.0.21.0.3/256MiB: **17 groups,7119 assertions passed**, including866 approved actual one-step trace transitions. That passes those fixtures, not the complete specification. Reviewer stress run:4,999,680 estimated store bytes,256 waves,20 planner iterations, p50~1.98ms/max~12.23ms; these are local offline timings, not configured-runner p99/skip acceptance. Full log: `revision2/independent-suite.txt`.

## Original findings status

| Finding | Revision2 status |
|---|---|
|1 exact physics|STOP/reversal/order and ordinary wall cases corrected and replayed; new unequal near-corner overshoot still fails actual full engine pipeline (R2-1 below).|
|2 certified shadows|Safely disabled; no unsupported runtime safety credit. Full shadow capability remains outstanding. Third pass is handling it.|
|3 quadrature overflow bound|Resolved for constructed learned density: bound uses kernel maxima and matching normalization. Arbitrary uncertified density safely produces positive fallback risk.|
|4 origin quality|Frozen quality/missing mask and multiplicative weighting implemented. New origin-time historical feature contamination remains (R2-2).|
|5 prior direction/support|Resolved: native CV, supported uniform prior and normalized truncated kernels with disclosed CDF approximation.|
|6 fire/heat phases|Request/creation/first-motion, current-heading origin, strict heat and creation heat corrected. Late-first-scan heat uncertainty remains (R2-4).|
|7 survivor displacement|Typed displacement branch added and avoids treating melee labels as GF. Full melee remains gated. Long/recent survivor implementations share one displacement view rather than distinct recent windows; not full ensemble acceptance.|
|8 full adapter review|Performed; terminal reconciliation and other blockers below remain.|
|9 fresh-scan starvation|Resolved for compatible fresh scans; new reserve/latest kill-cap revalidation remains (R2-3).|
|10 outgoing origin/cohort|Resolved for normal matched confirmations; request context/predictions retained, actual event origin and heading validated, phase clocks explicit.|
|11 same-batch contacts/visits|Association now follows all scan inference; exact origin and retained center enable visits. Some synthetic contact fixtures are physically inconsistent; do not claim they are actual engine chronology evidence.|
|12 denominators/forecast|Resolved raw denominator now `resolvedReal - resolvedContacts`, with `rawHits`; no double exclusion. Forecast uses high heat bound, but bootstrap interval/affordability/window origin require further work.|
|13 deduplication|HitBot includes victim/location/ram identity. HitWall has only wall identity; official engine emits one wall event per bot/turn, so distinct-side dedup is not an outstanding blocker on this profile.|
|14 continuation overlap|Prior second-wave union retained and integrated once, original contact clock preserved. Controlled action-change acceptance remains missing (suite prints actions1/1); third pass handling.|
|15 shared quotas/overflow|ThreadLocal accounts actual predictor calls; reserves provide bounded fallbacks. Real overflow uses explicit nontraining envelope. No empirical runner latency acceptance. New long-flight expiration blocker (R2-5).|
|16 state/telemetry/failure|Numeric maps/histograms,GC, schema validation, bounded export and disabled subsystem paths added. Hash omits decision counters, terminal rows duplicate, and second-game lifecycle remains unproven.|
|17 melee|Still gated; runtime cannot claim full specification.|
|18 exact URI guard|Resolved: lowercase official property precedence, exact validated explicit-port URI passed to constructor; no7654/default fallback. No-network guard tests pass.|
|19 rating cohort|Only confirmed physical opportunities now rate; common-cohort eligibility transition still inconsistent (R2-7).|
|20 Wilson contacts|Resolved: contacts counted once in resolvedReal, subtracted once in denominator.|
|21 wall-limited startup|Oblique single-wall no sliding corrected. Unequal corner crossing still wrong due single constraint (R2-1).|
|22 terminal drain/round identity|Open: final batch is not reconciled before snapshot, GameEnded snapshots same round again; third pass handling.|

## New reproducible blockers

**R2-1 — full near-corner engine pipeline is not reproduced (high).** Own `(781,580)`, heading45°, speed8,target8 on800x600: Facet returns `(783,582)`, outside legal x bound782. The actual compiled official private constraint returns `(783,582)` on its first application; the actual full engine pipeline applies the constraint again, producing `(782,581)`. Match the complete pipeline, including speed/wall outcomes, and test unequal near-corner overshoot in both signs/axes. Existing recorded exact-ray corner fixtures do not cover this. Analytic long projection must also stop at the valid first contact rather than returning out-of-bounds results. `revision2/CornerCheck.java`, `OfficialCornerCheck.java`, `corner-check.txt` and bytecode evidence reproduce this. A reviewer helper initially used800x800 instead of800x600; that invalid comparison was corrected and retracted, recorded in `corner-fixture-correction.md`. There is no source/runtime discrepancy claim.

**R2-2 — frozen origin features consume observations after origin (high).** `Core` updates ownContext to current turn before requesting incoming context for an earlier inferred origin; `Learner.context` reads mutable acceleration/timers, and `History.displacement(t,...)` uses `samples.lastEntry()` rather than state at t. Holding requested origin/body at turn40 fixed, adding only turn41 observation changes acceleration feature from.6667 to1 and displacement8 from0 to.125. Build feature values strictly from history no later than originTurn; missing past context remains flagged/halved. Retain last nonzero self orbit sign per source/enemy: current ownContext sign is relative to arena center and cannot serve as a stopped self's prior orbit sign about an arbitrary shooter. Test origin feature invariance under appended future observations and mirrored stopped-source contexts.

**R2-3 — stale plan can violate current energy policy (high).** `FireControl.valid` accepts a power2 shot with own energy2.1, violating the0.2 reserve. It also accepts power2 after latest target scan energy falls to.4 (kill cap now.1). Re-evaluate affordability/reserve and latest kill cap before committing a prior plan. If invalid, cancel and prepare a new plan; do not silently mutate a pending physical request. Reproductions print both unsafe accepts=true.

**R2-4 — delayed first scan falsely establishes exact cold heat (medium).** `Ledger.scan(40,98,.1,false)` yields heat interval[0,0]. A legal previously unseen power2 shot created at40 leaves heat1.4. Initialize heat as the interval of feasible unseen schedules whenever the first observation is after a possible firing window; zero startup cooldown alone does not exclude hidden shots. Use energy bounds conservatively and preserve origin/time uncertainty. Fixture should require interval containing feasible hidden-shot heat and no false certain forecast.

**R2-5 — unresolved real threat discarded by arbitrary512-turn age (medium).** On legal5000x5000 arena, origin(18,18),target(4982,4982),power3, originTime0: wave is still APPROACHING at513, but Core.resolve removes it due `turn > end+512`. Expire by proven physical clearance/death or conservative resolved envelope; retain unresolved origin/history references beyond base history capacity. Or explicitly reject unsupported arenas and document that narrowed rule profile, rather than silently dropping a real threat.

**R2-6 — avoidable immediate bot collision accepted (high).** With no credible waves, own(400,300),heading0,speed8 and fresh enemy(445,300),heading180°,speed8, planner chooses targetSpeed-8 (new own speed3). Continuing the enemy's legal observed command produces next gap34 (<36 collision), while STOP produces gap37 (safe). The no-wave branch checks walls only; surfing checks static last positions and no reachable displacement envelope. Test known-CV and bounded unknown-opponent hypotheses on the actual first translation; reject avoidable unsafe candidates and expose unavoidable emergency choice.

**R2-7 — gun means still compare different resolved cohorts (medium).** Analytic ratings accumulate before learned guns become eligible at16 reliable crossings; long/recent means begin only afterward. At the first learned selector comparison, analytic EMA includes earlier opportunity outcomes unavailable to challengers. Maintain a common comparison cohort (e.g. cohort-specific analytic mean/reset at eligibility transition) while retaining startup diagnostics separately. Test the eligibility transition, not only a hand-constructed wave already containing all three predictions.

**R2-8 — exported model hash omits decision-relevant learned state (medium).** Changing e.shots0->50 and rawHits0->10 leaves ModelSnapshot.write bytes identical, although this can change flattener activation. Serialize resolved rate counters and all decision-relevant learned selectors/context/schema state, with explicit schema version and deterministic order. A model hash should change when a learned state change can alter a command. No cross-game loader is requested.

These reproductions are independent no-network mathematical tests in `revision2/Revision2Checks.java` and `revision2/reproductions.txt`, executed against the frozen committed JAR. They are not performance claims.

## Other required scope/gates

- `f[14]=physical?0:1` silently substitutes a constant for specified nearest-known-fire-opportunity-distance/8 feature. Implement the specified feature, or record an explicit reviewed deviation before promotion.
- Wall availability currently stops its orbit simulation at bullet crossing and can report the full1.5H availability without checking the remaining horizon. Separate wall-horizon and escape-horizon accounting.
- Radar24-turn participant mismatch resets only its counter/sweep result; it does not clear stale acquisition bookkeeping as specified.
- Model/telemetry terminal identities, live adapter chronology, rejected-fire acquisition, skipped-turn reconciliation and bot-bot traces remain incomplete. Existing synthetic fixtures cannot replace those actual observations.
- Disabled shadows, missing controlled two-wave action-change case and gated melee are acknowledged incomplete work; no full-specification completion or competitive screen approval is appropriate.
- Configured-runner p99/skip/collision gates, prospective exact artifact/opponent/runtime pins, independent premeasurement approval and required CI remain prerequisites. No performance battles were run or approved by this reviewer.
