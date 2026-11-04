# Facet gun/wave audit: one reproduced coherence defect

2026-11-04. [PR103](https://github.com/stephanelee/robocode/pull/103) · [issue101](https://github.com/stephanelee/robocode/issues/101) · [prospective protocol](facet-gun-wave-audit-protocol.md) · [reproduction tools](../tools/facet_gun_audit/README.md) · [full audit evidence](../experiments/Facet/evidence/gun-wave-audit/)

**Offline audit completed: one concrete prepared-shot/rating coherence issue reproduced; recommend one separately specified correction.** No gameplay code, parameters, bot JAR or model schema changed. Zero new battles, training runs or parameter searches. The valid100-round Diamond result remains1/100firstplaces and5.7386344431%equal-game share: reject the profile; no promotion or merge.

## Finding: the selector can rate a prediction the selected gun did not use

Facet prepares a next-shot angle from a forecast firing context. FireControl.Plan retains the chosen angle but not the complete immutable gun prediction set. A subsequent valid request creates a fresh shot context and recomputes all gun predictions from the new scan/stores. On confirmation these recomputed predictions become the common rating cohort. They need not equal the predictions used to prepare the real shot.

A standalone Java17 fixture against the actual measured JAR demonstrates:

| Item | Value |
| --- | ---: |
| Prior analytic plan passes FireControl.valid |true|
| Actual Core.decide approves fire intent |true|
| Prepared/current gun heading |0.647667310544012rad|
| Analytic angle placed in the rated shot cohort |0.5582312011396784rad|
| Difference |**5.12431160493861degrees**|
| Maximum allowed firing alignment tolerance |0.5degrees|

Own position400,300; target legal scan40:(600,300),northbound speed8, followed by scan41:(600,307),northbound speed7. The synthetic state contains16 independent typed toy observations in long and each recent store so eligibility is represented coherently. No acquired data was fitted or model shipped. The prior plan remains valid, and the real Core approves a firing request at the prepared heading, while rating the fresh counterfactual analytic angle. This is an offline intent/semantic test, not an engine bullet or a competitive result. Initial empty-view fixture/output are retained as a weaker first attempt; the populated-store fixture is authoritative.

The cohort remains internally common and is frozen before its outcome. The defect is the connection between prepared gun choice and the prediction later credited to it. Specification8.2's frozen-prediction/non-recomputation rule and3.3's prior-plan/current-heading execution need one coherent interface contract. Fresh counterfactual ratings are not automatically physical prepared-gun accuracy. Choosing to retain that counterfactual meaning would require an explicit design deviation and separate attribution, not silent equivalence.

Source locations: Core prepares/chooses predictions around248–294, recomputes request predictions around191–204, transfers them to confirmed outgoing waves around475–502; Learner.trainGun scores them before label insertion. Context changes are already quality-flagged: forecast acceleration0/missingmask15888 versus request acceleration1/3/missingmask14336. That difference is not itself an unflagged-lookahead defect. [Independent source review](../experiments/Facet/evidence/gun-wave-audit/independent-source-review.md).

## What the recorded traces establish

Exactly500000 own records from five capped100000-line traces were inspected, plus100 own metric rows. The prefixes contain33615 submitted decisions and797 confirmed shots of1704 in the complete measurement. They contain53 observed hit callbacks of104; unobserved later outcomes are not misses.

- All797 confirmations match the recorded actual request gun heading, request origin and next-tick creation, with separate native BulletFiredEvent records. No gross radians/degrees, old-origin or current-heading error was found in those covered confirmations.
-796 have an own bullet.motion record on the creation tick. The remaining game2/round1/bullet12 has a separately observed BulletHitBulletEvent on that same tick120; its motion snapshot does not survive the immediate contact. This is not797 independent swept-distance or speed replays. Creation/first-motion fields also contain bot-reported clock labels; full actual power/trajectory coordinates were not independently serialized.
- Covered shot attribution is223analytic/244long/330recent: selection is not permanently stuck on startup. These selected, nonrandom prefixes do not compare the guns' competitive strength.
-35 complete round-end snapshots contain equal analytic/long/recent cohort denominators. Ratings lie within their construction bounds. This checks common accounting, not equality of prepared and rated angles.
-21477 submitted decision frames have zero heat, a funded prior plan and gun alignment within0.5degrees;20680 do not submit fire. The registered800x600 field makes0.5degrees the actual capped plan tolerance. These counts do not prove every freshness, policy, pending or analytic-drift gate passed. Rejection reasons were not logged; this audit does not diagnose all hold causes or relax a gate.

[Aggregate analysis and bounded examples](../experiments/Facet/evidence/gun-wave-audit/trace-analysis.json), [independent trace review](../experiments/Facet/evidence/gun-wave-audit/independent-trace-review.md), [independent cross-check](../experiments/Facet/evidence/gun-wave-audit/independent-crosscheck.py), [same-tick contact excerpt](../experiments/Facet/evidence/gun-wave-audit/creation-tick-contact-excerpt.json).

## Limits and the recommended next change

The traces do not independently serialize every outgoing wave, per-gun prediction and learned label. Their caps prevent reconstructing the rest of100rounds. We therefore cannot quantify how often the5.12degree coherence issue occurred in that measurement, reproduce full label chronology from those traces, identify every firing hold, or attribute the1/100result or6.1%aggregate hit-callback/confirmed-shot ratio to this defect. No extra concrete gun/wave numerical or native-clock bug was established by this bounded pass. Existing incoming-attribution and broader live-contract limits remain.

**Recommend one correction:** attach the full immutable prepared prediction set, selected gun, firing context/version and angle to the plan; carry those same predictions through acceptance/confirmation and rate the prepared opportunity. Keep actual submitted/confirmed heading separately and validate compatibility. Do not silently substitute a new cohort after a scan/model update. Alternatively explicitly define and name a fresh-counterfactual selector contract and separate it from prepared-gun attribution; that changes the design claim rather than repairing the same contract.

Before implementing the preferred correction, specify and test accepted minor scan changes, new learned samples between preparation/request, common denominators, invalid-plan cancellation and selected prepared prediction versus actual heading tolerance. This audit does not implement the correction or authorize another battle/training/tuning run. A gameplay change needs its own prospective hypothesis, review and separately authorized measurement. Synthetic proof does not establish strength improvement.

## Reproduction and provenance

Both already-selectively-extracted and verified20-ZIP-part modes reproduce the same aggregate JSON. The Java17 populated-store fixture reproduces the5.1243116degree mismatch against pinned own JAR817a9d53ce944d27e3f536a92869eeca2e0aec20ef5c52b2dab882e931205792. Bot canonical source/inputdc5cc37c364b7ff76b122116cdc6a3d65a85abd102202a2f75f8add65b68ec5d and profile9ee52f73dc46e8503fe784d116e247fefc566dcb4e183215385840869d412a25 remain unchanged. [Manifest and outputs](../experiments/Facet/evidence/gun-wave-audit/audit-manifest.json).

Inputs were Facet source/JAR, supplied clean-room specification, frozen public engine/API contracts, legal own traces/metrics and public official results. No opponent logs/models/source, original Diamond implementation/startup data, other experimental gameplay helpers or own exported models were read. The archive reader has an exact eleven-member own-data allowlist and bounded sizes; output is aggregate diagnostics, not a training dataset. Independent source/trace/tool reviews were performed; [tool review](../experiments/Facet/evidence/gun-wave-audit/independent-tool-review.md).

The new CI workflow uses exactlyubuntu-26.04 and Java17 and fetches only the hash-pinned own historical JAR. It reproduces a historical defect, not a future candidate's correctness or strength. Production AISurvivor source/settings/JAR/checkpoints and all measured evidence/SHAs remain untouched.
