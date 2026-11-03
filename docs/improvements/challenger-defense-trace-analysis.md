# Existing PR63 defense trace: bounded exploratory analysis definition

Defined before new calculations on2026-11-03. No acquisition, games, fitting, controller change, source parameter search or validation gate. The five-round PR63 trace is already spent exploratory/development evidence. No unseen Diamond state, unobserved bullets or alternate trajectories will be reconstructed.

Source/archive: `benchmarks/challenger-diagnostic-v5` in robocode-improve-cycle1. Capture source80e039bbc69dbc206e3e3d3290e948abd088c046, execution65fc5df10260967c2711b966ecb6fe87636039fe. Sole trace is raw/diamond-diagnostic-v5-g01/diagnostic.jsonl.gz, decompressed SHA2562c42ba7b3ed7a761565ed65016354918749fccda7de0884ac746926c5008979d. The archived source/analysis-tooling analyzer and archived ChallengerWaves/WaveSurfMove define field semantics. Existing own-step parity is a prior finding, not to be reinterpreted as long-horizon or enemy-wave validation.

Question: does this observed defense failure primarily expose unavailable/ambiguous legal wave matching, absent/sparse profile coverage, or a limitation of the already recorded planner choices and danger assumptions? Seek one actionable source/design weakness, with explicit uncertainty.

Calculate once over all five rounds:

1. Verify trace digest, event counts, status/round totals and missingness. Join each enemyHit to exactly one inferred wave using(round,owner,creationTurn); reject ambiguous/missing identities rather than guess. Report matched/unmatched/ambiguous counts; distributions of recorded radialResidual in pixels and divided by that wave's bullet speed. Inferred waves without a hit remain unlabeled, not true dodges or false positive inferences.
2. For each hit, use the latest planner decision STRICTLY before the hit turn AND earlier in trace event sequence. Report missing decision, decision lag, whether its recorded selected first-two wave slots contain the matched wave (creation,origin,speed checked), and whether each selected wave uses a profile and its recorded support mass. The selected tuple has no owner: require unique origin/speed/creation identification within the round, otherwise mark ambiguous. Report selected-wave rank and missingness. Do not look ahead for a more favorable prior decision. Same-turn decisions can already contain hit-learned state and are excluded.
3. On the same hit-adjacent decision, report actual chosen option, all three stored finite/unsafe scores, recorded chosen/alternate first-contact/done/wall/last-evaluated times, recorded TTC and support. Describe actual hit delay versus recorded chosen contact window only as a held-policy comparison; replanning and unchosen paths prevent interpreting this as factual trajectory error. Do not claim that an alternate option would have survived. Across all recorded planner rows summarize chosen/profile/fallback counts, finite alternatives, chosen-risk/tie ranking and score margins. Scores are damage/time-weighted density heuristics, not calibrated hit probabilities.
4. For exactly joined actual hits, compute raw signedGF=wrapped(hit.directionâˆ’wave.referenceBearing)*wave.sign/degrees(asin(8/bulletSpeed)). Using wave.aimOwn's legal saved heading/speed, compute the source prior's mirrored linearGF=asin(abs(speed*sin(headingâˆ’referenceBearing))/bulletSpeed)/asin(8/bulletSpeed). Report actualGF and absolute distance to the nearest of the fixed head-onGF0/linearGF modes, plus source contexts; no fitted histogram/prior or post-outcome threshold selection. Missing/invalid source fields remain unknown. Approximate legal inferred origin/reference can still be wrong even with low radial residual.

Outputs: a compact standalone read-only Python analysis in this review folder, JSON with every hit-level join and summary, and a concise findings note. No current controller/analyzer files or pending external PRs will be edited. Source/script/plan/output hashes retained. Any concrete source bug gets a deterministic local fixture before a future separately registered repair; this analysis itself authorizes no candidate or competitive claim.

Pre-calculation reviewer clarification: a wave not yet inferred/observed at the pre-hit decision is unavailable, not a planner omission; an earlier inference alone does not prove it was still active. Fixed-mode distances are descriptive because the prior also includes uniform mass: distance does not imply zero risk or establish a primary cause of failure.


## Findings and limits

The exact calculation independently reproduced all101195 output bytes. All12891 retained events match the fixed decompressed trace hash; status records five complete rounds, zero drops and no recorder failure. **No new games or controller changes.**

| Recorded evidence | Result |
| --- | --- |
| Observed incoming hit to inferred wave |52/52 unique associations; maximum radial residual3.98e-13 pixels |
| Latest strictly earlier planner coverage |52/52 matched waves selected, always one turn before impact |
| Profile support at those decisions |37 use a profile;15 prior only; mean mass5.7482, maximum12.839625 |
| Recorded choice implementation |All3629 choices have the lowest stored finite risk;3313 have3 finite options and316 have2; no chosen wall veto |
| Last-decision contact window |All3 held policies report first contact at step1 for every hit-associated wave; this is annulus contact, not a counterfactual actual hit |
| Fixed-mode distance |Mean nearest head-on/linear peak distance0.29646GF; uniform prior mass still assigns danger away from these peaks |

This cohort does not expose an obvious missing hit-wave selection or wrong minimum-score dispatch. Prior PR63 own next-step parity also remains valid. Neither finding validates long option rollouts or the usefulness of the danger objective.

The actionable model limitation is **sparse opponent-specific evidence relative to the fixed prior**: source `WaveSurfMove.density` adds prior mass16 before usable learned mass. Even the most supported observed hit has only12.839625 learned mass, so the analytic prior contributes more than55% of its density;15 hits have no usable profile. This supports investigating the justification and usefulness of that prior/support assumption in a separately specified design. It does not prove a primary cause, imply zero risk away from the peaks, or show that reducing prior mass or trusting sparse observations would help. No parameters or alternative model were fitted here.

The198 inferred waves without observed hit labels remain unknown, not confirmed dodges or false inferences. Actual origins/references are legal estimates, not private opponent state. Alternate stored scores and contact windows describe unexecuted held policies; no alternate-survival conclusion follows. Per-decision bin arrays and swept footprint intervals were not recorded, so full danger calibration is unavailable. The original invalid PR65 cohort is untouched and is not part of this analysis.

[Immutable output, script, hashes and reproduction](../../benchmarks/challenger-defense-trace-analysis/README.md). No candidate or new measurement is authorized by this report.
