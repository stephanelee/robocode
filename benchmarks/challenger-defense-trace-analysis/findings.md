# Spent PR63 defense trace: completed bounded exploratory analysis

The retained legal trace does not expose a missing inferred-hit wave or a wrong minimum-score selection in this cohort. It identifies a source constraint worth investigating separately: the fixed analytic prior dominates the sparse learned danger at every observed hit. Whether that constraint causes poor defense remains a hypothesis; this is not proof of a primary cause or a winning replacement.

All12891 records/3876266 original bytes reproduce the committed SHA256. Status reports five rounds, zero drops and no failure. Capture source80e039bbc69dbc206e3e3d3290e948abd088c046/run41686116538 remains unchanged. These observations are spent exploratory/development data, not fresh validation.

- All52 recorded incoming hits uniquely join a legal inferred wave by round/owner/creation. Recorded radial residual is0 to3.9790e-13 pixels. The other198 inferred waves have no observed hit label; their outcomes remain unknown. A radial match does not independently prove the inferred firing origin, reference, target or unobserved bullet lifecycle.
- All52 hit-associated waves occur in slot0 of the latest strictly earlier planner decision, exactly one turn before the hit, with earlier trace event sequence. None is absent, ambiguous or not-yet-inferred in that decision. Thirty-seven use a hit profile and15 use prior only. Their support mass averages5.7482, maximum12.839625.
- Every one of3629 recorded chosen scores is the finite minimum of the three scores.3313 decisions have all three options finite;316 have two finite options. No chosen rollout has a recorded wall veto. These facts check the logged selection path; they do not establish good defense or optimal movement.
- At each of the52 immediately preceding decisions, **all three** held policies predict first contact with that matched wave at step1. Chosen-wave completion is step2–5. The executed choices are CW18/STOP14/CCW20. An annulus contacting a possible body position is not the actual enemy ray hitting it, and alternate contact summaries do not show that an alternate would survive. Earlier decisions and replanning matter; this result provides no causal winner among the options.
- Using the saved legal aim-time own state and matched-wave reference, observed hitGF has mean0.2241, range[-0.4828,0.6896]. Distance from the nearest fixed head-onGF0/linearGF peak averages0.29646 (median0.28934). The prior includes uniform mass, so this does not mean the observed rays had zero assigned danger. Full per-decision bins and swept footprint intervals were not recorded and are not reconstructed here.

The archived `WaveSurfMove.density` always adds prior mass16 (8uniform,4head-on,4linear) before adding the matched context histogram. At observed hits the available learned mass never exceeds12.839625, so the fixed prior remains more than55% of the normalized density even in the most supported observed case;15 hits have no usable learned profile at all. Four coarse range/lateral contexts and hit-only updates supply little opponent-specific evidence. The actual hit directions are not concentrated on either analytic peak in these summaries. These source/data facts justify investigating the danger model's prior/support assumptions before another blind movement change. They do **not** show that reducing a prior, adding contexts, switching an option, or trusting sparse hits would improve live outcomes. No threshold, alternative model or gameplay change was fitted or executed.

Own next-step motion parity was already established by PR63 and is not recalculated here. It validates neither90-step held-policy trajectories nor inferred enemy waves/danger quality. The chosen-hit score averages0.5210 versus an all-decision average0.1828, which is compatible with the heuristic detecting some increased danger; those units are not calibrated hit probabilities and do not establish predictive skill from a hit-selected subset.

One actionable follow-up question for a separately reviewed design is whether a danger model with justified opponent-specific support can preserve safe motion while addressing these prior assumptions. This analysis itself authorizes no implementation, capture, parameter search, repeat or competitive claim; it does not overlap the external reachable-path gun or shielding experiments.

Reproduce from this folder with Python3.12+:

```text
python analyze_defense.py <repo>/benchmarks/challenger-diagnostic-v5/raw/diamond-diagnostic-v5-g01/diagnostic.jsonl.gz defense-analysis.json
```

`analysis-plan.md` and the matching Git-published definition were frozen before calculations. `defense-analysis.json` retains all52 hit-level strict joins, recorded option risks/timings/support, missingness and summary families. The script reads only the existing lossless trace/status and its local plan; it edits no controller or existing analyzer.
