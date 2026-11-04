# Independent first active melee pass review

2026-11-04. Frozen snapshot `/tmp/facet-build/melee-first`, JAR SHA256 `2d96ebb335fe53fa4eb95e393d861cd56e88e0d6ae4bc70f09c368ba40910384`. Reviewer read only this new implementation, supplied specification/official contracts and independent fixtures. No gameplay edits, fresh engine matches, competitive games or restricted input access.

Independently recompiled test sources against the frozen JAR and reran **19 groups /10,110 assertions**, including866 approved actual transitions: passed. Log `melee-first/independent-suite.txt`. This does not override the quota failure below.

The six integration fixes are substantially present: mode-aware immutable firing policy/primary; exclusive measured movement dispatch; legal first-step enemy velocity envelopes; primary projection/range metadata and bounded power coherence; actual survivor analytic/long/recent selector path with typed displacement stores; specified hot/near-ready expensive-aim scheduling. Descriptor/property/package gates are consistent. No forced learned selection is warranted.

## Blocking follow-up

1. **Global80-sample limit is violated.** `select` can invoke three independent `aim` passes, each querying80 samples. Independently constructed one-target/80-label fixture takes two power stages, selects power1.5/projected range539.13, and queries160 displacement samples in one expensive decision. Source/output: `melee-first/MeleeBudgetCheck.java` and `melee-first/budget-reproduction.txt`. Enforce a single global80 query/sample budget across the complete decision, including all power-coherence stages. Reusing a frozen bounded cohort or allocating remaining budget is acceptable if query context/projection assumptions are explicit; do not claim a final-power nearest-neighbor query when the cohort was actually selected at a different context. The initial profile was not authorization to increase this hard limit to240. Count and test the actual total, including cycling/fallback cases and multiple targets.

2. **Disabled melee gun path must actually fall back.** Core's melee branch invokes `scheduledAim` without checking the gun-disable flag. After three consecutive gun errors, it can continue learned melee projection rather than its analytic fallback. Follow-up must test the real active Core path and show bounded legal analytic aiming/firing or a declared no-shot fallback on unavailable evidence. No learned projection should run while that subsystem is disabled for the round.

3. **Risk normalizer and observability.** `observedTogether` still increments per scan, not per jointly observed turn. Define elapsed observation semantics explicitly, count each legal turn once and retain game totals through rounds as intended. Add bounded actual mode/selected-gun/primary/retained-store telemetry so the registered live transition can be independently validated without inspecting private models or guessing the path from a final score. Strong active multi-target quota/failure fixtures are appropriate before acquisition.

No additional gameplay defect is asserted from this review of the frozen first pass. Follow-up changes must preserve all previous invariants and receive new artifact/source/CI pins; no approval transfers to changed bytes automatically.

## Prospective survivor diagnostic interpretation

Clarified before live acquisition: a stationary probe may naturally keep **analytic** selected because startup prediction is already accurate and ties retain the current gun. Accept actual survivor dispatch with beganMelee=true, retained displacement counts, a fresh survivor scan, selected gun identity and subsequent confirmed fire through the selected ensemble path. Typed long/recent predictions and common cohort counts must be logged when their unchanged eligibility threshold is reached. If that threshold is not reached, mark those live cases unexercised and retain independent active-selection fixture evidence. Do not force learned selection, change eligibility/margins, tune probes for a desired selector outcome, or add replacement rounds.

The registered own-three-bot diagnostic remains two cold games ×two rounds, no retries, and has no competitive strength interpretation. Final descriptive300-round freeze follows corrected melee review/diagnostic, not this preliminary pass.
