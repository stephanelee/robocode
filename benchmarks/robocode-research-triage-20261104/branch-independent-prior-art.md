# MeridianBranch versus the already rejected Meridian v5

Independent source/text review only, 2026-11-04. No code implementation, replay, fixtures or games were run. Sources inspected: `origin/codex/meridian-motion-memory` at `b6843cba7e91f9e60839697592fff9cd5b1adf41`, its `experiments/Meridian/src/SurfPlan.java`, `HitAngles.java`, `Meridian.java`, `docs/improvements/meridian-motion-memory.md` and ledger, plus the proposed Branch text and frozen Peak scorer.

The initial Branch prior-art account was incomplete. Issue84 / PR85 directly tested the central idea: choose today's movement by continuing/reversing/stopping through the first incoming front and selecting a best continuation through the second, then execute only today's command and replan. It also proposed preserving an escape for the following shot. Branch cannot be described as introducing multi-wave continuation or solving a previously untested held-sign limitation. This omission must be corrected before registration.

## What the old implementation actually did

V5 collects actual inferred waves with positive `distanceLeft = own range - Flight.farEdge(...)`, then may append one speculative next-shot wave when fewer than two fronts remain and its legal cooldown estimator predicts an imminent opportunity. It sorts by initial distanceLeft/speed and retains at most two. First-leg modes are committed side, opposite side and STOP; exact ties retain enumeration precedence. It simulates forward until a front's far edge reaches/passes the predicted bot centre. If a second front remains, it recursively evaluates all three second-leg modes and uses the minimum. Only the first immediate mode is dispatched, and replanning occurs every tick.

The objective is crossing-time hit-angle risk multiplied by inferred damage/8, with half weight for a speculative front, plus chronological wall cost. It uses the old `Flight.step` forecast, infinite-line angular risk rather than finite swept segments, and no explicit second-leg flip penalty. Its per-leg horizon derives from arena diagonal / minimum bullet speed +3, rather than a shared50-tick full-path budget. It does not retain all actual waves' whole-path per-ray peaks.

The old live result is material negative evidence: four valid games/20 rounds, zero own skips. Diamond v3 control28.00%/1 of5 wins versus v5 12.08%/0 of5. Incoming damage/100enemy shots77.77→142.71. Retcon98.14% and AIS80.95%, each5 of5 wins. Zero confirmation rounds. The screen bundled continuation, speculative warnings, STOP and its danger objective, so it does not isolate which component caused the loss. Its enemy hit frequency was similar between arms while hit power rose, a pattern also present descriptively in Peak; neither decomposition establishes a cause.

## Real differences, and what they do not establish

| Dimension | Rejected v5 | Proposed Branch |
| --- | --- | --- |
| Continuation concept | First front then best second leg, replan next tick | First relevant inferred wave then one opposite-sign suffix, replan next tick |
| Mode family | CW/CCW/STOP on both legs | Two held signs and at most one reversal per sign, no STOP |
| Waves | Nearest two, potentially one speculative | Actual retained waves only; common legal target order; all waves still scored |
| Switch boundary | Far front reaches bot centre | Strict near front beyond centre+18, flip next tick; target/later-ray survival and held-suffix radial gates |
| Objective | One crossing risk, angular line kernel, damage weights/speculative discount | Frozen Peak finite-segment centre distance, per-ray whole-path maximum then sum, no damage weights |
| Motion/horizon | Original one-pass Flight, per-leg derived horizon | Peak route-only two-pass motion, shared50steps, at most198 new motion calls |
| Switch cost | First-mode change penalty | Same first-sign penalty plus2 for an actual future reversal |
| Evidence | Rejected20-round fresh screen | No implementation/results yet; Peak comparator itself rejected25rounds |

These differences are substantive enough that the proposed candidate would not literally rerun v5's code or repeat its exact treatment. They also mean it is not a controlled ablation of v5: objective, wave population, action family, switch boundary, physics, horizon and costs all differ. Relative to *frozen Peak*, held-score parity plus changing only continuation enumeration would be a narrow controlled software comparison, with trace instrumentation as an acknowledged additional runtime difference. It would answer whether this particular constrained continuation extension improves Peak under fixed fresh tests. It could not show that branching is novel, that removing speculation/STOP fixes v5, or that v5 failed specifically because of its geometry/objective.

## Recommendation

Keep registration and acquisition paused while the proposal is rewritten as a **constrained re-examination of v5's failed continuation idea on the Peak objective**, not a newly discovered missing capability. Preserve v5's exact source/result links and negative evidence prominently beside Peak's rejection. My earlier preliminary description of the mechanism as distinct should be read with this correction.

I would not authorize the45+conditional180 live budget solely because the implementation differences are numerous. The present evidence does not establish that a switch boundary after a near-front body clear is a useful improvement rather than another attractive forecast. The safest next prospective step is a small, explicitly bounded zero-game implementation-contract/feasibility diagnostic, with its own protocol and independent review, before deciding on fresh live acquisition. It should establish literal held-score/action parity, unchanged physical inputs, strict-clear/death aliases, full-path peak accounting, <=198 trajectory calls, bounded trace readiness, and at least one independently specified case where an eligible continuation changes the immediate first-sign choice after its future-flip penalty. Its outputs are geometry/activation evidence only and cannot demonstrate survival or strength; no parameter sweep, state/outcome cherry-picking or replay-based treatment-effect claim.

If parent instead chooses to pursue a live re-examination after registration, that is scientifically permissible as a distinct fixed candidate, but the report must state the repeated-mechanism history and weak prior rather than imply fresh novelty. Retain fresh contemporaneous Peak controls, the absolute >35%/win guards, unpooled equal-game damage estimator, separate fresh confirmation, no replacements and all failures. A failed v5 screen does not prove all narrower continuations fail, but it lowers the justification for repeatedly allocating games to the same explanatory story without a sharper test.

Disposition: no implementation/acquisition clearance. Resolve the prior-art framing first, retain the earlier arithmetic/trace clarifications, then request review of a newly registered concrete protocol. No changes to Peak source/freeze or existing evidence were made.
