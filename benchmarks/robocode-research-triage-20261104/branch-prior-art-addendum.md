# MeridianBranch prior-art addendum: Meridian v5 already tested continuation search

2026-11-04. Read-only source/document/issue review, before registration or implementation. No bot/tool changes, games, fixtures, probes, alternate rollouts, fitting or parameter search. The Branch draft is paused. This addendum supersedes my earlier audit's recommendation to treat a two-leg continuation as a previously untested next mechanism. I missed direct prior art in issue84/PR85; the earlier audit's numerical Peak observations remain descriptive, but its prior-art coverage and implied research novelty were incomplete.

## Judgment

**Do not advance the current Branch draft directly to its proposed225-round maximum on the current rationale.** It is a materially different controller implementation, but its central hypothesis—choose an immediate movement using an optimized second movement after the first incoming wave—is already the explicit hypothesis of rejected Meridian v5. The draft mostly revisits that failed mechanism with different geometry, risk aggregation, action restrictions and engineering. Those differences make it a new candidate in a literal source/payload sense; they do not supply evidence that the reused mechanism is worthwhile or that a fresh large comparison is justified.

V5's whole-strategy failure does not prove continuation search is harmful or prohibit every future variant. Conversely, multiple plausible differences or defects do not rehabilitate it or establish that Branch would improve. The appropriate next step is a corrected research inventory and a separately justified mechanism decision, rather than registering Branch because multi-wave continuation was supposedly absent. No alternative candidate or acquisition is authorized by this note.

## Sources actually read

- [Issue84](https://github.com/stephanelee/robocode/issues/84), its complete body and [outcome comment6618775550](https://github.com/stephanelee/robocode/issues/84#issuecomment-6618775550).
- [PR85](https://github.com/stephanelee/robocode/pull/85) metadata/body: open, unmerged, head `b6843cba7e91f9e60839697592fff9cd5b1adf41` at this read. Open status does not erase its completed negative experiment.
- `git show origin/codex/meridian-motion-memory:experiments/Meridian/src/SurfPlan.java`, plus that branch's `Meridian.java`, `Flight.java` and `SurfPlanTest.java`.
- `git show origin/codex/meridian-motion-memory:docs/improvements/meridian-motion-memory.md` and its benchmark README.
- The executed source `fd7ad162ea598b944bac058fc18b48e7a4303dbf:experiments/Meridian/src/SurfPlan.java` has the **same** SHA256 as the reviewed branch file: `7c4080cd84fdbdaf6db6f1fcc22350b8b24f88fde6cce797c7c15cb62d1907b7`. The reviewed report SHA256 is `5d8645f3333ae3c5348cc0108989bbb9219ed11d3808ea0dbcda05a70b28c92d`.

No raw PR85 games were independently re-reduced in this addendum. Its measured figures below are the existing published report/issue/PR outcome, not a new independent audit or new observation. Peak's previously audited25 rounds are also spent and remain separate.

## The direct overlap is explicit in both registration and code

Issue84 registered: continuing/reversing/stopping through the nearest incoming front, then the best of those options through the second; replan each turn. The hypothesis explicitly combined earlier route choice with preserving escape on the following shot.

In `SurfPlan.choose` lines45–71, v5 gathers unpassed inferred fronts, optionally adds a speculative cooldown front, sorts by distance-left/speed, keeps two, and compares first modes `{side,-side,0}`. In `simulate` lines74–101 it advances the first mode until any selected front crosses the simulated centre, then recursively evaluates all three next modes for the remaining front and adds the minimum continuation risk. `Meridian.run` uses only the selected first command and repeats planning next turn. This is the same broad receding first-command/two-leg rationale that motivated Branch; it is not merely Challenger's held CW/CCW/STOP or Breakwater's peak aggregation.

Reported live outcome: all four cold games/20 rounds valid, zero own skips. V3 Diamond control28.00%/1of5 wins versus v5 Diamond12.08%/0of5; incoming uncapped damage per100 enemy shots77.77 versus142.71. Retcon and AIS were each5of5 wins, shares98.14%/80.95%. No conditional180-round confirmation. V5 used two fronts on2,807 Diamond turns and selected STOP on259; this was an active tested mechanism, not an unused code path. Speculative launch warnings106/134 matched next-turn creations (106/210 enemy launches), but those activation/timing counts did not establish strength.

## Exact differences: meaningful implementation distinctions, not improvement evidence

| Aspect | Executed v5 `SurfPlan` | Paused Branch draft |
|---|---|---|
| Wave set | At most two selected fronts; may add one hypothetical cooldown wave at half danger | No hypothetical waves; all retained inferred waves contribute risk, legal relevance gates branching only |
| First/continuation choices | Three first modes and best of three continuations, including immediate STOP | Two held signs plus at most one sign-flip continuation for each; no STOP |
| Branch boundary | First selected far edge reaches simulated centre; second mode applies on following simulated step | Strict target near radius exceeds simulated centre range+18; flip only next step |
| Target order | Distance-left/speed at current own state; whichever front crosses first on simulated path triggers recursion | Common insertion-ID-tied legal order; fixed target; per-first-sign clear time; alive-ray/held-suffix radial-encounter eligibility |
| Risk | Unchanged original infinite-line hit-angle risk, evaluated at each front's one crossing pose, damage/8 scaling; speculative factor0.5 | Frozen Peak finite-sweep centre distance, maximum over all50 steps per weighted ray, sum all retained waves; no new damage factor |
| Wall/time treatment | Wall terms only until selected fronts cross, with up to94 steps per recursion leg in800×600 | Full50-step wall cost for every plan; <=198 actual motion calls; no late-wall-only branch eligibility |
| Physics | Original one-pass `Flight.step` for own routes and gun | Peak's two-pass own route physics, original gun `Flight.step` retained |
| Switching/ties | Root cost2 for a different first mode; no recursive continuation-switch charge; root strict-min order prefers current side | Original first-sign cost2 plus real future-flip cost2; exact tie prefers no actual flip then first sign−1 |
| Lifecycle/evidence | Existing v5 markers/counters; test covers speculative readiness | Reviewed Peak two-phase lifecycle plus prospective bounded lossless decision/model/continuation trace |

These differences prevent treating Branch as a byte-identical rerun. They also change several causal factors at once. A future result could not isolate “branching works” from risk geometry, wave selection, speculative removal, stopping removal, switching cost, physics or timing merely by comparing against old v5 or Peak. Trace/readiness engineering improves reviewability, not competitive evidence.

## Source-level limitations and possible defects

1. **V5 crosses/removes fronts earlier than full body clearance.** `distanceLeft=range−farEdge`; choose filters at `>0`, and simulate scores a front when that expression becomes<=0. Thus a far edge at centre range qualifies even while the near edge has not passed `range+18`. The new continuation may start while that wave could still overlap the bot on a later sweep. This is a source-level consequence of the literal criterion, not proof that an observed v5 hit was caused by it. For example, at speed14, far=range leaves near=range−14, which plainly does not meet near>range+18. No fixture or game was run to measure an effect. This overlaps the already studied contact-window/mask family; it is not a fresh discovery justifying a repair benchmark by itself.
2. **Single crossing-pose/infinite-ray danger is a different approximation from finite swept contact.** V5 calls `danger.risk` only at the centre crossing and has no per-ray wall-death mechanism. Body-footprint/sweep timing and earlier/later exposures are not integrated there. The paused Branch uses Peak's existing finite swept maximum, but Peak itself failed; replacing v5's approximation is not established to fix its losses.
3. **The speculative front changes both anticipation and the chosen pair.** V5 inserts a hypothetical wave when fewer than two inferred fronts are available and cooldown is imminent, then selects the nearest two. A speculative estimate can influence ranking even without a real shot. Its reported timing precision is incomplete and situation-dependent. Removing it would be an ablation/design change, not evidence that speculation was the primary failure.
4. **Future action changes are free in v5's continuation value.** Root switching incurs2, but recursive `simulate(...next...)` adds no transition charge. Branch's additional future2 is a real policy difference, not merely exact bookkeeping. There is no measured evidence here selecting that penalty or proving free continuation caused v5's result.
5. **V5's own route uses the known original one-pass corner approximation.** Its `Flight.step` is the same class of original Meridian gun/route forecast; Branch inherits Peak's route-only second pass. The known corner fixture characterizes a numerical difference, not its frequency or competitive impact in v5. Keep the old cohort/source intact.
6. **Existing tests/counters do not identify the loss mechanism.** The viewed SurfPlan test checks cooldown/speculative insertion and leaves original inferred waves untouched. It does not establish a per-decision wave/body-clearance join, continuation quality or alternate survival. A limited test does not invalidate the completed zero-skip gameplay cohort. No crash, corrupted run or competitive-result invalidity was demonstrated by this read-only audit.

## Evidence limits and disposition

V5 and Peak each bundled changes and used uncontrolled fresh starts with within-game learning. V5's Diamond incoming hit frequencies were close across arms (17.89% control/17.14% v5), while mean power on a hit rose0.90→1.70. Peak also has a hit-rate-versus-damage composition distinction. That resemblance is descriptive, not a common diagnosed cause or permission to change the registered damage metric.

The rejected v5 experiment is direct negative prior evidence against the broad motivation, despite confounding; it must appear prominently in any future Branch-related registration. It cannot be dismissed because PR85 remains unmerged or because its other simultaneous mechanisms differed. Conversely its failure cannot prove that the paused exact four-plan variant fails without testing. The justified conclusion now is **insufficient mechanism-specific justification to prioritize this large live follow-up**, not a claim about an unimplemented bot's strength.

Keep `/private/tmp/meridian-branch-proposal.md` as a paused, unregistered draft, with this addendum attached to any later discussion. The six engineering-review clarifications were written before this pause (current proposal SHA256 `3dc0a3486d01b62481622b6dd5b152087f077b934d1fec58a7d5b16c54209562`); they neither cure the missing prior art nor grant implementation/benchmark clearance. No registration, source changes or acquisition should proceed on that draft's former rationale. Await root's research-direction decision; do not silently substitute another candidate, retrofit the old study or rerun a failed mechanism.
