# Retained-recording aiming diagnostic

**Offline diagnostic only; no promotion or gameplay change.** All eight candidate melee games from rejected learned-gun 0.5 are analyzed. Zero new games/rounds. All 10,601 captured shot traces match physical fired bullets and previous-own-pose origins; 9,969 have observed range crossings and 632 are censored (target disappearance/round end).

| Intended opponent | Crossing shots | Actual mean error | Learned desired mean error | Circular reference mean error |
| --- | ---: | ---: | ---: | ---: |
| Keel | 4,943 | 9.88° | 9.97° | 11.81° |
| Prism | 2,915 | 8.97° | 8.89° | 9.31° |
| Diamond Port | 2,111 | 10.05° | 9.97° | 10.07° |

The proxy shows a modest overall angular reduction (desired 9.65° versus circular 10.71°), concentrated against Keel (9.97° versus 11.81°). Diamond Port differs little (9.97° versus 10.07°). These are conditional errors on trajectories responding to the actual shot, not counterfactual scores, exact collision tests, or hit probabilities. They do not establish why a competitive score changed or overturn the registered learned-gun rejection.

7,987 observed crossings were at scan ranges ≥500; 1,910 at 250–500; 72 below 250. Recorded scan ages were 0–3 for 5,293, 4–7 for 4,646, and 8–12 for 30. Availability is reconstructed through fireturn−1; callback completion before policy evaluation is not established. Censored 632 shots are not assigned zero error. Results are exploratory and will not tune the fixed movement candidate or count as fresh validation.

[Protocol](keelhelm-aim-diagnostic-protocol.md), [summary with all bins](../../benchmarks/keelhelm-aim-diagnostic/summary.json), [input hashes](../../benchmarks/keelhelm-aim-diagnostic/inputs.json), [per-shot archive proof](../../benchmarks/keelhelm-aim-diagnostic/artifact-proof.json), [original rejected study](keelhelm-learned-gun-results.md), [PR #133](https://github.com/stephanelee/robocode/pull/133).
