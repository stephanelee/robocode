# Retained-recording aiming diagnostic

**Offline diagnostic only; no promotion or gameplay change.** Corrected explicit-death censoring supersedes the first published summary. All eight candidate melee games from rejected learned-gun 0.5 are analyzed: zero new games/rounds. All **10,601** captured traces match physical fired bullets and previous-own-pose origins; **9,952** range crossings are observed and **649** censored.

| Intended opponent | Crossing shots | Actual mean error | Learned desired mean error | Circular reference mean error |
| --- | ---: | ---: | ---: | ---: |
| Keel | 4,935 | 9.87° | 9.96° | 11.80° |
| Prism | 2,909 | 8.98° | 8.90° | 9.32° |
| Diamond Port | 2,108 | 10.05° | 9.97° | 10.07° |

The conditional proxy shows a modest overall angular reduction (desired 9.65° versus circular 10.71°), concentrated against Keel. Diamond Port differs little. This compares bearings on trajectories responding to the actual shot, not counterfactual scores, exact collision tests or hit probabilities. It does not establish why competitive scores changed, overturn the learned-gun rejection, or tune the frozen movement candidate.

Observed scan ranges: 7,972 crossings at ≥500, 1,908 at 250–500, 72 below 250. Scan ages: 5,284 at 0–3, 4,638 at 4–7, 30 at 8–12. These reconstruct recorded own-scan availability through fireturn−1; callback completion before policy evaluation is not established. Censored shots receive no invented zero error.

## Correction history

V1 counted 17 crossings on retained target states at their explicit BotDeathEvent turn. Independent review found this violated the registered death-censor rule. The original V1 archive and public data remain preserved; its 9,969/632 counts are superseded. An unused V2 correction added energy≤0 as a cutoff, over-censoring six living-target cases (some subsequently regained energy). Zero energy alone is not death. That failed assertion/source/log and V2 derived outputs are retained. V3 applies explicit death-event turn or disappearance only, restoring those six and yielding 9,952/649.

Independent raw recomputation checks all physical origins/powers/headings, scan contexts, geometric errors, grouped means/medians and 36 original/recovered input hashes. Exactly 17 removed keys have an explicit death at/before crossing; zero retained crossings occur at/after death. All surviving V1 crossing times/errors remain unchanged. No gameplay score, policy or input was changed.

Corrected full per-shot/source/auditor archive (including unused V2): `robocode-keelhelm-calibration/2026-11-06/aim-diagnostic-corrected.tar.gz`, SHA256 `8ab818175c9c2e639f76ae147cbe8b8bcd6524ef3b6ccea2ba6f1783b7eec45c` (1,771,742 bytes). Original V1 archive SHA256 `0ba1b0bec5289ee91a8ede0a34d14199c548736253500adb184718d79b4b0d5e`; original game corpus remains in `robocode-keelhelm-displacement/2026-11-05/full-artifacts.tar.gz`, SHA256 `7c301b68229eaab957571416d2c7010722c25a83d24ccab7125e74e9ffaf7353`.

[Protocol](keelhelm-aim-diagnostic-protocol.md), [corrected summary](../../benchmarks/keelhelm-aim-diagnostic/summary.json), [independent review](../../benchmarks/keelhelm-aim-diagnostic/independent-review.json), [input hashes](../../benchmarks/keelhelm-aim-diagnostic/inputs.json), [superseded V1](../../benchmarks/keelhelm-aim-diagnostic/superseded-v1), [artifact proof](../../benchmarks/keelhelm-aim-diagnostic/artifact-proof.json), [original rejected study](keelhelm-learned-gun-results.md), [PR #133](https://github.com/stephanelee/robocode/pull/133).
