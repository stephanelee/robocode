# Final frozen default-TRAIN confirmation

This is the final finite confirmation after [PR43's whole-product comparison](coherent-duel-v2.md). It changes no gameplay, model, component or default policy. Its outcome tests the accepted bot against pinned Retcon in fresh games; historical results are not pooled, and a failed confirmation does not undo the accepted A/B improvement.

## Frozen source, fresh population and budget

- Accepted source `94640028a1b5bf072713bc19ea1b608d14f9d26d`; JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`.
- Exactly ten fresh20-round Retcon games, two independent five-game batches, **200 physical/scored AIS rounds** total. No baseline A/B, self-match or sample repeat. The [registered layout](../../tools/fixtures/final-default-confirmation/games.json) fixes IDs `dc946-b1-g01` through `g05`, `dc946-b2-g01` through `g05`; AIS seeds20262301–20262305 and20262311–20262315. These IDs/seeds have not been used for a prior performance experiment.
- Actual default TRAIN, duel-policy override unset. Every game starts with an independent missing checkpoint, fresh bot/RNG process, private absolute model/metrics paths, no pretraining or shared model. All20 rounds of online training count toward the confirmation outcome. Preserve every model endpoint.
- Reuse unchanged `coherent_duel_v2_experiment.play` for per-participant launcher/config discovery, labelled official identity, default environment, cold counters, confirmed action mechanism, strict native stats and finite checkpoint validation. Its candidate JAR pin is the same frozen9679. Do not mutate historical modules' globals or decisions. The descriptor-only `AISurvivor-Candidate` label is a harness identity, not a new candidate.
- Pinned unchanged Retcon0.2.0 source `28981c7a469ac064194b41001cd7183ec5ecf189`; official Tank Royale1.4.0 runner. Ubuntu26.04, Java17 fresh build/tests/JAR parity, Java25 match runtime. Record exact assets/source/runtime/launch pins and execution SHA. Ports are isolated from interactive games.

## Binding confirmation rule

All conditions must pass using unrounded official whole-game score shares:

1. Equal-game mean share **strictly above50%** across all ten games.
2. **More than100 official round wins** out of200.
3. **More than five games** individually above50% share.
4. **Both** separate five-game batch means strictly above50%.
5. One-sided95% Student-t(df9) lower bound for the ten-game mean **strictly above50%**, using critical value `1.8331129326536335` and sample SD/sqrt(10).
6. All200 rounds/ten official arrays/ten native Retcon exports validate; zero AIS skipped turns; no loaded checkpoint; per-game counters1..20; each finite numeric model header equals20 with matching hash; positive/default forced-controller mechanism.

These are the complete prospective rules, independently cleared before collection. No performance early stopping, selective batch omission, replacement seed/game, rescue retry, outcome-dependent tuning or extra optimization stage. All declared games run unless acquisition is invalid; preserve any partial failure and seek independent review before continuation. An invalid or failed rule means **reliable beating not confirmed**, without retrospectively rejecting PR43. A pass completes this finite validation goal and ends the automatic optimization/benchmark loop.

Seeds initialize AIS RNG, not server starts; rounds adapt within games and are dependent. The t calculation uses games as macroblocks and its assumptions remain approximate with only ten games. Report all game shares, both batch results, SD/t lower bound and descriptive Wilson interval for round wins; the latter does not assume independent evidence for a superiority claim. A confirmation concerns this frozen default bot, pinned Retcon/runtime/arena and fresh population, not arbitrary opponents or all tournament conditions.

## Evidence

Preserve the immutable Git plan/layout bytes before matches, complete200 per-round records, ten official arrays/logs, ten native exports/hashes, ten independent20-round numeric model endpoints, labelled/original launcher and bundle pins, runtime provenance, both batch reports, original execution driver/workflow and one canonical aggregate decision. No private opponent data enters the live bot. Report every outcome and update `docs/improvements.md` with this PR/result/evidence. Independent review and required protected-branch CI precede merge; no gameplay/JAR changes are authorized by this experiment.

## Actual outcome: invalid acquisition; reliability not confirmed

[PR #44](https://github.com/stephanelee/robocode/pull/44), [single run41575285157](https://github.com/stephanelee/robocode/actions/runs/41575285157), execution `e631f152fd06468e541552a962d7256407fc1216`. The frozen946 source/9679 JAR and acquisition helpers were unchanged. Batch1 completed five games/100 rounds. Batch2 completed its first game, then its second game `dc946-b2-g02` recorded **one AIS skipped turn in round4**, with `checkpointLoaded=false`. The existing strict harness rejected that game after its20 rounds finished and stopped batch2; the aggregate job was correctly skipped.

The attempt therefore contains **140 physical rounds /140 raw records /seven official games /seven native exports /seven model endpoints**, with **120 strictly validated rounds** and **60 planned rounds unplayed**. All seven checkpoints have finite numeric state and header counter20. No loaded model, source substitution or performance retry occurred. The cause of the skipped turn is not established; it is not attributed to CI infrastructure or waived as harmless.

**Reliable beating is not confirmed by this attempt.** Completeness and zero-skip requirements failed. The complete-cohort performance rules and binding ten-game Student-t gate were not evaluated on this incomplete population. No complete confirmation decision was fabricated, no subset was promoted, and the skipped game was not replaced. PR43's accepted whole-product improvement remains intact. This closes the finite automatic benchmark sequence with an unconfirmed outcome; no additional tuning, rescue or benchmark is scheduled.

The following are every actually played game's descriptive official outcomes, including the invalid game. They are not a substitute for the registered confirmation:

| Game | Share | Official wins /20 | Acquisition |
| --- | --- | --- | --- |
| dc946-b1-g01 | 89.910667% | 20 | Valid |
| dc946-b1-g02 | 81.217985% | 17 | Valid |
| dc946-b1-g03 | 87.843137% | 19 | Valid |
| dc946-b1-g04 | 93.914917% | 20 | Valid |
| dc946-b1-g05 | 82.523759% | 18 | Valid |
| dc946-b2-g01 | 95.020325% | 20 | Valid |
| dc946-b2-g02 | 93.434874% | 20 | **Invalid: one skipped turn, round4** |

[Preserved original reports/raw/models and explicit invalid-attempt summary](../../benchmarks/final-default-confirmation/README.md). Both batch artifacts and the original registered layout/plan are retained. The one-shot workflow was removed before any further push. The round-level Wilson formula assumes IID binomial trials and would be descriptive only here; no interval or t gate is computed from this incomplete population.
