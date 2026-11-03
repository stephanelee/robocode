# Coherent duel v2: prospective whole-product comparison

Registered after the complete candidate build and before outcomes. This tests the [coherent gun revision](../coherent-duel-v2.md) as one product change; v1 and older observations are not pooled into the decision. The initial unreachable fallback-mismatch hypothesis was withdrawn, not patched or benchmarked.

## Frozen comparison

- Accepted baseline: `728f5a68c2494881f5658d1a80934ef2c97f98d1`; JAR `9daf9cf0e24f319900fee1c8869d50df4ffb004719b1380a8db40fb4961551f6`.
- Complete candidate: `eebb36e35f7ab9df566aa248b862a733d15384e9`; JAR `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`. All78 Java17 tests and five standard CI checks passed before this registration. Production Java, tests and bot assets stay frozen; later commits may add only protocol, runner, workflow, fixtures and evidence.
- Both versions use actual default TRAIN, duel-policy override unset, fresh independent absent models and seeded RNG per source/game. Every training round is scored. Network/checkpoint dimensions unchanged; no model shared or imported.
- Official Tank Royale1.4.0 runner and samples, unchanged Retcon0.2.0 source `28981c7a469ac064194b41001cd7183ec5ecf189`; Ubuntu26.04, Java17 fresh build/JAR parity and Java25 matches. Pin exact runtime, source and bundle hashes.

## Single fixed roster and budget

1. Retcon: three20-round games per source, seeds20262101/20262102/20262103; previous/candidate, candidate/previous, previous/candidate order. **120 physical rounds /120 AIS records**.
2. Candidate versus frozen accepted AISurvivor: three10-round self games. Candidate seeds20262111/20262112/20262113; accepted seeds20262211/20262212/20262213; alternate launch order. **30 physical rounds /60 AIS records**.
3. Official Target, SpinBot and Walls: three rounds per source/opponent, seed20262121; alternate arm order by opponent. **18 physical rounds /18 AIS records**.

Total **168 physical rounds /198 AIS records**,15 official games and18 independent model endpoints. Use the existing isolated per-bot harness. Bot directories must keep basename `AISurvivor` for official script discovery. Descriptor names become `AISurvivor-Candidate` / `AISurvivor-Accepted` solely to identify official results; JAR/source bytes remain pinned. Each launcher explicitly binds its own absolute model/metrics paths, seed and TRAIN mode and clears overrides. Record original/labelled descriptor and launcher hashes. No shared checkpoint, metrics file or global AIS environment.

Seeds initialize bot RNG, not arena starts; alternated games are not paired initial states. No component matches, extra screen/confirmation stage, performance early stop, replacement games, favorable rerun or outcome-dependent tuning. Preserve every partial/invalid attempt for independent review before any continuation.

## Locked decision: every gate is binding

Compute unrounded game shares from official whole-game scores; use equal-game Retcon block means.

- Retcon mean candidate-minus-baseline share **at least +5 percentage points**, at least2/3 strictly positive blocks, worst block at least-5 points, candidate total wins at least baseline minus3, candidate aggregate walls/1000 turns at most1.2 times baseline. Zero baseline wall rate requires zero candidate wall rate.
- Aggregate Retcon **hit rate strictly improves**, and **damage dealt per confirmed shot strictly improves**. Both versions require finite positive shot denominators; undefined metrics are inconclusive and cannot promote. These two observed mechanism guards do not isolate causality between GF prediction and finishing, or prove calibrated aiming confidence.
- Self-match candidate equal-game mean share **strictly above50%**, at least16/30 round wins, and at least2/3 game shares strictly above50%.
- Each sample candidate share loses at most5 percentage points versus baseline.
- All rounds and official identities validate, with zero AIS skipped turns in either version across the roster. Every independent cold model has no loaded checkpoint, sequential per-round counters1..20/10/3 and matching endpoint header/hash. Candidate forced-action telemetry validates: controller default, eligible equals forced, consistent selected/executed counts and positive forced use in each game.

All gates must pass for a **retain** decision, followed by independent source/raw-evidence review and required protected-branch checks. Failure means **reject**; there is no retrospective guard waiver. Retcon absolute scores/wins, per-block uncertainty, hit rate, damage/shot, shot volume, turns, damage taken, walls and sample/self outcomes are all reported. Three macroblocks do not establish80% power or broad generalization. Known game-order/guard noise is disclosed before outcomes and does not justify favorable reruns. No isolated GF, finishing or neural-learning benefit is claimed from this bundled test.

## Durable evidence

Preserve all198 AIS records,15 official arrays/logs, six native Retcon exports/capture hashes,18 independent cold checkpoint endpoints, exact source/JAR/runtime/launcher pins, immutable plan and layout, executed runner/workflow source, canonical unrounded decisions and every partial failure. Retcon private exports remain offline diagnostics and never enter the live controller.

## Measured result: all registered gates passed

[PR #43](https://github.com/stephanelee/robocode/pull/43), [single actual run41571217551](https://github.com/stephanelee/robocode/actions/runs/41571217551), execution `a180fb9373b25069e34e60a8287ef1e8b1b5d554`. All168 physical rounds/198 AIS records,15 official games, six native Retcon captures and18 independent cold endpoints validated. **Retain**: every registered performance, shooting, self-match, sample and safety gate passed. There were zero AIS skipped turns, no model loads, replacement games, performance reruns or outcome-dependent changes. [Original registration, source and raw evidence](../../benchmarks/coherent-duel-v2/README.md). Production/JAR remain exactly the frozeneebb/9679 candidate above.

| Retcon20-round block | Accepted v1 share / wins | Candidate v2 share / wins | Difference |
| --- | --- | --- | --- |
| 1 | 54.231771% /13 | 90.844233% /20 | +36.612462pp |
| 2 | 63.186441% /15 | 80.537313% /17 | +17.350873pp |
| 3 | 31.495017% /7 | 86.056888% /18 | +54.561871pp |
| Equal-game mean / total wins | 49.637743% /35 of60 | 85.812811% /55 of60 | +36.175069pp |

| Aggregate Retcon metric | Accepted v1 | Candidate v2 |
| --- | --- | --- |
| Confirmed shots / hits | 3748 /27 | 3659 /191 |
| Hit rate | 0.720384% | 5.220005% |
| Damage dealt / confirmed shot | 0.035928495 | 0.270210445 |
| Total damage dealt | 134.66 | 988.700019 |
| Damage taken | 731.76 | 420.7 |
| Recorded turns | 79402 | 70076 |
| Walls /1000 turns (collisions) | 0.062971 (5) | 0.057081 (4) |

Self-match v2 shares were99.585406%,99.140893%,98.870056% (mean99.198785%), with30/30 round wins. V2 scored100% against each sample; v1 scored100%/96.899225%/100% against Target/SpinBot/Walls. Both versions won all nine sample rounds.

This establishes a measured gain for the **combined** real GF/circular-candidate and finishing revision in this roster. It does not identify how much came from either change, virtual-gun selection, passage learning or interactions with the unchanged controller. Shot hit rate remains only5.22%; neither calibrated model confidence nor broad tournament strength is established. There is no offline learned predictor or private opponent input in live decisions.

The three candidate Retcon checkpoints contain class-GF passage masses1183/1252/1144, versus0/0/0 in v1. This confirms that the restored passage-learning path accumulated and persisted real legal observations; it is not evidence isolating the effect of that learning on wins.

All three v2 Retcon game shares exceeded50%. Descriptive one-sided95% Student-t(df2) lower bounds are77.117523% for its absolute equal-game mean and+4.802405pp for the three block differences; only three games per version were collected, and alternated starts are unpaired. The round-level Wilson interval81.931058–96.387954% is descriptive because within-game learning/adaptation makes rounds dependent. These uncertainty summaries do not replace the prospectively registered gates or establish80% design power/generalization.

### Activation provenance

The original ready-event activation `46b003ce542d0c345a90564c323fce732b52a283` was marked ready at2026-11-03T03:14:16Z, but no performance workflow/run appeared in the complete branch/workflow APIs. No performance games started. The cause is unknown; an indexing race was suspected, not demonstrated. Independent review cleared a **trigger-only** change from `ready_for_review` to `synchronize`, after another complete branch-run query confirmed zero performance attempts. That single push produced the sole measured run above. Its source, gates, seeds, steps and budget were unchanged. The [unstarted workflow, prelaunch API snapshot and provenance](../../benchmarks/coherent-duel-v2/activation/provenance.json) are preserved separately from performance data. The temporary workflow was removed before the final evidence push.
