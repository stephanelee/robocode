# Coherent default duel bot: one whole-product comparison

This registration follows the complete build, before integrated performance collection. It tests the [whole composition](../coherent-duel.md), not isolated gun/radar/timing benefits. Prior PR #41's screen is supporting history only and is not pooled into this decision.

## Frozen production and runtime

- Accepted production baseline: `7104b65d03298caceb010a5bfd085bda65092d17`; JAR SHA256 `d047bf99325d2ad73aedd87eac6a4baa1677170afff0c86b948f03ea93390e1d`.
- Complete candidate: `9c79e4e26b3323704df5dad467af182e0b48fafd`; JAR SHA256 `9daf9cf0e24f319900fee1c8869d50df4ffb004719b1380a8db40fb4961551f6`. Java17 clean build and all73 tests passed. Exact independent source review and runner/workflow clearance precede collection. Later commits may add only protocol, runner, workflow, fixtures and evidence; production Java, tests and bot assets remain identical to this source.
- Actual default TRAIN mode in both arms, `AISURVIVOR_DUEL_POLICY` unset. No pretrained/checkpoint sharing. Independently cold model/RNG state per bot/game; every online-training round counts toward the objective.
- Official Tank Royale1.4.0 runner/sample assets, pinned unchanged Retcon0.2.0 source `28981c7a469ac064194b41001cd7183ec5ecf189`. Ubuntu26.04, Java17 fresh build/JAR parity, Java25 match runtime. Record exact source/artifact/runtime hashes.

## One fixed budget

1. Retcon: three independent20-round games per source, seeds20261901/20261902/20261903. Alternate previous/candidate, candidate/previous, previous/candidate arm order. **120 physical rounds,120 AIS metric rows.**
2. Self-match: candidate versus frozen accepted AISurvivor, three10-round games. Candidate seeds20261911/20261912/20261913; accepted seeds20262011/20262012/20262013. Alternate launch order across games. **30 physical rounds,60 AIS metric rows.** Descriptor names become `AISurvivor-Candidate` and `AISurvivor-Accepted` solely to map official identities; JAR/source contents are unchanged. Each copied launcher explicitly sets its own absolute model/metrics paths, seed and TRAIN mode and clears the duel-policy override. No global shared AIS checkpoint/metrics configuration. Original and labelled descriptor/launcher hashes are recorded.
3. Samples: official Target, SpinBot and Walls, three rounds per source/opponent, seed20261921. Alternate arm order by opponent. **18 physical rounds,18 AIS metric rows.**

Total **168 physical rounds /198 AIS metric rows**, exactly once. Seeds control bot RNG, not arena initial state; alternating order does not create paired arena starts. No extra screening or automatic confirmation stage. No performance early stop, favorable rerun, replacement seed or omitted failure. Invalid/partial acquisition is preserved and independently reviewed before any continuation.

## Prospective decision

Compute shares from each whole official game and compare equal-game Retcon block means, without rounding. All conditions must pass:

- Retcon mean candidate-minus-baseline share at least +5 percentage points; at least two of three deltas strictly positive; worst delta at least -5 points; total candidate wins at least baseline minus three; aggregate candidate walls/1000 turns at most1.2 times baseline (zero baseline requires zero candidate).
- Self-match candidate equal-game mean share strictly above50%, at least16 of30 round wins, and at least two of three game shares strictly above50%.
- Each sample candidate share loses at most five points against baseline.
- All requested rounds, both participants' official identities, and complete telemetry validate. Both AIS versions have zero skipped turns across the roster. Independent fresh model counters run1..20,1..10 or1..3; each preserved endpoint has matching final header/hash. No loaded checkpoint.
- Candidate forced-action telemetry validates in every game: controller policy, eligible equals forced, selected/executed action counts consistent and positive forced use. Self-match metrics are validated separately for each labelled participant.

A pass supports retention of this bundled product change in this roster, subject to independent full-source/raw-evidence review and required CI. Retcon absolute shares/wins and block uncertainty are reported separately; improvement against a weaker baseline alone is not proof of beating Retcon. Three Retcon macroblocks do not establish80% power or broad generalization. A failure does not justify retrospectively attributing effects to one component or changing gates; preserve the coherent candidate and all outcomes for an explicit review decision.

## Durable evidence

Preserve all198 per-bot round records,15 official result arrays (six Retcon, three self, six samples), six native Retcon exports/capture hashes,18 independently cold model endpoints (six Retcon, six self, six sample), source/bundle/labelled-launcher/runtime pins, immutable registration and runner/workflow source, per-game aggregates, executed-action mechanism, exact unrounded gate report and every partial failure. External Retcon private exports are offline diagnostics only and never enter the live controller.

## Measured outcome: registered retention gates passed

[PR #42](https://github.com/stephanelee/robocode/pull/42), [single actual run41564742819](https://github.com/stephanelee/robocode/actions/runs/41564742819), execution `addae500040fe8507afc654f00ee3dd229f4dace`. The complete168 physical rounds/198 AIS records,15 official games, six native Retcon exports and18 independent cold checkpoints validated. Every registered condition passed; the canonical decision is **retain**, subject to independent review and merge. There were no replacements, extra performance attempts or skipped AIS turns. Production remains exactly the frozen9c79 source/9daf JAR above. [Original registration, raw evidence, endpoints and exact decision](../../benchmarks/coherent-duel-v1/README.md).

| Retcon20-round block | Previous share / wins | Candidate share / wins | Share difference |
| --- | --- | --- | --- |
| 1 | 15.187850% /0 | 53.319502% /12 | +38.131652pp |
| 2 | 16.326531% /1 | 55.278174% /12 | +38.951643pp |
| 3 | 11.101137% /0 | 66.551486% /15 | +55.450349pp |
| Equal-game mean / total wins | 14.205172% /1 of60 | 58.383054% /39 of60 | +44.177882pp |

Retcon walls/1,000 turns fell8.092445→0.145357 (229→11 collisions). Self-match candidate shares were87.669544%,85.489722%,80.641653% (mean84.600306%), with30/30 round wins. Target/SpinBot/Walls shares were100% for the candidate, against100%/97.026022%/94.186047% for the previous bot; both versions won all nine sample rounds.

**The gun did not demonstrate an accuracy improvement.** Against Retcon, confirmed shot hit rate fell12.195122%→1.070252%, recorded damage dealt1328.158881→162.7, while shots increased1312→3644 and recorded turns28298→75676. Damage taken fell4534.591066→506.085. These aggregate observations are consistent with a much longer-lived composition; this experiment cannot attribute the outcome to movement, selection, radar, wave inference, power, virtual-gun learning or any individual change. The selector's reported confidence is only a support factor, and no per-model use telemetry establishes a learning benefit.

All three candidate Retcon game shares exceeded50%, and39/60 round wins were observed. Nevertheless, the descriptive one-sided95% Student-t lower bound for its three-game mean is46.343461%, below50%; the round-level Wilson interval52.362619–75.832226% treats dependent/adaptive rounds as if independent and must not override that caveat. Three games against one pinned opponent do not establish reliable broad superiority. The equal-block improvement's descriptive lower bound is+27.705653pp; alternating order does not pair arena starts. This is a measured improvement of the actual default product against this frozen baseline and roster, not an isolated gun/AI gain or a general tournament claim.
