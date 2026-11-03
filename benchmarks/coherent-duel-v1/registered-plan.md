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
