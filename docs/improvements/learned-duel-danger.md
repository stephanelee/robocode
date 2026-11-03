# Learned duel danger: first advanced-opponent candidate

Prospective protocol for [issue53](https://github.com/stephanelee/robocode/issues/53). No candidate gameplay outcomes exist when this registration is committed. The Diamond/Adaptive negative baselines and all previous rejections remain unchanged.

## Candidate and hypothesis

Baseline accepted source `12348fb656c64fa9eb36f400b28aa96cbb5fd383`, JAR SHA256 `21c5c3d2cb4007e3e0603556af313df7f5717732e384f039b005e9e72b2daecd`. Candidate `768f6647dab0265475c5812825c8186c2d01f2c3`, JAR SHA256 `b2da09ef774f8c699d4e12e7a69ae77d202914bbe8048317c61bb4fcefe7d5c8`. Later commits add only registration/tooling/docs, not gameplay or these assets.

The default duel controller previously used only head-on/linear clearance, despite collecting legal matched-hit danger bins. The candidate snapshots the correct shooter's profile and integrates its smoothed mass across a predicted bot footprint when at least3 effective matched hits exist. Otherwise the old analytic score is unchanged. Physics, walls, worst-wave aggregation and hysteresis are unchanged. This is a normalized distribution of observed matched hits, not a calibrated unconditional hit probability; misses are not new labels. Other modes receive no profile. Gun, radar, energy policy, opponent code and model format are unchanged.

Local XPSDESTKOP1 Java17 clean build passed86 tests, including5 new profile tests and existing movement conformance. Tests do not establish competitive strength.

## Fixed budget and setup

[Machine-readable layout](../../tools/fixtures/learned-duel-danger/protocol.json) fixes both sources/JARs, scenarios and seeds. Use default TRAIN with a fresh initially absent private checkpoint per arm/game; all learning rounds count. Opponents always start fresh, with unchanged pinned native source: Retcon0.2 at28981c7a; Diamond Port1.8.28 and Adaptive Prime1.0 at67b68c22. Reuse the merged source/wheel lock and native opponent loader. Do not use opponent-private state as AIS inputs.

- **Screen:** one10-round game per arm against each of Diamond, Adaptive and Retcon:60 physical rounds, seeds20262901–20262903.
- **Only if the screen passes:** fresh confirmation with two20-round blocks per arm against each of those three opponents:240 rounds, seeds20262911–20262916. Then Target/SpinBot/Walls, five rounds per arm each:30 rounds, seed20262921. Sample checks always run after a valid confirmation cohort, regardless of score outcome.
- Maximum330 physical rounds/24 private model endpoints:6 screen games,12 confirmation games and6 sample games. No reuse of screen models or results as fresh confirmation. Baseline/candidate order alternates by block/scenario. IDs carry candidate/stage/opponent/block/arm; seeds initialize AIS, not paired arena starts.
- Official1.4.0 runner,800x600 1v1 arena,100000-microsecond timeout, isolated ephemeral ports. Exactly Ubuntu26.04, Java17 fresh build/tests/committed-JAR parity, then JDK25 matches and the pinned private Python3.13 dependency environment. Preserve exact source/model/asset/runtime/opponent hashes.

## Frozen decisions

Use unrounded official whole-game score shares. For every Diamond/Adaptive/Retcon stage cohort, require complete matching official identities, all telemetry and finite numeric cold endpoints, candidate skipped turns no greater than baseline, and candidate walls per1,000 reported turns no greater than baseline plus10. Reported turns are the existing telemetry denominator, not an independently measured alive duration.

**Screen passes only if:**
1. Candidate-minus-baseline share is strictly positive against each advanced opponent, and the equal-opponent mean gain across Diamond/Adaptive is at least3 percentage points.
2. Retcon share difference is at least−5points; candidate Retcon share is strictly above50% and candidate first places exceed5/10.
3. All complete-evidence, skip and wall guards pass for all three opponents.

**Accept only after fresh confirmation if:**
1. Mean share gain is at least5points separately against Diamond and Adaptive, with positive differences in both blocks for each.
2. Retcon mean difference is at least−5points, candidate share exceeds50% in each Retcon game, and candidate first places exceed20/40.
3. All three opponent cohorts pass complete-evidence, skip and wall guards.
4. Each sample share difference is at least−5points and each candidate sample has zero skipped turns. Baseline sample skips are reported, not silently excluded.

This is a bounded, prospectively specified practical improvement test, not a powered tournament-superiority claim. Report all block variation and actual wins, including whether the candidate still loses to either advanced opponent. Intermediate acceptance is not completion of the user's objective: eventually fresh confirmation must show winning mean share and majority round wins separately against Diamond, Adaptive and Retcon before moving to melee.

No outcome-dependent tuning, favorable repeats, replacement games, threshold changes or optional budget extensions. Acquisition failure stops and preserves partial evidence; a new acquisition continuation requires explicit prospective independent review. A screen rejection skips confirmation. Any rejection restores accepted gameplay/JAR while retaining the patch and every result.

## Evidence and review

Independent exact source/protocol/workflow clearance precedes games. Preserve raw official arrays, every metric row, all actual model endpoints, Retcon native exports, unchanged Python source hashes and runtime packages, original logs and executed tooling, including invalid attempts. Update docs/improvements.md after each screen/confirmation with budgets/results/decisions/reasons and PR/report/raw links. A separate agent audits and merges only after required protected CI; implementers never self-merge.

## Measured outcome: rejected at screen

[PR #54](https://github.com/stephanelee/robocode/pull/54), [single run 41652803853](https://github.com/stephanelee/robocode/actions/runs/41652803853), execution `e3ae719b7a8e2529669c0a0060386e65441d4ac5`, completed all six screen games / 60 rounds. [Preserved evidence](../../benchmarks/learned-duel-danger/README.md) includes the original immutable registration, raw rows, official results, six cold counter-10 models, native Retcon exports and candidate patch. No acquisition failed or was repeated. Ubuntu 26.04.1, Java 25 and the pinned Python 3.13 environment ran the games after Java 17 fresh build/parity.

| Opponent | Baseline share | Candidate share | Difference | Baseline / candidate first places |
| --- | ---: | ---: | ---: | ---: |
| Diamond Port | 3.088326% | 7.291667% | +4.203341 pp | 0 / 0 of 10 |
| Adaptive Prime | 19.612591% | 23.066577% | +3.453986 pp | 1 / 1 of 10 |
| Retcon | 81.369248% | 65.354331% | −16.014917 pp | 9 / 6 of 10 |

**Reject; accepted gameplay and JAR restored.** The advanced-opponent mean gain was +3.828663 pp and both differences were positive, passing the registered screen thresholds. The candidate also passed Retcon's absolute share/win requirements, all skip guards (zero in both arms), and all wall-rate guards. However, Retcon's −16.014917 pp difference failed the binding −5 pp relative guard. It was the sole failed check. The planned 240 confirmation and 30 sample rounds were therefore not run; no gate was waived, tuned or replaced.

Damage dealt / taken was 50 / 823.2 → 105.2 / 619.136513 against Diamond, 252.800001 / 697 → 272.400001 / 519.4 against Adaptive, and 160.4 / 101.28 → 195.740002 / 63.2 against Retcon. Hits/shots were 10/183 → 21/405, 45/387 → 54/575 and 26/690 → 34/707 respectively. Wall contacts were 0→0, 0→1 and 1→4; reported-turn denominators were 3,471→8,781, 6,365→9,299 and 12,523→13,373. These counters do not override official score or establish precise survival duration.

Each opponent has only one unpaired ten-round game per arm; arena starts are not paired by the AIS RNG seed. These observations neither establish a causal Retcon regression nor prove benefit against advanced opponents, which still won most rounds. Matched-profile-use counters were not collected, so the screen does not establish when the new profile affected commands. The rejected candidate remains a source patch and immutable measured source/JAR hash, not active production code. Issue 53 remains open. The next experiment requires a separate prospective plan.
