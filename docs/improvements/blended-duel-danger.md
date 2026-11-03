# Blended duel danger: fixed analytic prior and matched-hit support

Prospective protocol for [issue53](https://github.com/stephanelee/robocode/issues/53). No gameplay outcomes for this candidate exist at registration. PR54 rejected the profile-replacement candidate after60 rounds; its−16.014917pp Retcon difference and every raw/model byte remain unchanged. This is a separately registered design, not a rescue, threshold change or repeat of that cohort.

## Candidate and hypothesis

Baseline accepted PR48 source `9b1624da2d9b09ec9cc4f580b51bc82edd2c1485`, JAR SHA256 `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800`. Candidate `5b4bfdff52927148e5092a333769d4c3630530a5`, JAR `aaf2c92d0df5ad203f27bd1da554dd73acfdf929349746150b405c6b404ad5a9`. Later commits add only registration/tooling/docs.

The rejected replacement discarded the analytic score once sparse matched-hit support existed. That is a source-based concern, not a proven cause of its Retcon loss. This candidate keeps the same legal same-shooter history, unsigned GF footprint kernel and defensive snapshot, but blends each wave score. Define analytic A=min(60,lateral clearance), learned L=60*(1−footprint mass), support n=sum of finite nonnegative decayed matched-hit bins. For validn≥3, w=min(0.25,n/(n+24)); otherwise use the original analytic branch bit-for-bit. Score=(1−w)*A+w*L before the unchanged worst-wave minimum. Both scores stay in[0,60]. The learned weight is1/9 atn3 and reaches its25%cap atn8; it does not represent24 prior observations or calibrated confidence. At least75%analytic defense remains, and this is a fixed support heuristic with no cap/support search.

Physics, wall veto, wave matching, worst-wave aggregation, hysteresis, gun, radar, energy and model format are unchanged. Only default TRAIN/controller duel movement receives the legal profile; other policies/modes remain analytic. Pure tests must verify no-evidence/invalid evidence exact fallback, bounds/support cap, fixed blended arithmetic, same-shooter snapshot and wall safety before play. No profile-use telemetry is added, so actual activation frequency is not established by the round metrics.

## Fixed budget and setup

[Machine-readable layout](../../tools/fixtures/blended-duel-danger/protocol.json) fixes both sources/JARs, scenarios and seeds. Use default TRAIN with a fresh initially absent private checkpoint per arm/game; all learning rounds count. Opponents always start fresh, with unchanged pinned native source: Retcon0.2 at28981c7a; Diamond Port1.8.28 and Adaptive Prime1.0 at67b68c22. Reuse the merged source/wheel lock and native opponent loader. Do not use opponent-private state as AIS inputs.

- **Screen:** one10-round game per arm against each of Diamond, Adaptive and Retcon:60 physical rounds, seeds20263001–20263003.
- **Only if the screen passes:** fresh confirmation with two20-round blocks per arm against each of those three opponents:240 rounds, seeds20263011–20263016. Then Target/SpinBot/Walls, five rounds per arm each:30 rounds, seed20263021. Sample checks always run after a valid confirmation cohort, regardless of score outcome.
- Maximum330 physical rounds/24 private model endpoints:6 screen games,12 confirmation games and6 sample games. No reuse of screen models or results as fresh confirmation. Baseline/candidate order alternates by block/scenario. New IDs use blend-v1 plus stage/opponent/block/arm; seeds initialize AIS, not paired arena starts.
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

Local Java17 clean build passed all95 tests, including six new profile fixtures and accepted replay/physics coverage. Pure tests and old replay states are not strength evidence.

## Measured outcome: screen passed, fresh confirmation rejected

[PR #55](https://github.com/stephanelee/robocode/pull/55) / [sole run41658783365](https://github.com/stephanelee/robocode/actions/runs/41658783365), execution `4ad4227a28c9849730146bae5c22956689a80671`, completed every registered stage:60 screen +240 fresh confirmation +30 samples =330 rounds,24 distinct cold finite checkpoints. [Raw evidence, patch and models](../../benchmarks/blended-duel-danger/README.md) preserve the immutable18279a4 registration, every official result/metric row, six native Retcon exports, source/runtime provenance and executed tooling. No game was replaced or repeated.

The screen advanced under all unchanged rules. Share differences were Diamond +1.085697pp, Adaptive +5.909850pp (equal-opponent mean +3.497774pp), Retcon +1.293031pp; Retcon candidate share94.652406% and10/10 first places passed absolute guards. Both arms had zero skips.

Fresh confirmation alone decided retention:

| Opponent | Baseline mean share | Candidate mean share | Mean difference | Per-block differences | Baseline / candidate first places |
| --- | ---: | ---: | ---: | --- | ---: |
| Diamond Port | 4.234608% | 4.522532% | +0.287924pp | +1.107940, −0.532093pp | 0 / 0 of40 |
| Adaptive Prime | 17.639776% | 16.750978% | −0.888798pp | −1.957210, +0.179615pp | 1 / 1 of40 |
| Retcon | 84.717824% | 89.336197% | +4.618373pp | +6.339517, +2.897230pp | 36 / 38 of40 |

**Reject at confirmation; accepted9b1624/B41B production restored.** The sole failed check was `advancedGains`: neither advanced opponent reached the required +5pp mean, and each had one negative block. Retcon relative and absolute guards passed: candidate game shares91.431730% /87.240664%,38/40 first places. Skip/wall guards and all sample comparisons also passed. Every one of330 metric rows records zero AIS skips. Samples: Target100→100%, SpinBot96.125908→100%, Walls100→100%, with5/5 candidate first places each. The30 sample rounds ran as registered even though the fresh advanced gains failed.

Confirmation hits/shots were Diamond47/856→58/851, Adaptive189/1721→175/1708 and Retcon157/2395→155/2344. Damage dealt/taken was274/3220.05→295.2/3328.317668,1091.600001/2711→1015.000001/2700 and883.080009/314.8→934.240011/279.48. Wall contacts were5→1,6→3 and0→6; reported-turn denominators15,987→17,354,27,723→27,820 and44,534→45,116. These counters do not establish a causal movement effect or precise survival duration.

Do not pool the favorable screen with confirmation, or either stage with rejected PR54. Only two unpaired fresh games per arm/opponent support the confirmation comparison; this is not an IID-round superiority estimate. Matched-profile-use counters remain absent, so activation frequency is unknown. The blend did not establish beating either advanced opponent. No cap/support tuning or gate relaxation follows this result. During the already-running experiment, the user redirected work to a separate DiamondChallenger bot; no further AIS candidate or automatic gameplay promotion was authorized. Its separate protocol will leave production AIS unchanged until a separately reviewed transfer experiment.
