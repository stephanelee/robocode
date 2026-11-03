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
