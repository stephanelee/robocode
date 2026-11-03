# Fixed shallow-tree gain selector: one nonlinear family test

Prospective declaration before implementation/fitting. The preceding ridge-family rejections stay unchanged. This tests a combined hypothesis: shallow nonlinear partitions plus direct gain-over-current targets may separate straight/circular/surf-like contexts better than marginal linear scores. Both model family and target objective change together; no result can isolate their causal contributions. No live gun, movement or DQN change is authorized.

## Same population, features and outcomes

Use the exact v4 mixed-role manifest `tools/fixtures/mixed-population-v4/manifest.json` (SHA-25624d0c6bfe49f62d3a44b1941e032c5514b0950b29bbb3befad573976cc6c5308) and original d2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2 raw corpus. Nine whole training games: original six Retcon plus three spent Target/SpinBot/Walls games now explicitly TRAINING. Same three Retcon development games, adaptively reused only for screening. Original withheld games and all subsequent fresh/failed observations remain excluded from fit/DEV. Sample training labels are never independent transfer evidence.

Preserve all26 legal v3 raw inputs/context, history8, exact observed labels at horizons8/16/24/32/40, masks/eligibility and four original candidate projections. No new scan, private state, game identity feature, interpolation, sample weighting or candidate is added. Each eligible observed label has equal weight, retaining long-game/Retcon dominance. Trees use raw finite features without normalization because ordered splits do not require scaling. Original feature/projection helpers stay byte-identical.

## Frozen model, objective and abstention

One independent deterministic multioutput regression tree per horizon, predicting three targets in candidate order linear/circular/headon. Each target is candidatePointContainedWithin18 minus currentPointContainedWithin18, so values are−1/0/+1. Current gain is identically zero and supplies the abstention baseline.

Root depth0, maximum leaf depth3 (at most8 leaves), minimum leaf support200 eligible training labels. At each node and each of26 features in index order, sort only that node's training feature values. Candidate thresholds are values at index floor(j*(n−1)/33), j=1..32; deduplicate and examine in ascending order. Split is feature<=threshold versus greater. Disallow children below200. Choose the greatest reduction in the sum of squared errors over all three gain outputs. Require reduction>1e−12; deterministic ties within that tolerance retain lower feature index then lower threshold. No random feature sampling, pruning search or hyperparameter optimization.

Each leaf estimates the three unweighted empirical mean gains. On an eligible query, choose the largest predicted candidate gain only when it is at least0.02; otherwise use current. Candidate ties follow linear, circular, headon. Insufficient/invalid eligibility uses current exactly as before. These empirical gain scores and row support are not calibrated probabilities, confidence intervals or independent evidence because neighboring observations are correlated. Depth/support/.02 abstention/cutpoint count and loss are frozen before fitting; no post-outcome tuning.

The tree replaces ridge; alpha10 is not a tree parameter and is not reinterpreted as a new regularizer. The only declared model-family change is this complete fixed gain-tree selector. Preserve all earlier tools and report/gate semantics unchanged.

## One fit and one development screen

Publish exact source/tests/workflow after independent preclear. Fit once on TRAIN only; publish frozen model bytes/SHA artifact before a separate dependent single Retcon DEV screen. Lock all source/helper/input hashes. No old withheld/transfer scoring, repeated fit, search or rescue run.

Keep the identical DEV gates: equal-game/equal24/32/40-turn containment gain over current≥2percentage points, at least2/3 improving games, worst-game and each short-horizon loss≤2points, unchanged coverage rules. Radial error is diagnostic, not an alternative success criterion. Any failed guard or invalid evidence stops before new collection and is recorded without threshold waiver.

## Conditional fresh confirmation and limits

Only a valid DEV pass plus independent exact workflow/registration review may run `tools/fixtures/nonlinear-gain-v5/fresh-games.json`: tg5-retcon-01, tg5-target-01, tg5-retcon-02, tg5-spinbot-01, tg5-retcon-03, tg5-walls-01, unique acquisitions/seeds2026110701..06, three rounds each (18 new rounds). Those games have never run; no original, w3/w3b or mf4 identity is reused. Three Retcon games are fresh withheld; samples are fresh transfer.

Use unchanged fixed-minimum-risk controller,800x600, engine1.4.0, pinned Retcon28981c7a and official1.4 samples, current collector/JAR5ebf via explicit reviewed old→new collector equivalence. Java17 build/parity, JDK25 games, Ubuntu26.04, complete legal/status/official/native evidence, zero AIS skips and no DQN writes. Freeze new model/report before an explicit new compatibility wrapper/registration; never weaken an old evaluator or edit model provenance to force acceptance.

Fresh gates stay Retcon≥2pp gain/2of3/worst-short≤2pp/coverage AND each sample containment loss≤2pp plus error increase≤5pixels. No Retcon-only promotion if transfer fails. Offline success still requires a separate prospectively reviewed live gun-only battle A/B before deployment or a competitive-strength claim. Update docs/improvements.md with measured results/decision/PR/raw links after this test; independent reviewer owns merge.
