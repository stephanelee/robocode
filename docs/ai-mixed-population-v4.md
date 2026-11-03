# Mixed-opponent training population: one fixed test

Prospective protocol before fitting. This changes only the training population of the rejected old-own-wave v3 scorer. The prior v1/v2/v3 decisions remain unchanged. No live gun, feature, model equation, candidate, fallback, threshold or controller change is authorized.

## Data roles, fixed before fitting

Use only the original45-round corpus in `benchmarks/ai-motion-v1/events.jsonl.gz`, uncompressed SHA-256 `d2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2`. The derived immutable role manifest is `tools/fixtures/mixed-population-v4/manifest.json`. Apart from datasetId, its sole changes from the original manifest are the three original sample games' splits from transfer to train. Every game/acquisition identity, order, opponent pin, arena, seed and original capture provenance remains unchanged.

Training: the original six Retcon training games plus the original single3-round Target, SpinBot and Walls games (nine whole games/27 already recorded rounds). Those three sample games are **spent observations now explicitly reused for training**, never independent confirmation. Development: the same three original Retcon development games (nine recorded rounds), used only for the one development screen. The three original Retcon withheld games remain spent, excluded from fitting, normalization and development scoring. All previous fresh/failed v3 games are excluded from this training population and from future confirmation IDs. No data collection is required for this screen.

## Exactly preserved model and weighting

Run the accepted, byte-identical `tools/containment_forecast_v3.py` and its original dependencies. Preserve all26 features, strict delivered old-wave context, horizon8/16/24/32/40, history8, eligibility/missing-label masks, four candidates/ties, current-projection fallback and target labels. This is a population test, not a new architecture or selector.

Each eligible observed training label has equal weight exactly as before. There is no game/opponent balancing, oversampling or extra loss term. Per-horizon normalization/intercepts use only that horizon's eligible training rows. The original unnormalized ridge sufficient-statistic matrix receives the same absolute diagonal alpha10 and unpenalized intercept; sample count increases, so effective regularization relative to total rows changes naturally with the declared training population, not through a new alpha setting. Long games and Retcon contribute more rows; this limitation is disclosed rather than silently corrected.

Retcon-only training is a plausible contributor to v3's transfer failure, not an established cause. These extra sample labels may improve or degrade Retcon/transfer behavior; no favorable outcome is assumed.

## One fit, one development screen

Publish the manifest/protocol and exact-source execution workflow before fitting. Verify the original corpus hash and unchanged scorer/dependency bytes. Fit once on the nine training games; publish frozen model bytes/SHA artifact before a separate dependent development job. Screen once on the three fixed Retcon development games. No old withheld/transfer outcomes are scored as confirmation and no sample training loss becomes a promotion metric.

Use unchanged v3 development gates: equal-game/equal24/32/40-turn containment gain over current at least2percentage points; at least2/3 improving games; worst-game and each short-horizon loss at most2points; original label/eligibility coverage rules. No adjustment for mixed-population convenience. If any guard fails or evidence is invalid, stop, preserve and document; no refit, parameter search or replacement run.

## Fresh mixed confirmation only after a valid pass

The preregistered new layout `tools/fixtures/mixed-population-v4/fresh-games.json` has six unique physical games, interleaved `mf4-retcon-01`, `mf4-target-01`, `mf4-retcon-02`, `mf4-spinbot-01`, `mf4-retcon-03`, `mf4-walls-01`; seeds2026110601..06, new acquisition IDs, three rounds each. Three Retcon games are new withheld and the three samples are new transfer. Budget exactly18 new rounds only if the unchanged development screen passes and the exact capture/evaluation workflow is independently reviewed. No earlier game ID/acquisition may be reused.

Keep accepted fixed-minimum-risk controller, pinned Retcon28981c7a/official1.4 samples,800x600, engine1.4 and current optional collector `legal-scan-v1.1-closed-round`/JAR5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883. Java17 build/parity, JDK25 games, Ubuntu26.04, strict dataset/status/official/native completeness, zero AIS skipped turns and no DQN writes. The prior immutable collector equivalence remains the rationale, but the old wrapper pins the old model: a NEW explicit model/report/collector-pair registration and external compatibility evaluator must be independently reviewed after the new model freezes. Never mutate the new model metadata, original scorer or old compatibility wrapper to bypass provenance.

Fresh gates stay unchanged: Retcon gain at least2points,2/3 improving games, worst/short losses at most2points, coverage; each transfer game's primary containment loss at most2points AND mean radial error increase at most5pixels. All six games are required; no Retcon-only deployment if transfer fails. A pass would establish only a bounded offline mechanism. A separate prospectively reviewed live gun-only A/B is required before any gameplay promotion.

Retain every outcome in docs/improvements.md with measured budgets, decision, PR/report/raw links. Independent reviewer owns merge. No automatic fit/collection retries, threshold changes or further model variation are authorized.

## Measured result: reject at development

The exact single [fit/screen run41540789004](https://github.com/stephanelee/robocode/actions/runs/41540789004), source `dd2ce34c55f413b0b6fcd075be367cd27cf8076c`, completed successfully under the prospectively reviewed044e manifest/protocol. Frozen model SHA-256 `10599435a4fa41480efa011b6cc4a0ff63ccc61a8f3c280f31cedff0c7ec6f4b` was published before the development job. No model/source/hyperparameter changes or second fit occurred.

**Reject:** development containment6.551200%→7.794587% (**+1.243387pp**, required≥2). All three development games improved (+0.638350,+1.801202,+1.290609pp); coverage, improved-game, worst-game and short-horizon checks passed. The sole failed check is primaryGain. Mean radial error148.020468→125.686838pixels is diagnostic and does not override rejection. Independent review reproduced all cells, selection/containment counts and the exact gate decision from raw data and frozen weights.

The declared training population supplied14,128 raw anchors:11,533 Retcon and2,595 reused sample anchors. Eligible fitted counts at horizons8/16/24/32/40 were13,487/13,272/13,056/12,840/12,625. The unchanged row weighting is dominated by long Retcon games; this was declared before fitting. No sample training error is presented as transfer success.

**The18 fresh rounds were not run.** Their mf4 IDs/layout remain unused. Zero new observation games, old-withheld scoring, live gun changes, refits, rescue runs or gate waivers follow. This test does not establish whether mixed training improves fresh sample behavior; it failed the prerequisite Retcon development gate. Existing v3 fresh-transfer rejection remains unchanged.

[PR #33](https://github.com/stephanelee/robocode/pull/33), [exact frozen model/reports/provenance](../benchmarks/ai-mixed-population-v4/README.md), [experiment index](improvements.md). The optional collector and all production source/JAR remain byte-identical to accepted50b2. These research weights are never loaded by AISurvivor. Any further variation requires a new prospective protocol and independent review.
