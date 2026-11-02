# Legal motion-data and first AI forecast pilot

Prospective proposal; **no acquisition or training has started**. The accepted targeting seam is [PR #25](https://github.com/stephanelee/robocode/pull/25), merged as `fc847a82a67a6142386b9031badd805bb08121ae`. Existing aggregate round metrics contain no scan histories and cannot train this predictor.

The immediate implementation is an optional, passive legal-data recorder, disabled unless explicit dataset-path/game-ID settings are supplied. It snapshots own state, cached legal scans, strictly previous submitted intent, and confirmed own BulletFired events before selecting current commands. No private Retcon exports or observer truth enter features/labels. No movement, gun decisions, RNG calls, DQN, model format, or rejected scan-lag repair changes. Buffer bounded round data and flush outside turn decisions; report truncation/I/O failures as invalid acquisition, never silently select records. Retain accepted-command/state goldens with recording disabled and test enabled recording separately.

## Proposed bounded acquisition

After independent recorder/schema/protocol review, freeze one source/JAR and a manifest before the first match: **12 fresh Retcon games ×3 rounds**, with whole games assigned **6 train /3 development /3 withheld**, plus **three rounds each** against official 1.4.0 Target, SpinBot and Walls as frozen transfer games. Total: **45 rounds**. This is data acquisition, not a candidate score-promotion experiment.

All games use real fixed minimum-risk mode, absent model paths, isolated fresh processes and preassigned unique IDs/seeds. Fixed mode does not use bot RNG and does not pair starts. Define IDs `r-train-01..06`, `r-dev-01..03`, `r-withheld-01..03`, `x-target-01`, `x-spinbot-01`, `x-walls-01`; every round/target from a game keeps its split. Final immutable manifest must pin actual source/JAR/collector/engine/opponents, arena dimensions, acquisition IDs and launch order before collection. Proposed order interleaves train/dev/withheld games, then transfer. Do not inspect withheld forecast outcomes to revise features, hyperparameters or thresholds.

Require complete official results/round telemetry, zero AIS skipped turns, no model writes, complete native Retcon diagnostics kept separate from inputs, recorder success and bounded finite schema-valid rows. Preserve every invalid attempt and stop for review; no automatic replacement games or hidden rescue reruns. All jobs use exact Ubuntu26.04, Java17 build/parity and JDK25 matches, with actual provenance and raw hashes.

## Supervised proof, before live deployment

Use only legal histories ending at the anchor. Future labels exist only when the same target has a fresh observed scan at exactly horizons **8/16/24/32/40**; mask missing labels, never interpolate or cross rounds/games. The 24–40-turn range covers relevant Retcon bullet-flight times rather than only short synthetic horizons. Split whole games before window construction, normalization or fitting. Exclude game/capture/target IDs and splits from learned features. Report coverage and errors by horizon and whole game; correlated turns are not independent samples.

Compare frozen head-on, linear, circular and current accepted projection baselines on identical masks. First learned candidate should be a small separately supervised residual predictor, not a DQN rewrite. Fit normalization and coefficients on train games only; select only preregistered hyperparameters using development games; freeze source/model/preprocessing hashes before a single withheld/transfer evaluation. Include legal own-shot timing because Retcon reacts to us, while treating the model as predictive rather than a causal simulator.

Before fitting, independently lock the exact features, model family/training budget, development selection metric, fallback, minimum label coverage, held-out displacement/virtual-containment criteria and later live A/B budget. No model is promoted from offline error alone. If mechanism evidence is weak, record rejection before spending competitive rounds. A later live gun-only experiment must preserve movement/DQN, match learning budgets, test Retcon plus the existing pinned samples, and publish its own prospective rules and every outcome.

## Milestones

1. Accepted minimal seam: complete, exact 288-case replay and five-round compatibility documented.
2. Optional recorder + independent schema/label-builder fixtures: implementation in progress; no collection yet.
3. Reviewed immutable acquisition/model protocol, then 45-round dataset and frozen offline proof.
4. Only after supported mechanism and separate preclear: bounded live gun A/B, report/index, independent review/merge. No claim that adding AI guarantees wins.
