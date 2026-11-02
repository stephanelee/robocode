# Legal motion-data and first AI forecast pilot

Registered before acquisition; the completed first dataset/model result appears below. The accepted targeting seam is [PR #25](https://github.com/stephanelee/robocode/pull/25), merged as `fc847a82a67a6142386b9031badd805bb08121ae`. Existing aggregate round metrics contained no scan histories, so a separate optional recorder was required.

The immediate implementation is an optional, passive legal-data recorder, disabled unless explicit dataset-path/game-ID settings are supplied. It snapshots own state, cached legal scans, strictly previous submitted intent, and confirmed own BulletFired events before selecting current commands. No private Retcon exports or observer truth enter features/labels. No movement, gun decisions, RNG calls, DQN, model format, or rejected scan-lag repair changes. Buffer bounded round data and flush outside turn decisions; report truncation/I/O failures as invalid acquisition, never silently select records. Retain accepted-command/state goldens with recording disabled and test enabled recording separately.

## Proposed bounded acquisition

After independent recorder/schema/protocol review, freeze one source/JAR and a manifest before the first match: **12 fresh Retcon games ×3 rounds**, with whole games assigned **6 train /3 development /3 withheld**, plus **three rounds each** against official 1.4.0 Target, SpinBot and Walls as frozen transfer games. Total: **45 rounds**. This is data acquisition, not a candidate score-promotion experiment.

All games use real fixed minimum-risk mode, absent model paths, isolated fresh processes and preassigned unique IDs/seeds. Fixed mode does not use bot RNG and does not pair starts. Define IDs `r-train-01..06`, `r-dev-01..03`, `r-withheld-01..03`, `x-target-01`, `x-spinbot-01`, `x-walls-01`; every round/target from a game keeps its split. Final immutable manifest must pin actual source/JAR/collector/engine/opponents, arena dimensions, acquisition IDs and launch order before collection. Proposed order interleaves train/dev/withheld games, then transfer. Do not inspect withheld forecast outcomes to revise features, hyperparameters or thresholds.

Require complete official results/round telemetry, zero AIS skipped turns, no model writes, complete native Retcon diagnostics kept separate from inputs, recorder success and bounded finite schema-valid rows. Preserve every invalid attempt and stop for review; no automatic replacement games or hidden rescue reruns. All jobs use exact Ubuntu26.04, Java17 build/parity and JDK25 matches, with actual provenance and raw hashes.

## Supervised proof, before live deployment

Use only legal histories ending at the anchor. Future labels exist only when the same target has a fresh observed scan at exactly horizons **8/16/24/32/40**; mask missing labels, never interpolate or cross rounds/games. The 24–40-turn range covers relevant Retcon bullet-flight times rather than only short synthetic horizons. Split whole games before window construction, normalization or fitting. Exclude game/capture/target IDs and splits from learned features. Report coverage and errors by horizon and whole game; correlated turns are not independent samples.

Compare frozen head-on, linear, circular and current accepted projection baselines on identical masks. First learned candidate should be a small separately supervised residual predictor, not a DQN rewrite. Fit normalization and coefficients on train games only; select only preregistered hyperparameters using development games; freeze source/model/preprocessing hashes before a single withheld/transfer evaluation. Include legal own-shot timing because Retcon reacts to us, while treating the model as predictive rather than a causal simulator.

Before fitting, the exact 24 features, fixed ridge alpha10, train-only normalization, baseline selection/fallback and offline thresholds were independently locked in [the model protocol](ai-forecast-model-lock.md), included in immutable registration `3b63589112518e71207ba427912e0224f14640bc`. No model is promoted from offline error alone. If mechanism evidence is weak, record rejection before spending competitive rounds. A later live gun-only experiment must preserve movement/DQN, match learning budgets, test Retcon plus the existing pinned samples, and publish its own prospective rules and every outcome.

## Milestones

1. Accepted minimal seam: complete, exact 288-case replay and five-round compatibility documented.
2. Optional recorder + independent schema/label-builder fixtures: complete; disabled and enabled command/state parity pass.
3. Reviewed acquisition/model protocol, 45-round dataset, fixed fit and one frozen evaluation: complete; **first learned model rejected**.
4. Only after supported mechanism and separate preclear: bounded live gun A/B, report/index, independent review/merge. No claim that adding AI guarantees wins.

## Actual first AI result

**Reject the learned forecaster; no live AI gun is enabled.** [PR #28](https://github.com/stephanelee/robocode/pull/28) completed the full real-data pipeline in [run 41520911644](https://github.com/stephanelee/robocode/actions/runs/41520911644). All **45 rounds /15 fresh games** validated with zero AIS skipped turns and no model writes. The recorder produced **26,962 legal records**: 11,533 train, 6,069 development, 6,765 withheld and 2,595 transfer. Twelve Retcon games supplied 36 rounds; Target, SpinBot and Walls supplied three transfer rounds each.

The fixed per-horizon ridge models used six Retcon training games; counts of eligible exact labels were 10,976 /10,833 /10,689 /10,545 /10,402 at horizons8/16/24/32/40. Development selected **head-on** as the comparison/fallback baseline. The frozen model SHA-256 **`2b9650591220b17847ada24c19bc142d16360a9d40f263065452bdbcc665b8d5`** and artifact were published by the fit job before the separate, single withheld/transfer job began. There was no retuning, replacement game, or second evaluation.

Primary values are equal-game means over the three withheld Retcon games, averaging horizons24/32/40 equally. Radius-18 containment means predicted-point error≤18 pixels; it is **not an engine bullet hit rate**.

| Frozen method | Mean position error | Radius-18 containment proxy |
| --- | ---: | ---: |
| Head-on, development-selected comparator | 121.983906 px | 6.227910% |
| Linear | 160.037971 px | 3.569443% |
| Circular | 139.558180 px | 10.955735% |
| Current projection family with arena clamp | 135.668436 px | 11.691522% |
| Learned ridge residual | 124.118914 px | 1.969872% |

The learned mean error was about **8.51% lower than the current projection**, but **1.75% higher than the best development-selected simple baseline**, rather than the required 10% improvement. Containment fell by **4.258038 percentage points** against the selected baseline, rather than gaining two points. Only one of three withheld games improved; worst-game and transfer guards also failed. Coverage and short-horizon guards passed. Exactly five checks failed: primaryError, improvedGames, worstGame, containment, transfer. Unrounded values and the unchanged original criteria determined rejection.

| Horizon | Current error / proxy | Learned error / proxy |
| --- | ---: | ---: |
| 8 | 20.364 px /54.891% | 22.783 px /52.353% |
| 16 | 59.872 px /27.687% | 61.292 px /6.576% |
| 24 | 101.310 px /16.974% | 96.336 px /2.943% |
| 32 | 136.989 px /10.813% | 124.523 px /2.077% |
| 40 | 168.707 px /7.287% | 151.497 px /0.889% |

Each withheld game had at least 1,735 observed exact labels at every primary horizon, above the required50; fresh-label coverage over those cells ranged87.85%–96.34%. These are radar-observable subsets, not hidden trajectories. Correlated anchors are not independent trials.

### Transfer and interpretation

| Transfer game | Selected head-on error / proxy | Current error / proxy | Learned error / proxy |
| --- | ---: | ---: | ---: |
| Target | 45.135 px /46.992% | 55.822 px /44.901% | 64.567 px /8.643% |
| SpinBot | 73.756 px /0.446% | 35.172 px /31.232% | 98.233 px /0.106% |
| Walls | 208.216 px /0% | 49.476 px /61.816% | 138.285 px /0% |

Lower average displacement error does not imply better aim: a conditional mean can fall between plausible motion paths. That is a hypothesis for the failure, not a proven explanation or permission to retune this model. Any diagnosis or next candidate must use training/development data; the already inspected withheld and transfer games are spent confirmation data. A future model requires new prospective criteria and fresh confirmation.

During acquisition the **unchanged fixed controller**, not this offline learned model, won5/36 Retcon rounds and9/9 sample rounds. Those scores do not measure an AI gun and establish no competitive improvement. No live gun A/B was warranted after the offline rejection.

### Durable evidence and final disposition

- Exact measured source `eeff2c053f77e1e4dc4e8b814239c5af13ff1aed`; original plan/layout registration `3b63589112518e71207ba427912e0224f14640bc`. All jobs used actual Ubuntu26.04.1, Java17 build/parity and JDK25 matches; engine/server/runner1.4.0, unchanged Retcon0.2.0/API1.3.1 and official1.4.0 samples were hash-pinned.
- [Raw observations, manifest, official games/statuses/metrics/native exports, exact frozen model and reports](../benchmarks/ai-motion-v1/README.md) are retained. Raw input SHA-256 `d2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2`; the 16,215,891 original bytes are losslessly stored as deterministic gzip with per-game byte offsets/hashes.
- Original artifacts: [raw capture](https://github.com/stephanelee/robocode/actions/runs/41520911644/artifacts/13176834026), [frozen model published before evaluation](https://github.com/stephanelee/robocode/actions/runs/41520911644/artifacts/13177827253), [single withheld results](https://github.com/stephanelee/robocode/actions/runs/41520911644/artifacts/13177412908). Independent review reconstructed train-only coefficients/scaling, selected baseline and every evaluation cell.

The temporary pilot workflow is removed after preserving the outcome. Retain the optional default-off recorder, tested fitting/data tools and research evidence. No learned weights are loaded by AISurvivor, and movement, targeting decisions, DQN and model format remain unchanged. [Issue #27](https://github.com/stephanelee/robocode/issues/27) remains the tracker for further separately reviewed AI work.
