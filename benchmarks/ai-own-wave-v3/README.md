# Old-wave context model: development passed, fresh transfer rejected

[PR #31](https://github.com/stephanelee/robocode/pull/31) / [prospective feature and confirmation protocol](../../docs/ai-old-wave-context-v3.md).

Exactly one train-only fit and one development-only screen in [run41533756340](https://github.com/stephanelee/robocode/actions/runs/41533756340) passed all unchanged gates: current proxy6.551200%→scorer9.300498% (+2.749298 points), with all three development games improving. These are not fresh-confirmation or battle-strength results. The old withheld/transfer games were not scored.

The exact frozen model `a34109512f3d451ceee64b1e47342d91b07eae8be3f1670896c448925562ddcc`, passed screen `4070177e8917fc9a131f31e6c63f215a78996bfcf665ae512412cb32126b9005`, fit report, original pre-fit plan and actual fitting/context/workflow sources are preserved here before new matches. Original input observations/dependencies remain in [the first legal corpus](../ai-motion-v1/README.md). Research weights are not loaded by AISurvivor.

The first fresh attempt used the original collector and failed as preserved below. A separately reviewed second18-round attempt used only the explicitly registered recorder boundary repair; the original model/math/gates stayed frozen. Its fresh transfer rejection is recorded below and in the experiment index.

## Invalid initial fresh attempt

[Run41535141065](https://github.com/stephanelee/robocode/actions/runs/41535141065) completed six physical rounds, of which only the first three validated; capture failed after Target's round3 close. Confirmation was skipped. [Preserved failed evidence](failed-attempt-41535141065/preservation.json) includes3,136 raw rows in deterministic gzip, original statuses, official results, metrics, native Retcon exports and logs. No original status is changed, no partial confirmation is reported, and no live model is enabled. The development pass remains distinct from an uncompleted fresh confirmation.


## Final fresh outcome: rejected, no live model

The independently precleared second attempt [run41537758736](https://github.com/stephanelee/robocode/actions/runs/41537758736) completed18 fresh rounds/six games/8,243 rows under immutable registration `c99d83dd69df3f325cc2f76cfe06076832b37787`. Collector source245e plus exact frozen model/dependencies were verified against actual sourcef1886f2; the explicit wrapper records original training and actual capture provenance. No model metadata was changed to bypass the strict original evaluator.

Fresh Retcon containment3.584676%→6.318742% (+2.734066pp), all three games positive; every Retcon guard passed. **Reject solely for transfer:** SpinBot98.894176%→23.157008% and Walls51.876150%→12.875550%, with error increases63.564212px/106.914268px, violate unchanged transfer guards. Target passed. This is an offline point-containment mechanism test, not a live AI battle result.

- [Frozen single-evaluation report](fresh-report.json).
- [Actual capture/provenance manifest](confirmation/manifest.json), [registered collector compatibility](confirmation/collector-compat.json), [raw-data preservation/index](confirmation/preservation.json).
- `confirmation/events.jsonl.gz` is deterministic gzip of4,953,284 original bytes, SHA-256 `ab8a2d69625b46a8f60d31f966585ddb5da926bf315f02ff36c16afec1f4131c`; per-game offsets/checksums are indexed. Statuses/official/metrics/native/logs are retained per game.
- Original capture artifact13185222035 and results artifact13183364717 contain the full runnable bundles; no duplicate research JAR is committed.
- The original failed6-round attempt remains invalid/unscored; total24 physical rounds across both attempts. No refit, rescreen, threshold change or further retry occurred. Modela341 remains research-only. The default-off recorder correction alone is retained.
