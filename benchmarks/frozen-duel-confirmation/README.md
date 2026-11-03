# Frozen fixed-duel confirmation

[PR #40](https://github.com/stephanelee/robocode/pull/40) / [immutable protocol and measured result](../../docs/improvements/frozen-duel-confirmation.md).

**Passed the registered observation gates:**55.903600% equal-game mean share,107/200official round wins,6/10games above50%. Source7104b65/JARd047 stayed frozen. Both independent runner batches completed100rounds without skipped turns or model loads/writes. No retry or replacement; no gameplay change.

This is a bounded fixed-policy result against pinned Retcon. The descriptive t lower bound48.279885% and Wilson46.586647%–60.281436% interval cross50%. Within-game adaptation creates dependence. Statistical superiority, default-mode benefit and general strength are not established.

- `confirmation.json`: exact single aggregate artifact13196578720.
- `batch-1.json` / `batch-2.json`: exact five-game outputs, artifacts13197012715/13196458790.
- `raw/<gameId>/`: all10official results,200telemetry rows,10native Retcon exports/capture manifests and runner logs. No runnable bundles or duplicate JARs retained here.
- `registered-plan.md` / `registered-games.json`: original immutable8af14d7 Git blobs.
- `confirmation-source.py` / `execution-workflow.yml`: exact measured1853127 source; temporary active workflow removed before the evidence push.
- `files.sha256.json`: hashes of57 retained original evidence/source files plus run, source, registration and artifact identities.

The [single actual run](https://github.com/stephanelee/robocode/actions/runs/41560662466) used Ubuntu26.04.1, Java17 build/parity then JDK25 matches, engine1.4.0 and unchanged Retcon28981c7a/API1.3.1. Per-batch provenance records preserve actual OS/Java/opponent hashes. Bot RNG seeds are fixed; starting positions are unseeded/unpaired. No old development/A-B game was reused as confirmation.
