# Coherent default duel: original whole-product evidence

**Registered decision: retain; every gate passed.** [PR42](https://github.com/stephanelee/robocode/pull/42), [single run41564742819](https://github.com/stephanelee/robocode/actions/runs/41564742819), artifact13198895411. [Full protocol, numbers and limitations](../../docs/improvements/coherent-duel-v1.md).

- 168 physical rounds/198 AIS telemetry rows,15 official games, six native Retcon captures and18 independent cold model endpoints. All are preserved here, without duplicated runnable JARs or opponent bundles.
- Default TRAIN Retcon equal-game share14.205172→58.383054%, wins1→39/60. Self-match30/30 wins; all sample guards and zero-skip checks passed.
- Gun hit rate fell12.195122→1.070252%, damage1328.158881→162.7, shots1312→3644 and turns28298→75676. This is a whole-composition gain; no individual gun/learning benefit is established.
- Candidate absolute three-game mean's descriptive95% t lower bound46.343461% remains below50%. Retcon-only performance and sample wins do not prove general strength.

`experiment.json` is the original canonical result; `registered-plan.md` and `registered-games.json` preserve the immutable0d595 plan and full fixed layout. `source/` contains the exact executed driver, focused tests and temporary workflow. `raw/` preserves official arrays/logs, each AIS participant's metrics/model and original/labelled launcher provenance, and complete native Retcon exports. `bundle-jar-hashes.json` records all18 original bundled JAR hashes; source/JAR pins and runtime provenance are also in the original report. Every model header counter equals its game's20/10/3 scored rounds, with no loaded checkpoint.

`files.sha256.json` verifies139 original archived files. Git attributes preserve evidence bytes. The GitHub artifact also contains runnable/opponent copies; source and asset hashes identify those bundles without storing them repeatedly here. Original execution/source/JAR data and the original registration remain unchanged. Tests reproduce the frozen result and validate evidence; they never launch games or fit a model.
