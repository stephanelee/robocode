# Coherent duel v2: original whole-product evidence

**Registered decision: retain; all gates passed.** [PR43](https://github.com/stephanelee/robocode/pull/43), [single measured run41571217551](https://github.com/stephanelee/robocode/actions/runs/41571217551), artifact13201776252. [Full immutable protocol, results and limitations](../../docs/improvements/coherent-duel-v2.md).

- 168 physical rounds/198 AIS records,15 official games, six native Retcon captures and18 independently cold models. No skipped AIS turns or loaded models.
- Default Retcon share49.637743→85.812811%, wins35→55/60. Hit rate0.720384→5.220005%; damage per confirmed shot0.035928495→0.270210445. Self-match30/30 wins; sample guards pass.
- The tested bundle combines a genuine GF/circular candidate with consistent firing/virtual evidence and strict energy finishing. Results do not isolate either change or a learning benefit. Only three Retcon games per version were measured; no broad superiority or calibrated confidence claim.

`experiment.json` is the original canonical report. `registered-plan.md` is byte-identical to immutable433f; `registered-games.json` records the locked layout. `source/` contains the exact executed driver, tests and one-shot workflow. `raw/` preserves official arrays/logs, both labelled AIS participants' metrics/launch provenance and18 original `model.dat` endpoints, plus native Retcon exports. `bundle-jar-hashes.json` records the18 actual runnable JAR hashes without duplicating JARs here; full bundles remain in the original artifact. Candidate9679 and accepted9daf JAR/source/runtime pins are in the report.

`activation/` separately documents an unstarted ready-event activation and the independently cleared trigger-only correction before any performance data. It is not a failed or replacement score attempt. `files.sha256.json` verifies142 preserved evidence/provenance files; Git attributes preserve their bytes. Tests validate original hashes/cold counters/identities and reproduce the frozen decision, never launching new games or fitting a model.
