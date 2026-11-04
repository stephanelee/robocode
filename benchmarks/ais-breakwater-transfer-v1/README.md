# Rejected AIS Breakwater movement-only transfer

The sole [registered run41726813473](https://github.com/stephanelee/robocode/actions/runs/41726813473) completed all90screen rounds. The candidate improved several secondary measurements but failed all three Breakwater-specific progress gates. Decision: **reject at screen; zero confirmation rounds**. Original report and thresholds are unchanged. [Detailed results](../../docs/improvements/ais-breakwater-transfer-v1.md) · [PR74](https://github.com/stephanelee/robocode/pull/74)

`experiment.json`, registered plan/protocol, and directly readable raw numerical records/logs preserve original bytes. All956original artifact files, including10private AIS checkpoints, two Breakwater endpoints, two Retcon exports, all staged JARs/scripts/source and compiler caches, are retained losslessly in `original-payloads.zip`. Its550unique payloads are keyed by SHA256; `original-files.json` maps every original path to its bytes/hash. Duplicate files are stored once. This compact representation preserves all file content, not original ZIP headers/timestamps. Original service ZIP SHA256 `4c2b9db52e9a5a75526400bf78a348b772be7dc8361374b79404e7565de96d0f` matched artifact13275875597 before import and was independently verified.

`source/` contains the exact executed driver, tests, dependencies, locks, workflow and original job log from ce56e83. `candidate.patch.gz` retains the binary-capable production patch from baseline9b1624 to measured source1cd8db2; the exact candidate JAR4aceb212 is also in the original payloads. `provenance.json` records the source, service IDs, budget and archive identity. No source/JAR or past result is silently substituted.

Offline verification, including every original file hash, all private model counters and the frozen decision:

```text
python benchmarks/ais-breakwater-transfer-v1/reproduce.py
```

Restore all original file paths into a new directory if needed:

```text
python benchmarks/ais-breakwater-transfer-v1/reproduce.py --extract restored-transfer-v1
```

No matches run during reproduction. Official first places are primary; missing win callbacks remain in raw rows and are disclosed in the report. A later strategy must have a new source/registration/population; these positive secondary observations cannot waive the rejected primary or serve as fresh confirmation.
