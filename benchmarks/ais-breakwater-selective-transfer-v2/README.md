# Rejected AIS Breakwater movement plus selective-fire transfer

The sole [registered run41730684973](https://github.com/stephanelee/robocode/actions/runs/41730684973) completed all90screen rounds and was rejected by five unchanged guards. No540-round confirmation was run. [Full protocol/results/limitations](../../docs/improvements/ais-breakwater-selective-transfer-v2.md) - [PR74](https://github.com/stephanelee/robocode/pull/74).

Breakwater share improved to35.714%/3wins but failed the winning requirements; Retcon lost9.546pp, Adaptive lost9.809pp and Diamond reached0%share. The guard passed only21 of43383 eligible proposals across the five candidate games; Diamond/self fired zero shots. This is a negative whole-candidate result; production source/JARb41b was restored. V1 remains a separate preserved rejection.

`experiment.json`, registered plan/protocol and readable raw numerical records/logs preserve original bytes. All956original artifact files, including10private AIS checkpoints, two Breakwater endpoints, two Retcon exports and every staged asset/cache, are retained losslessly in `original-payloads.zip`;550unique payloads are keyed by SHA256. `original-files.json` maps every original path to bytes/hash. This preserves content rather than original ZIP metadata. Original service ZIP SHA256 `f953c44b769d89f0721916c5ac87a83e650afaa4a2af389fcc4ceea579e7a0e0` matched artifact13276183049 before import and was independently audited.

`source/` retains exact execution9c76456 drivers, tests, dependencies, locks, workflow and original job log. `candidate.patch.gz` preserves the binary-capable production source/tests/JAR patch from accepted9b1624 to measured1409f164; the exactaf94 candidate JAR is also in original payloads. `artifact-metadata.json` and `provenance.json` bind service/source/budget identities. Official first places remain primary; raw callback discrepancies and starvation are disclosed in the report.

Verify every original payload hash, cold counters/finite models, raw summaries/selectivity counters and frozen rejection offline:

```sh
python benchmarks/ais-breakwater-selective-transfer-v2/reproduce.py
```

Restore all original paths into a new directory if needed:

```sh
python benchmarks/ais-breakwater-selective-transfer-v2/reproduce.py --extract restored-transfer-v2
```

Reproduction runs zero games. `files.sha256.json` covers every retained file except itself. No old checkpoint, pooled score, new threshold or rescue run is used.
