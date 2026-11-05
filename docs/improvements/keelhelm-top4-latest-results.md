# Latest published KeelHelm: fresh top-four melee results

**Diamond Port wins this snapshot. KeelHelm 0.2.1 finishes third by aggregate score and second by round wins.** Four fresh cold games × 20 rounds, 80 rounds total, ran sequentially on minimind. No gameplay code or policy changed.

| Rank by score | Bot | Version | Total score | Round wins | Mean game score share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | Diamond Port | 1.8.28 | 14,969 | 32 | 32.73% |
| 2 | Prism | 0.2.0 | 14,598 | 16 | 31.91% |
| 3 | KeelHelm | 0.2.1 | 13,378 | 29 | 29.24% |
| 4 | Keel | 0.1.0 | 2,806 | 3 | 6.13% |

KeelHelm has the highest aggregate survival points (7,950), but less bullet damage than Diamond Port and Prism. Score components are descriptive observations, not evidence that a specific code change caused the result.

| Bot | Survival points | Bullet damage |
| --- | ---: | ---: |
| Diamond Port | 7,500 | 5,958 |
| Prism | 7,000 | 6,778 |
| KeelHelm | 7,950 | 4,103 |
| Keel | 1,550 | 1,083 |

## Per-game official scores

| Game | Diamond Port | Prism | KeelHelm | Keel | Winner by score |
| --- | ---: | ---: | ---: | ---: | --- |
| melee-1 | 3,948 | 3,768 | 3,170 | 615 | Diamond Port |
| melee-2 | 4,610 | 3,512 | 2,906 | 361 | Diamond Port |
| melee-3 | 3,436 | 3,693 | 3,609 | 669 | Prism |
| melee-4 | 2,975 | 3,625 | 3,693 | 1,161 | KeelHelm |

## Per-game round wins

| Game | Diamond Port | Prism | KeelHelm | Keel |
| --- | ---: | ---: | ---: | ---: |
| melee-1 | 9 | 5 | 6 | 0 |
| melee-2 | 12 | 4 | 4 | 0 |
| melee-3 | 8 | 4 | 8 | 0 |
| melee-4 | 3 | 3 | 11 | 3 |

## Registered scope and validation

Decision: accept as valid descriptive benchmark evidence; no gameplay promotion or rejection. KeelHelm is the latest normal published 0.2.1 bundle at main 255f3230e4854a1db00e0aa214030db5f7c28f2d, JAR 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. All 40 published project files match that main source. Rejected 0.4 research and all experimental masks were excluded. Other entrants retain tournament pins: Diamond Port 1.8.28, Keel 0.1.0, Prism 0.2.0.

Each entrant occupied every launch slot once. Each game used a fresh process/home, bot-side seeds 2026112100..2026112103, pinned JDK 25 / runtime/runner/Python and unchanged 100000 us deadline/heaps/GC. Bot seeds do not pair or seed engine starting positions. No Actions gameplay, warmup game, retry, replacement, tuning, early score stop or historical pooling.

All four original recordings have valid CRC, exact setup/participants, contiguous ticks, 20 round endpoints and matching final official scores. Independent audit verifies 7,014 frozen inputs and 464 staged copies, including 352 shared bot-core/license copies. Zero own skipped turns or partial games; all admission/resource/teardown checks pass. Minimum sampled memory availability 57%, no new Swapouts, maximum logged own loop 5 ms (excludes callbacks and termination work). This does not diagnose prior timing failures or guarantee responsiveness.

Java 17 build/tests: 23 passed; committed/rebuilt JAR and every entry match the published hash. Two pre-game schedule/aggregation contracts passed. Java 17 log/XML and all harness/auditor sources are retained.

Fresh cost: four games / 80 completed rounds, zero partial rounds. Previous KeelHelm development/diagnostic accounting 1,910 completed plus three partial rounds; including this snapshot, 1,990 completed plus three historical partial rounds. These physical costs are separate from valid experimental populations and the frozen completed tournament.

Four cold games are a bounded snapshot. Aggregate engine score determines this table’s rank; mean share weights games equally. Do not infer historical improvement, statistical significance or general competitive strength from this unpaired cohort. The accepted source/JAR and official tournament standings remain unchanged.

All original recordings, staged inputs, source/JAR/execution snapshots, emitted files, telemetry, resources, invocations, cleanup and independent audit sources are retained privately in robocode-keelhelm-top4/2026-11-05-latest/full-artifacts.tar.gz (104,732,578bytes), SHA256 5e3ab661a7d76484b1a8e735a2edef5fad1b562f355341e7e5880d4c0070a67f.

[Per-game data](../../benchmarks/keelhelm-top4-latest/results.json), [independent audit](../../benchmarks/keelhelm-top4-latest/independent-audit.json), [attestation](../../benchmarks/keelhelm-top4-latest/audit-attestation.json), [source/runtime provenance](../../benchmarks/keelhelm-top4-latest/provenance.json), [build/parity validation](../../benchmarks/keelhelm-top4-latest/build-validation.json), [archive proof](../../benchmarks/keelhelm-top4-latest/artifact-proof.json).

[Prospective protocol](keelhelm-top4-latest-protocol.md), [issue127](https://github.com/stephanelee/robocode/issues/127), [PR128](https://github.com/stephanelee/robocode/pull/128).
