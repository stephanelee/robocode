# Prism initial-profile screen: rejected

The sole registered screen [run41863990617](https://github.com/stephanelee/robocode/actions/runs/41863990617), [PR105](https://github.com/stephanelee/robocode/pull/105), completed 13 games / 200 official rounds at execution `0ac58d50ddaf041d0103a125b745fd4f2709a90c`. The canonical [decision](raw/build/prism-screen/decision.json) is **reject**, solely because Diamond Port equal-game mean score share 28.608206% did not exceed 40%. No confirmation, replacement or retune was run. Production AISurvivor is unchanged; Prism remains an unpromoted standalone experimental reference.

| Pinned opponent | Games / rounds | Mean game score share | Prism official first places |
|---|---:|---:|---:|
| Python Diamond Port 1.8.28 | 3 / 60 | 28.608206% | 8 / 60 |
| Retcon 0.2.0 | 3 / 60 | 94.327397% | 54 / 60 |
| Python Adaptive Prime 1.0 | 3 / 60 | 69.294953% | 43 / 60 |
| Official Spin Bot 1.0 | 2 / 10 | 100% | 10 / 10 |
| Official Walls 1.0 | 2 / 10 | 100% | 10 / 10 |

Official first places are the runner's field, not an exclusive win reconstruction: Retcon game1 reports Prism19 and Retcon2 across20 rounds, consistent with ties. All engineering gates passed: zero Prism recorded skips, collisions and subsystem failures across154,131 reported movement decisions. The registered skip gate was below0.5% per opponent, not a retrospectively substituted zero-skip rule. All other registered score gates passed. This is a screen, not baseline A/B evidence or fresh superiority confirmation; no general-strength or original Java Diamond claim follows.

## Preservation and reproduction

- [Manifest](manifest.json): every preserved raw/provenance/source payload has a SHA-256 and byte count.
- [Audit summary](audit-summary.json): official totals, cumulative telemetry, schedule and exact frozen-rule reproduction.
- [Raw original artifact files](raw/): all13 results, attempt metadata, runner logs and telemetry, the consumed receipt and actual runtime.
- [Exact execution source/tooling](source/): workflow, immutable approval plan, protocol, harness, runtime loaders and opponent locks, with Git blob provenance.
- [Job log](job.log) and [launch history](registration-events.json).
- Original Actions artifact13343025344 ZIP SHA-256 `26489fff821c0237536c0442f6c674e845dfe2756ac208e37cceb7203ed46a8f`; all extracted bytes retained. No duplicated candidate JAR is needed: the measured committed artifact remains `experiments/Prism/Prism.jar` SHA-256 `8c045f80d37a3076ced7b337b020d14bf3942c6c2a89152f11021f8ff9e9e7ca`.

Run `python -m unittest discover -s tools -p test_prism_screen_evidence.py` from the repository root. It verifies every preserved hash and reconstructs official/cumulative metrics and the unchanged rejection using the archived executed gate function. It runs no bots.

## Limits and chronology

There are **199 per-round telemetry snapshots**, not200: `diamond-3/prism-telemetry.jsonl` lacks round15. Its final cumulative game record has round count20 and the official result has20 rounds. The missing row remains missing, with unknown cause; no learning-curve point is invented. The frozen harness requires final game telemetry and official round completeness, so this gap does not alter its canonical rejection.

The registered runner did not retain `.battle.gz` recordings or opponent-private skipped-turn events. Zero skips refers to Prism's own telemetry only. Prism does not read/write persistent model files;13 separate staged launcher paths/process invocations and their registered seeds support the cold-game setup. There are no endpoint model files to audit or claim. Latency quantiles and maxima summarize a bounded rolling4096-sample ring, not all lifetime turns. No private opponent inputs were supplied to Prism.

One initial manual dispatch returned HTTP404 before creating any run, receipt or game. A label-trigger route was independently reviewed. Before any launch, two acquisition defects were corrected: Adaptive sourcePath came from its immutable lock, and the native pinned code/venv is prepared once while each game still stages and starts fresh bot processes. Earlier intermediate clearances were explicitly withdrawn before games. The final separate [full-acquisition clearance6635424606](https://github.com/stephanelee/robocode/pull/105#issuecomment-6635424606) binds `e4d51c5`; execution `0ac58d5` changes only approvedHead/approvedReview. The semantic guard passes. Exactly one label launched this attempt, and the durable receipt `refs/tags/prism-screen-v1-8c045f80d37a3076` points at that execution head and is retained permanently. The spent acquisition workflow is retired; its exact executed bytes remain archived here. No criterion or candidate changed after outcomes.

Runtime recorded in the original artifact: Temurin25.0.5+8, Python3.13.17, Ubuntu26.04 runner. Java17 build/39JUnit tests/source-JAR parity and exact-head clearance succeeded before matches. GUI7654 was untouched.
