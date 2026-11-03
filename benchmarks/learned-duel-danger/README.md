# Rejected learned-danger screen

[Report and prospective protocol](../../docs/improvements/learned-duel-danger.md) · [PR #54](https://github.com/stephanelee/robocode/pull/54) · [single actual run](https://github.com/stephanelee/robocode/actions/runs/41652803853).

All 60 screen rounds are retained, with six official games, six private cold finite counter-10 model endpoints, metrics/logs, two native Retcon exports, exact source/runtime/package provenance and both bot asset hashes. `experiment.json` is the original canonical artifact. `registered-plan.md` and `registered-protocol.json` are the immutable `fe286963` Git blobs. `files.sha256.json` verifies all archived bytes; models are research evidence and are never loaded from the default bot path.

The sole failed screen guard was Retcon's relative score difference −16.014917 pp below the −5 pp floor. The 270 conditional rounds were not run. Accepted source `12348fb` / JAR `21c5c3d2…` was restored. `candidate.patch` preserves the rejected source/test change; candidate JAR `b2da09ef…` is identified by its measured source and hashes without duplicating it here. Original runnable bundles remain in artifact `13237574013`.

`source/` preserves executed tooling, reviewed/executed workflow bytes and the Python opponent lock. Reviewed/executed workflow hashes are recorded in `execution-provenance.json`; equality after CRLF→LF normalization is checked. `all-opponent-file-hashes.json` preserves per-game packaged source hashes, while licenses and descriptors remain local. Full source remains independently pinned by the immutable upstream lock. No new games are needed to reproduce the decision.
