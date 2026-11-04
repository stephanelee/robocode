# Wait for sealed Challenger outputs before closing the runner

PR68 writes a completed round's model and telemetry together after the Bot API's terminal event drain: at the next RoundStarted, or GameEnded for the last round. Official battle results can become available before that final callback finishes. Closing the runner immediately can therefore interrupt final experimental output. This integration change adds the completion wait explicitly required by the [lifecycle repair](challenger-round-lifecycle.md); it does not repair or rerun PR65's invalid attempt.

The existing BenchmarkBattle adapter now opts into a bounded three-second wait when both CHALLENGER_MODEL_PATH and CHALLENGER_METRICS_PATH are configured. Both paths must be distinct and absolute. Before closing the battle handle, it waits for the final model round count, exactly the requested number of nonblank telemetry rows, a complete final newline, and the final row's matching round count. Missing/incomplete outputs fail after the bound. Model and telemetry reads are capped at 2 MB and 32 MB respectively. No configured Challenger paths leaves other battles unchanged; configuring just one path is an error.

This is a file-readiness barrier, not a substitute for the experiment's strict JSON/schema, identity, row, cohort and provenance validators. It does not validate gameplay or turn a descriptive result into valid evidence. Previously frozen runners and evidence remain byte-for-byte unchanged. Any future measured candidate must freeze this runner version in its new protocol and use fresh game identities; this PR launches no games.

Validation: the real BenchmarkBattle source compiles with Java17 against the existing hash-pinned official Runner1.4.0. Seven synthetic checks cover immediate completion, delayed final-row arrival after the model, partial append, extra rows, stale model, missing model, and aliased paths. A standalone fixture calls the actual readiness method; it never starts a runner/server/bot. The ordinary Java17 CI job runs this fixture. Production AIS source/JAR, experimental controller/JAR and opponent/runtime pins are unchanged. New battle rounds: **zero**; no competitive improvement claim.

Reproduce from the repository root with Java17 selected:

```text
python -m unittest discover -s tools -p test_challenger_completion.py -v
```

The first fixture run downloads the existing SHA256-pinned Runner1.4.0 asset into build/completion-test-cache. No additional library, battle runtime or game data is introduced.
