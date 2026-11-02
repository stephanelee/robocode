# Benchmark methodology

## Recommended performance direction: Retcon 1v1

Use the [Retcon benchmark workflow](https://github.com/stephanelee/robocode/actions/workflows/retcon.yml) with manual `full=true`, or select `--suite retcon` explicitly. Build/test AISurvivor with Java 17, then run the unchanged Retcon source launcher with JDK 25:

```sh
./gradlew clean build
python tools/benchmark.py --suite retcon --java /path/to/jdk25/bin/java --rounds 5 --training-rounds 20 --timeout 900 --require-metrics --output build/retcon-evaluation
```

See the pinned-opponent details below and [completed measured cycles](improvements.md). These small, in-roster experiments do not establish general playing strength. The default CLI sample roster remains backward compatible but is intended for regression/integration checks.

## Legacy sample regression/integration suite

Requires Python 3.10+ and Java 17 on PATH. No pip packages or GUI are needed.

```sh
./gradlew clean build
python -m unittest discover -s tools -p 'test_*.py' -v
python tools/benchmark.py --smoke
python tools/benchmark.py --rounds 20 --training-rounds 100 --require-metrics --output build/experiment-001
```

The official Tank Royale 1.4.0 Runner embeds the server and booter. Both its release JAR and the sample-bot ZIP are downloaded to `.benchmark-cache` and checked against literal SHA256 hashes. Each experiment gets isolated bot working directories and model files; the output directory must be empty. `--java /path/to/java` selects a JDK explicitly.

The short smoke runs one cold evaluation round against Target. The full experiment runs Target, SpinBot, Walls, and a four-bot melee scenario, comparing minimum-risk movement, a fresh seeded DQN, and a DQN evaluated after a separate training phase. Only the `train` phase may create/change its checkpoint. The trained checkpoint is shared only with the subsequent trained evaluation for the same scenario. Evaluation never uses a checkpoint from your regular bot folder.

The bot receives `AISURVIVOR_MODE`, `AISURVIVOR_SEED`, `AISURVIVOR_MODEL_PATH`, `AISURVIVOR_ACTION`, and `AISURVIVOR_METRICS_PATH`. Actions always passes `--require-metrics`: every round must have one unique 1-based telemetry ID with the requested mode, seed, action, checkpoint-loaded state, and training budget. Counter types, finite values, hit-rate consistency, and outcome flags are validated before reports are accepted. The CLI option can be omitted for historical builds that lack telemetry; absent telemetry is then reported as missing rather than zero. Present telemetry is always validated.

Open `dashboard.html` locally to filter the comparison table. `results.json` records the source commit (GitHub SHA or local Git HEAD when available), asset hashes, the bot JAR hash, training budget, seed, full opponent results, and bot telemetry aggregates. `results.csv` is suitable for spreadsheets. Raw battle results, per-round telemetry, and runner logs remain in each scenario directory. Rebuild before running locally: a source commit alone does not prove an existing JAR was built from that source.

Metrics use actual fired bullets and hit events; hit rate aggregates hits divided by shots across rounds. Score, rank, survival score, and round wins come from the official runner. Training rows describe the training budget and should not be compared with evaluation rows as equal-budget results.

## Interpretation

These are reference experiments, not proof that DQN improves performance. The bot RNG is seeded; starting positions and engine scheduling are not seeded. Repeat with multiple bot seeds and larger round budgets. Compare score shares and win rates within matching opponent scenarios and report uncertainty before claiming stronger play. The default small sample and manually triggered Actions reference run primarily verify the whole integration.

The [sample reference report](benchmarks/samples-2026-11-02.md) retains the original sample-roster results. The `Battle benchmark` workflow runs a smoke battle on every PR/push. Run it manually with `full=true` for the four-scenario reference experiment. Download its `battle-results` artifact to view the standalone dashboard; no hosting is required.

Each Java battle has a deadline and uses the official runner's resource cleanup. The Python wrapper also bounds the entire process and terminates its POSIX process group on timeout. On Windows, it uses `taskkill /PID <owned-pid> /T /F` to terminate that runner's process tree because forced termination does not execute JVM shutdown hooks. Actions uses Linux.


## Retcon 0.2.0: stronger 1v1 challenge

```sh
# Build/verify AISurvivor with Java 17 first. Use JDK 25 only for the match runtime.
python tools/benchmark.py --suite retcon --smoke --java /path/to/jdk25/bin/java --require-metrics --output build/retcon-smoke
python tools/benchmark.py --suite retcon --java /path/to/jdk25/bin/java --rounds 5 --training-rounds 20 --timeout 900 --require-metrics --output build/retcon-reference
```

The Retcon suite is **1v1 only**. It downloads only the 94 files in the pinned official source tree from `robocode-dev/rumble-bots`, not other opponents or an entire repository archive. `tools/opponents/retcon.lock.json` records commit `28981c7a469ac064194b41001cd7183ec5ecf189`, descriptor version 0.2.0, authors/license metadata, each Git blob identity, and the catalog's SHA256 tree hash. Every file is verified and the complete tree is SHA256-verified before execution. Source files, descriptor, and launchers remain unchanged. The upstream descriptor declares MIT; its author and license fields are preserved in every isolated copy and in result provenance.

Retcon's official launcher uses Java 22+ multi-file source execution and declares Java 25. It needs Bot API 1.3.1; that exact official JAR is SHA256-verified into an isolated cache before launch, avoiding its launcher's fallback download. Actual matches use the pinned Tank Royale 1.4.0 server bundled in the official Runner. Compatibility must be established by a complete match, not merely a source-startup marker.

The dedicated `Retcon benchmark` workflow builds and verifies AISurvivor's committed JAR under Java 17 before installing JDK 25 for matches. Gradle does not run under JDK 25. The existing Java 17 CI and sample-smoke workflow remain independent. A Retcon PR smoke runs one cold evaluation round; manual `full=true` runs five rounds each for fixed minimum-risk, cold DQN, and trained DQN, plus 20 training rounds: 35 rounds total.

Training and evaluation both use Retcon, so this measures **in-roster adaptation**, not held-out generalization. Retcon gets a fresh source/working directory for every invocation; no opponent files or adaptation data carry from training into evaluation. AISurvivor's trained checkpoint is the only shared learned artifact. AISurvivor's RNG is seeded; Retcon randomness, server starting positions, and scheduling are not seeded. Report runtime Java, source commit, opponent source hash, API hash, candidate JAR hash, checkpoint status, and failures with each result.

The sample suite remains selectable with `--suite samples` for regression coverage. Retcon challenge results never replace the original sample baseline. See [the improvement log](improvements.md) for planned measured cycles and their evidence requirements.
