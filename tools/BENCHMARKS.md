# Battle experiments

Requires Python 3.10+ and Java 17 on PATH. No pip packages or GUI are needed.

```sh
./gradlew clean build
python -m unittest discover -s tools -p 'test_*.py' -v
python tools/benchmark.py --smoke
python tools/benchmark.py --rounds 20 --training-rounds 100 --require-metrics --output build/experiment-001
```

The official Tank Royale 1.4.0 Runner embeds the server and booter. Both its release JAR and the sample-bot ZIP are downloaded to `.benchmark-cache` and checked against literal SHA256 hashes. Each experiment gets isolated bot working directories and model files; the output directory must be empty. `--java /path/to/java` selects a JDK explicitly.

The short smoke runs one cold evaluation round against Target. The full experiment runs Target, SpinBot, Walls, and a four-bot melee scenario, comparing minimum-risk movement, a fresh seeded DQN, and a DQN evaluated after a separate training phase. Only the `train` phase may create/change its checkpoint. The trained checkpoint is shared only with the subsequent trained evaluation for the same scenario. Evaluation never uses a checkpoint from your regular bot folder.

The bot receives `AISURVIVOR_MODE`, `AISURVIVOR_SEED`, `AISURVIVOR_MODEL_PATH`, `AISURVIVOR_ACTION`, and `AISURVIVOR_METRICS_PATH`. Use `--require-metrics` once the learning/configuration PR is integrated to require exactly one telemetry row per round. With older bot builds, absent telemetry is reported as missing rather than zero.

Open `dashboard.html` locally to filter the comparison table. `results.json` records the source commit (GitHub SHA or local Git HEAD when available), asset hashes, the bot JAR hash, training budget, seed, full opponent results, and bot telemetry aggregates. `results.csv` is suitable for spreadsheets. Raw battle results, per-round telemetry, and runner logs remain in each scenario directory. Rebuild before running locally: a source commit alone does not prove an existing JAR was built from that source.

Metrics use actual fired bullets and hit events; hit rate aggregates hits divided by shots across rounds. Score, rank, survival score, and round wins come from the official runner. Training rows describe the training budget and should not be compared with evaluation rows as equal-budget results.

## Interpretation

These are reference experiments, not proof that DQN improves performance. The bot RNG is seeded; starting positions and engine scheduling are not seeded. Repeat with multiple bot seeds and larger round budgets. Compare score shares and win rates within matching opponent scenarios and report uncertainty before claiming stronger play. The default small sample and manually triggered Actions reference run primarily verify the whole integration.

The `Battle benchmark` workflow runs a smoke battle on every PR/push. Run it manually with `full=true` for the four-scenario reference experiment. Download its `battle-results` artifact to view the standalone dashboard; no hosting is required.

Each Java battle has a deadline and uses the official runner's resource cleanup. The Python wrapper also bounds the entire process and terminates its POSIX process group on timeout. On Windows, it uses `taskkill /PID <owned-pid> /T /F` to terminate that runner's process tree because forced termination does not execute JVM shutdown hooks. Actions uses Linux.
