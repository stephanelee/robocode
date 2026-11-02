# Rejected aim-corridor minimum-risk experiment

The protocol, results and decision are in the [report](../../docs/improvements/aim-corridor.md). The measured candidate source/JAR is commit `b2f4ccbe69cf2cdecb103b9d501f9d3f934e56e5`, JAR `a804716b881710488c980e64b84428346060dc28a07554417e57eba8bfd05bac`. The baseline is `cea19c8c451149cd5377db79972c591489a58ef1`, JAR `1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a`. Retcon is pinned at `28981c7a469ac064194b41001cd7183ec5ecf189`, and the engine and samples are Tank Royale 1.4.0.

- `raw/experiment.json`: driver output, including the final decision, Retcon mechanism aggregates and all sample battles.
- `raw/primary/comparison.json`: canonical `compare.py` output. It contains all six Retcon battles, the thresholds, bundle/opponent/runtime hashes and the plan hash.
- `raw/primary/preregistered-plan.md`: the registered plan blob, byte-identical to `docs/improvements/aim-corridor.md` at `b2f4ccb`.
- `raw/**/battle.json`, `metrics.jsonl`, `runner.log`, `retcon-stats/*`: per-battle official results, AIS per-round telemetry, runner logs and unmodified Retcon native exports with their capture manifests.
- `files.sha256.json`: SHA256 of all 51 retained files.
- `candidate.patch`: the rejected source, test and JAR change relative to the baseline.

The retained files came from Actions artifact 13183485154 of [run 41532956899](https://github.com/stephanelee/robocode/actions/runs/41532956899). A no-match transfer job imported the artifact. Per-battle copies of the two bot bundles and the Retcon source tree were then removed: they duplicate pinned or hashed inputs (`bundleSHA256` and `opponentProvenance` in `comparison.json`). The complete original remains in the Actions artifact until it expires.

To reproduce, check out `b2f4ccb`. Build with Java 17 (`./gradlew clean build`), make the baseline available with `git worktree add <dir> cea19c8`, then run with JDK 25:

```sh
python tools/aim_corridor_experiment.py --previous-bot-dir <dir>/AISurvivor \
  --previous-source cea19c8c451149cd5377db79972c591489a58ef1 \
  --expected-previous-jar 1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a \
  --candidate-source b2f4ccbe69cf2cdecb103b9d501f9d3f934e56e5 \
  --plan-commit b2f4ccbe69cf2cdecb103b9d501f9d3f934e56e5 \
  --java /path/to/jdk25/bin/java --output build/aim-corridor-repeat
```

A repeat is a new, unpaired sample, not a replacement for this result.
