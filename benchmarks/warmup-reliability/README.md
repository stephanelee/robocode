# Warm-up reliability check (retain-neutral)

The protocol, results and decision are in the [report](../../docs/improvements/warmup-reliability.md).

## Provenance

| Item | Value |
| --- | --- |
| Candidate | `f3db2d4ee9c170d719909e8cb7dd83513130266a` |
| Candidate JAR | `21c5c3d2cb4007e3e0603556af313df7f5717732e384f039b005e9e72b2daecd` |
| Baseline | `e4e4c94`, JAR `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6` |
| Retcon | `28981c7a469ac064194b41001cd7183ec5ecf189` |
| Engine and samples | Tank Royale 1.4.0 |

## Contents

- `raw/experiment.json` contains every game, with per-round skipped turns, the candidate's per-round `maxTurnMicros` and `slowTurns`, scores, summaries, the decision, and bundle, opponent and runtime hashes.
- `raw/preregistered-plan.md` is byte-identical to the registered blob at `f3db2d4`.
- Per game, the evidence keeps `battle.json`, `metrics.jsonl`, `runner.log`, the Retcon native exports, and the fresh-model endpoint checkpoints.
- `files.sha256.json` lists SHA256 for every retained file, plus the source run and artifact.

The source is [run 41638002007](https://github.com/stephanelee/robocode/actions/runs/41638002007), artifact 13233215708. A no-match transfer job imported it and removed only the per-battle bot and opponent bundle copies, which are hashed in `experiment.json`.

## Reproduce

From `f3db2d4`, after a Java 17 build and `git worktree add <dir> e4e4c94`, run with JDK 25:

```sh
python tools/reliability_experiment.py --previous-bot-dir <dir>/AISurvivor \
  --expected-previous-jar 9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6 \
  --previous-source "$(git rev-parse e4e4c94)" --candidate-source f3db2d4ee9c170d719909e8cb7dd83513130266a \
  --plan docs/improvements/warmup-reliability.md --plan-commit f3db2d4ee9c170d719909e8cb7dd83513130266a \
  --games 15 --rounds 2 --seed-base 20261601 --java /path/to/jdk25/bin/java --output build/warmup-repeat
```
