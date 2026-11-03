# Retained shot-reactive duel movement experiment

The protocol, results and decision are in the [report](../../docs/improvements/duel-surf.md).

## Provenance

| Item | Value |
| --- | --- |
| Retained candidate source | `5f5ff460eaad190a6bc3dec507f6eee732474164` |
| Candidate JAR | `1a588ead187a7fa90fbf1468cb383f5f75696b6fbfc4fcef9621b05c09d59d02` |
| Baseline source | `241156f7d6b78c4ca2365859884b1486c9f41194` |
| Baseline JAR | `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883` |
| Retcon | `28981c7a469ac064194b41001cd7183ec5ecf189` |
| Engine and samples | Tank Royale 1.4.0 |

## Files

- `raw/experiment.json`: all 32 Retcon battles, the 6 sample battles, the decisions, the Retcon mechanism aggregates, and bundle/opponent/runtime hashes.
- `raw/preregistered-plan.md`: byte-identical to the Git blob (LF) of `docs/improvements/duel-surf.md` at `5f5ff46`.
- `raw/online/block-*/*/Retcon-train/model.dat`: the 16 online-training endpoint checkpoints, added after the review from the same artifact. Their SHA256 values match `modelSHA256` in `experiment.json`.
- Per-battle files under `raw/{primary,online}/block-*/{previous,candidate}/` and `raw/samples/*/`:
  - `battle.json`: official results.
  - `metrics.jsonl`: AIS telemetry.
  - `runner.log`: runner log.
  - `retcon-stats/*`: unmodified Retcon native exports.
- `files.sha256.json`: SHA256 of all 196 retained files (180 original plus 16 models), plus the source run and artifact.

## How the evidence was collected

Source: [run 41551240407](https://github.com/stephanelee/robocode/actions/runs/41551240407), artifact 13191578758. A no-match transfer job imported the artifact and removed per-battle bot and opponent bundle copies (hashed in `experiment.json`) before committing.

## Reproduce

From `5f5ff46`, after a Java 17 build and `git worktree add <dir> 241156f`, run with JDK 25:

```sh
python tools/block_experiment.py --previous-bot-dir <dir>/AISurvivor \
  --expected-previous-jar 5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883 \
  --previous-source 241156f7d6b78c4ca2365859884b1486c9f41194 \
  --candidate-source 5f5ff460eaad190a6bc3dec507f6eee732474164 --plan docs/improvements/duel-surf.md \
  --plan-commit 5f5ff460eaad190a6bc3dec507f6eee732474164 --blocks 8 --rounds 20 --seed-base 20261401 \
  --sample-seed 20261409 --java /path/to/jdk25/bin/java --output build/duel-surf-repeat
```

A repeat produces a new, unpaired sample. It is not a replacement result and not a fresh confirmation.
