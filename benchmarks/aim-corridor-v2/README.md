# Rejected aim-corridor v2 experiment (corridor + two-way wall smoothing)

The protocol, results and decision are in the [report](../../docs/improvements/aim-corridor-v2.md).

| Item | Value |
| --- | --- |
| Measured candidate source | `15d2b02795f6353c5380483f30769714f8b0b32d` |
| Candidate JAR | `3619be973d3417111703df3d69622a8447285abda2e7a866d6c1e49f2f973ebc` |
| Baseline source | `116c47ccc77e85f9c8f570c554f5e626a4ed76d4` |
| Baseline JAR | `1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a` |
| Retcon | `28981c7a469ac064194b41001cd7183ec5ecf189` |
| Engine and sample bots | Tank Royale 1.4.0 |

## Files

- `raw/experiment.json`: the final decision, Retcon mechanism aggregates, and all sample battles.
- `raw/primary/comparison.json`: canonical output from `compare.py`, covering all six Retcon battles, thresholds, and bundle/opponent/runtime hashes.
- `raw/primary/preregistered-plan.md`: the registered plan, byte-identical to `docs/improvements/aim-corridor-v2.md` at `15d2b02`.
- `raw/**/battle.json`, `metrics.jsonl`, `runner.log`, `retcon-stats/*`: per-battle official results, AISurvivor per-round telemetry, runner logs, and the unmodified Retcon native exports.
- `files.sha256.json`: SHA256 of all 51 retained files, plus the source run/artifact identity.
- `candidate.patch`: the rejected source/test/JAR change relative to the baseline.

## How the evidence was collected

[Run 41542518773](https://github.com/stephanelee/robocode/actions/runs/41542518773) produced Actions artifact 13187208307. A no-match transfer job imported it. Before committing, that job removed the per-battle copies of the bot bundles and the Retcon source tree. Those copies are pinned and hashed in `comparison.json`. The complete original remains in the artifact until it expires.

## Reproduce

1. Check out `15d2b02`.
2. Build with Java 17: `./gradlew clean build`.
3. Create a baseline worktree: `git worktree add <dir> 116c47c`.
4. Run with JDK 25:

```sh
python tools/aim_corridor_experiment.py --previous-bot-dir <dir>/AISurvivor \
  --previous-source 116c47ccc77e85f9c8f570c554f5e626a4ed76d4 \
  --expected-previous-jar 1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a \
  --candidate-source 15d2b02795f6353c5380483f30769714f8b0b32d --plan-commit 15d2b02795f6353c5380483f30769714f8b0b32d \
  --plan-path docs/improvements/aim-corridor-v2.md --seeds 20261211,20261212,20261213 --sample-seed 20261214 \
  --java /path/to/jdk25/bin/java --output build/aim-corridor-v2-repeat
```

A repeat is a new, unpaired sample. It does not replace this result.
