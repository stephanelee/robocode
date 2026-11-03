# Rejected two-way wall smoothing experiment

The protocol, results and decision are in the [report](../../docs/improvements/wall-search.md).

| Item | Value |
| --- | --- |
| Measured candidate source | `f250c95d093efa245468a4a9fae34c38679e2587` |
| Candidate JAR | `9981c06b716db4ce6eb5dbdeef63fbbd0b29add8df76cfd2985dcfee9bc66f6f` |
| Baseline source | `241156f7d6b78c4ca2365859884b1486c9f41194` |
| Baseline JAR | `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883` |
| Retcon | `28981c7a469ac064194b41001cd7183ec5ecf189` |
| Engine and sample bots | Tank Royale 1.4.0 |

## Files

- `raw/experiment.json`: the final decision, Retcon mechanism aggregates, and all sample battles.
- `raw/primary/comparison.json`: canonical output from `compare.py`, covering all six Retcon battles, thresholds, and bundle/opponent/runtime hashes.
- `raw/primary/preregistered-plan.md`: the registered plan, byte-identical to `docs/improvements/wall-search.md` at `f250c95`.
- `raw/**/battle.json`, `metrics.jsonl`, `runner.log`, `retcon-stats/*`: per-battle official results, AISurvivor per-round telemetry, runner logs, and the unmodified Retcon native exports.
- `files.sha256.json`: SHA256 of all 51 retained files, plus the source run/artifact identity.
- `candidate.patch`: the rejected source/test/JAR change relative to the baseline.

## How the evidence was collected

[Run 41546243537](https://github.com/stephanelee/robocode/actions/runs/41546243537) produced Actions artifact 13188028349. A no-match transfer job imported it. Before committing, that job removed the per-battle copies of the bot bundles and the Retcon source tree. Those copies are pinned and hashed in `comparison.json`. The complete original remains in the artifact until it expires.

## Reproduce

1. Check out `f250c95`.
2. Build with Java 17: `./gradlew clean build`.
3. Create a baseline worktree: `git worktree add <dir> 241156f`.
4. Run with JDK 25:

```sh
python tools/aim_corridor_experiment.py --previous-bot-dir <dir>/AISurvivor \
  --previous-source 241156f7d6b78c4ca2365859884b1486c9f41194 \
  --expected-previous-jar 5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883 \
  --candidate-source f250c95d093efa245468a4a9fae34c38679e2587 --plan-commit f250c95d093efa245468a4a9fae34c38679e2587 \
  --plan-path docs/improvements/wall-search.md --seeds 20261221,20261222,20261223 --sample-seed 20261224 --primary-category correctness \
  --java /path/to/jdk25/bin/java --output build/wall-search-repeat
```

A repeat is a new, unpaired sample. It does not replace this result.
