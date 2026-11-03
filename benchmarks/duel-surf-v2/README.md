# Retained engine-conformant duel movement (v2)

The protocol, results and decision are in the [report](../../docs/improvements/duel-surf-v2.md).

| Item | Value |
| --- | --- |
| Candidate source | `8db129f3d405d12ea14fe662e7e7adf5366e3f69` |
| Candidate JAR | `d047bf99325d2ad73aedd87eac6a4baa1677170afff0c86b948f03ea93390e1d` |
| Baseline source | `dccc19d1d37ae07ce37ea5d300659d45395bfc1f` |
| Baseline JAR | `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883` |
| Retcon | `28981c7a469ac064194b41001cd7183ec5ecf189` |
| Engine and sample bots | Tank Royale 1.4.0 |

## Files

- `raw/experiment.json`: all 32 Retcon battles, 6 sample battles, the Student-t decisions, mechanism aggregates, and bundle/opponent/runtime hashes.
- `raw/preregistered-plan.md`: the registered plan, byte-identical to the Git blob (LF) of `docs/improvements/duel-surf-v2.md` at `8db129f`.
- Per-battle files, under `raw/{primary,online}/block-*/{previous,candidate}/` and `raw/samples/*/`:
  - `battle.json`
  - `metrics.jsonl`
  - `runner.log`
  - `retcon-stats/*`
- Online checkpoints: the 16 `model.dat` files from the online stage. Their hashes match `experiment.json`.
- `files.sha256.json`: SHA256 of all 196 retained files, plus the source run and artifact.

## Provenance

The source is [run 41556987786](https://github.com/stephanelee/robocode/actions/runs/41556987786), artifact 13194444591. A no-match transfer job imported it. Before committing, that job removed only the per-battle copies of the bot and opponent bundles, which are hashed in `experiment.json`.

## Reproduce

Check out `8db129f`, build with Java 17, and run `git worktree add <dir> dccc19d`. Then, with JDK 25:

```sh
python tools/block_experiment.py --previous-bot-dir <dir>/AISurvivor \
  --expected-previous-jar 5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883 \
  --previous-source dccc19d1d37ae07ce37ea5d300659d45395bfc1f \
  --candidate-source 8db129f3d405d12ea14fe662e7e7adf5366e3f69 --plan docs/improvements/duel-surf-v2.md \
  --plan-commit 8db129f3d405d12ea14fe662e7e7adf5366e3f69 --blocks 8 --rounds 20 --seed-base 20261501 \
  --sample-seed 20261509 --bound t --java /path/to/jdk25/bin/java --output build/duel-surf-v2-repeat
```

A repeat is a new, unpaired sample, not a fresh confirmation.
