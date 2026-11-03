# A/A Retcon noise calibration

See the [report](../../docs/aa-calibration.md) for the registered plan and its results.

## Run details

| Item | Value |
| --- | --- |
| Measured code (both arms) | master `241156f7d6b78c4ca2365859884b1486c9f41194` |
| JAR | `5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883` |
| Plan and driver commit | `79359f27dc6fd011d472030bdd916c565d564c26` |
| Retcon | `28981c7a469ac064194b41001cd7183ec5ecf189` |
| Engine | Tank Royale 1.4.0 |

## Files

- `raw/calibration.json`: all 20 battles, the preregistered summary, bundle/opponent/runtime hashes, and the runtime OS.
- `raw/preregistered-plan.md`: the registered plan, byte-identical to `docs/aa-calibration.md` at `79359f2`.
- `raw/block-*/{a,b}/Retcon-fixed-minrisk/`:
  - `battle.json`: official results
  - `metrics.jsonl`: AIS per-round telemetry
  - `runner.log`: runner log
  - `retcon-stats/*`: unmodified Retcon native exports
- `files.sha256.json`: hashes of all 102 retained files, plus the source run and artifact.

## Provenance

The files come from [run 41548798991](https://github.com/stephanelee/robocode/actions/runs/41548798991), Actions artifact 13189987653. A no-match transfer job imported them and removed per-battle bot/opponent bundle copies before committing; those bundles are hashed in `calibration.json`.

## Reproduce

Run this from `79359f2` after a Java 17 build, using JDK 25:

```sh
python tools/aa_calibration.py --bot-dir AISurvivor \
  --expected-jar 5ebfa64adf38ab227766b97f7f68e82aefb20251ff8d5f30575bf6b69e67c883 \
  --source 79359f27dc6fd011d472030bdd916c565d564c26 --plan docs/aa-calibration.md \
  --plan-commit 79359f27dc6fd011d472030bdd916c565d564c26 --blocks 10 --rounds 20 --seed-base 20261301 \
  --java /path/to/jdk25/bin/java --output build/aa-calibration-repeat
```
