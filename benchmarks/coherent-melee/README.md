# Coherent melee run (registered reject)

The protocol and results are in the [report](../../docs/improvements/coherent-melee.md).

## Arms

| Arm | Source | JAR |
| --- | --- | --- |
| Candidate | `c4b2cae035e34605b8e7bd48dc62956b70a1d388` | `351a006829dfeb0a43850f0d54de7a1442296d19518d68ee8b17d8f9e32c16a1` |
| Baseline | `e4e4c94a45c770a494f6a60f6b98456e215bee42` | `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6` |

## Contents

- `raw/experiment.json`: all 10 melee battles, 16 Retcon battles and 6 sample battles, with checks, decision and hashes.
- `raw/preregistered-plan.md`: the registered plan, byte-identical to the blob at `c4b2cae`.
- Per-battle official results (`battle.json`), telemetry (`metrics.jsonl`), runner logs and Retcon native exports.
- `files.sha256.json`: the integrity manifest.

## Provenance

The evidence comes from [run 41645248808](https://github.com/stephanelee/robocode/actions/runs/41645248808), job 125948759531, artifact 13237120936 (zip SHA256 `db49537c4b5ceacb69eccde26b17a45507c91863c01c9fbcaee50b930221b603`). A no-match transfer job imported it and removed only the per-battle copies of the bot and opponent bundles.
