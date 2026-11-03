# Replay-harness non-inferiority run (registered fail; decisions proven identical)

The protocol, results and follow-up investigation are in the [report](../../docs/improvements/replay-harness-parity.md).

## Arms

| Arm | Source | JAR |
| --- | --- | --- |
| Candidate | `22dd537e52bf4234f991abb37bb452eb62d9e1bf` | `28e6322721cea1e4430dccc9b34031ead60513372f0e94bbeaeec1d629e129e9` |
| Baseline | `e4e4c94` | `9679fc5e…` |

## Contents

- `raw/experiment.json`: all 16 Retcon battles and 6 sample battles, with decisions and hashes.
- `raw/preregistered-plan.md`: the registered plan, byte-identical to the blob at `22dd537`.
- Per-battle official results, telemetry, runner logs, Retcon native exports and fresh-model checkpoints.
- `files.sha256.json`: the integrity manifest.

## Provenance

The evidence comes from [run 41642071703](https://github.com/stephanelee/robocode/actions/runs/41642071703), artifact 13234135776. A no-match transfer job imported it and removed only the per-battle copies of the bot and opponent bundles.
