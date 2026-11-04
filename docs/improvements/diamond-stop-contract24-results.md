# Diamond STOP24: infrastructure rejection before activation

The sole acquisition workflow [41843951756](https://github.com/stephanelee/robocode/actions/runs/41843951756) failed its immutable runtime-profile equality gate before activation. **0/24 numeric cases and 0 games ran.** No STOP prediction result was obtained. This is an infrastructure rejection, with no accept/reject conclusion about the predictor or competitive strength. [PR #100](https://github.com/stephanelee/robocode/pull/100) and [issue #99](https://github.com/stephanelee/robocode/issues/99) retain the registered scope.

## Registered comparison and prior review

The [prospective protocol](diamond-stop-contract24.md) fixed 24 ordered one-step cards, including eight STOP cards, zero games and one numeric acquisition attempt. It compares unchanged pinned Diamond prediction with actual SDK controller/clamp and engine pure-helper calls plus source-derived translation/wall composition. No bot correction, live budget or strength gate was opened. The original inputs and tolerances remain unchanged.

Zero-case preparation [41818421304](https://github.com/stephanelee/robocode/actions/runs/41818421304) passed independent audit. Its complete original profile and bridge/dependency provenance were embedded in the write-once freeze. A separate [exact-head independent review](https://github.com/stephanelee/robocode/pull/100#issuecomment-6631478128) cleared only the registered 24-card attempt at head `6007bcdd616e793030c5900e12bdb2e786964c52`, after fresh equality and admission gates. No numerical smoke calls were permitted.

## Failure and measured budget

Run 41843951756, attempt 1, was created at 2026-11-04 14:04:31 UTC and finished at 14:05:43 UTC. Its profile job was skipped; acquire ran from 14:04:36 to 14:05:42 UTC. Fresh Java 17 compilation and ordinary read-only preparation validation succeeded. The first `verify(True)` in the acquisition step raised:

```text
ValueError: Actual profile differs from independently frozen zero-case profile
```

This occurred before exact-head API/admission queries, receipt creation, activation, evaluator invocation or bridge numeric execution. The independently checked fixed receipt `refs/tags/experiment-receipts/diamond-stop-contract24-v1` returned 404 after failure. The artifact contains no activation, acquisition command/log/root, raw case rows or numeric summary. Compile provenance records zero cases. The workflow attempt failed; the permanent numeric receipt remains unused. No retry, repeated label, normalization or repaired continuation is authorized by this report.

Wins, score share, damage, firing/hit rates and round variation are **N/A**: no games or numeric cards ran. Invalid numeric rows/rounds are also N/A, rather than measured zero. No 300-second numeric process launched.

## Exact runtime mismatch

Both runs used Ubuntu 26.04.1/x86_64, Temurin 17.0.21+9 and CPython 3.13.18. The immutable freeze selected image **20261101.186.1** (`profile.json`); the fresh run received **20261026.180.1** (`profile-20261026.json`). Both belonged to the earlier allowed family, but the new full freeze bound the first image's actual bytes, so the second was correctly rejected.

The archived metadata-only comparison contains all **181 differing leaves**: 168 standard-library pyc hash entries, eight repeated Brotli native-library bindings, two loader-cache entries, and three image/selected-profile/gh-executable fields. Repeated bindings are not distinct causal effects. There is no evidence these differences would change the unexecuted diagnostic, and no bytes were normalized or waived.

All 30 tooling hashes, six core-artifact entries, JDK 17 hashes, base Python executable and pinned Python package/source/wheel graph matched. The bridge compilation reproduced the frozen artifacts. This establishes preparation parity, not bridge-main/import/numerical feasibility. The pinned original source, SDK, engine and committed JAR identities remain in the original freeze and preserved profiles.

| Identity | SHA256 or value |
| --- | --- |
| Payload head | `6007bcdd616e793030c5900e12bdb2e786964c52` |
| Freeze | `be6e1bc1271f5e5e0f1d2d1f76b008ed47125c6a3797876788adeb949e242c92` |
| Frozen profile | `dfbd5df2660bf5207144aa5990c2c019ce3662d1fd6efa93b2f9d65567a7ca1d` |
| Fresh actual profile | `ead57b2dd0deec7189a99ab111fea4a28cdb7c12dd56384ecf9d4ffe9bb024a7` |
| Fresh Python runtime proof | `8d126a4020e61d8cccbbc9cd4aa69fda95e93ced4de79111bbfecdf703765b52` |
| Bridge JAR | `e93b90303b8a93418c7ee75c3dfed9b52dc2325e1ebafb89ab71b6314ad699ba` |
| Actual boot | `9d31cbd2-f3b3-47ed-a31f-43e99b0537cb` |
| Actual workspace | `/home/runner/work/robocode/robocode` |

## Retained raw evidence and commands

The [failure manifest](../../benchmarks/diamond-stop-contract24/failed-preparation-41843951756/manifest.json) records all 72 original regular artifact files losslessly, deduplicating only exact immutable archived bytes, plus the complete original ZIP in verbatim parts, complete job log, run/jobs API metadata, all 181 profile differences, static audit and independent reviews. The original ZIP is authoritative; no archive was recompressed.

Original artifact 13329404282 is 19,607,166 bytes, SHA256 `38ddbe432db9dc9965a7f8d498c19239f175e9d40d38a8fa6bbc1269aef71e6d`. Full decoded job log is 54,914 bytes, SHA256 `f563ad6c0496ecc8f0fcf7cb45c3a1fb11d2acd8a56c657386e76a40bbe3c415`. Full profile comparison is 49,782 bytes, SHA256 `4056fe2ec2a34f61e771489fec8adde652bd52addd42e2963756d2429a463d7f`. The original successful preparation evidence is retained separately and unchanged.

The failed hosted commands were `python -I -B -S tools/diamond_stop24_host.py --collect`, then `--verify`, then `--acquire`. The complete workflow log and preparation command/provenance JSON retain actual resolved commands and runtime paths. Read-only archive verification executes no tested imports or cases:

```sh
python3 -I -B -S benchmarks/diamond-stop-contract24/failed-preparation-41843951756/verify.py
```

Disposition: preserve this failure and **do not rerun the failed workflow or relabel until a favorable image appears**. Reproducibly obtaining the exact frozen hosted runtime is unresolved. Any changed runtime plan, freeze or payload requires prospective amendment and independent review. No STOP discrepancy, movement improvement, bot correction, strength, promotion or merge claim follows from this attempt.
