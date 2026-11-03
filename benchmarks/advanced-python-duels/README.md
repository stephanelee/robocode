# Advanced Python duel evidence

This directory preserves the separately registered 1v1 measurements in [PR51](https://github.com/stephanelee/robocode/pull/51). [Protocol and qualified report](../../docs/improvements/advanced-python-duels.md).

## Attempt1: no games

[Run41647695154](https://github.com/stephanelee/robocode/actions/runs/41647695154), execution `f25c7ffb54232400c0c4f72c4b674f20169fbdaa`, immutable registration `b4fdd57a2c905e180cb16eb21ddce0fe093053d4`. Both Diamond and Adaptive jobs stopped before battle launch because the shared source downloader omitted pinned zero-byte files. Exact file-set verification rejected the incomplete source. **0 physical rounds; no checkpoint, score or competitive inference.**

[attempt1-no-games.tar.gz](attempt1-no-games.tar.gz) contains every downloaded original artifact file, both invalid empty-game reports, executed tooling/workflow/lock, original failed-job logs and GitHub artifact metadata. [attempt1-no-games-files.json](attempt1-no-games-files.json) records each extracted file's SHA256 and size. No failed evidence is replaced by a later continuation.

## Corrected acquisition: Diamond Port

[Run41648730946](https://github.com/stephanelee/robocode/actions/runs/41648730946), execution `cdc9683319af11b795d4ce5befb846f96fae47e5`; original artifact13237080515. [diamond-evidence.tar.gz](diamond-evidence.tar.gz) contains all60 raw rows, three official arrays/logs, three distinct cold20-round models, the full original acquisition report with source/runtime/wheel/launcher hashes, both registrations, exact executed driver/loader/workflow/lock, and GitHub artifact metadata. [diamond-files.json](diamond-files.json) identifies all21 archived files by SHA256 and size. The archive excludes duplicate runnable AIS JARs and copied upstream sources/wheels; their exact immutable pins and executed source hashes remain in the report/lock.

Outcome: **0/60 AIS wins,3.885963% mean share, rank2 in all three games,0 AIS skips.** Every loss is retained; no favorable rerun or subset. Adaptive's independent job remains in progress at this publication. See the qualified report for shooting and damage details; this is a negative descriptive baseline, not a retrospective acceptance test or a claim about original Java Diamond.
