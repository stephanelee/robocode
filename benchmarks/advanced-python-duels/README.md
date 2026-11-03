# Advanced Python duel evidence

This directory preserves the separately registered 1v1 measurements in [PR51](https://github.com/stephanelee/robocode/pull/51). [Protocol and qualified report](../../docs/improvements/advanced-python-duels.md).

## Attempt1: no games

[Run41647695154](https://github.com/stephanelee/robocode/actions/runs/41647695154), execution `f25c7ffb54232400c0c4f72c4b674f20169fbdaa`, immutable registration `b4fdd57a2c905e180cb16eb21ddce0fe093053d4`. Both Diamond and Adaptive jobs stopped before battle launch because the shared source downloader omitted pinned zero-byte files. Exact file-set verification rejected the incomplete source. **0 physical rounds; no checkpoint, score or competitive inference.**

[attempt1-no-games.tar.gz](attempt1-no-games.tar.gz) contains every downloaded original artifact file, both invalid empty-game reports, executed tooling/workflow/lock, original failed-job logs and GitHub artifact metadata. [attempt1-no-games-files.json](attempt1-no-games-files.json) records each extracted file's SHA256 and size. No failed evidence is replaced by a later continuation.
