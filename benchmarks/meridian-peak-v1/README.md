# MeridianPeak v1: registered acquisition

Protocol: [issue #95](https://github.com/stephanelee/robocode/issues/95) and
[prospective method](../../docs/improvements/meridian-peak-v1.md).

Status: source/JAR/tool freeze independently reviewed; no screen or confirmation
games have run. [Draft PR #96](https://github.com/stephanelee/robocode/pull/96).
Freeze SHA256: `c28024373ffd79b900952790d01db517c4fc9e1bb0132222b0ca0b4a59e6b145`.
Local preparation and contract tests establish implementation properties only.

The screen is five cold five-round games, in `games.json` order. A complete valid
screen PASS is required before the unchanged candidate's nine fresh twenty-round
confirmation games. Stop at the first invalidity; preserve partial evidence and
never replace or resume an acquisition. Each stage has its own permanent receipt.

`preparation/` preserves local build/test/parity commands, logs and provenance.
Hosted zero-game preflight and both acquisition stages use the pinned workflow
`.github/workflows/meridian-peak-v1.yml`. Every hosted job runs Ubuntu 26.04;
Java 17 builds/tests precede the Java 25 match runtime. Runtime/profile or frozen
payload mismatches prevent collection. Profile-label runs cannot start games.

Own private skips are polled during execution. Observer-only skips and all other
recording/model validity checks are evaluated at the completed-game boundary
before the next game. The observer never supplies data to bot policy code.

Keep full original recordings, logs, commands, explicit own model snapshots,
runtime/package/host/receipt documents and hashes. Results and a standalone
independent audit will be linked here after the sole screen attempt.

Hosted zero-game [preflight41779652044](https://github.com/stephanelee/robocode/actions/runs/41779652044)
passed at head `ed58717805d7bb4d1aff333cabeb02709ad380c0`, allowed image
`20261026.180.1`. [Full original ZIP, log, runtime and preparation evidence](preflight/run-41779652044/)
are preserved. ZIP SHA256: `68282d4bb7a715d314a38d3e28d57a996a5b3387c270f882e5b5063526f49df4`.
Both acquisition jobs were skipped; no durable acquisition receipt was created.
