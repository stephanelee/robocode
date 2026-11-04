# MeridianPeak v1: registered acquisition

Protocol: [issue #95](https://github.com/stephanelee/robocode/issues/95) and
[prospective method](../../docs/improvements/meridian-peak-v1.md).

Status: implementation and zero-game preparation. No screen or confirmation games
have run. Candidate/source/tool hashes remain unfrozen until independent review.
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
