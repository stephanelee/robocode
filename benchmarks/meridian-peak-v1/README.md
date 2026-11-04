# MeridianPeak v1: registered acquisition

Protocol: [issue #95](https://github.com/stephanelee/robocode/issues/95) and
[prospective method](../../docs/improvements/meridian-peak-v1.md).

Status: **REJECTED after the sole five-game/25-round valid screen, zero skips.**
No confirmation opened. [Draft PR #96](https://github.com/stephanelee/robocode/pull/96).
Freeze SHA256: `07e72b257dea6f9c81217ff1143bb87addb300fa650bf0d7374b216ac24e973c`.
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

Screen-label [run41780412075](https://github.com/stephanelee/robocode/actions/runs/41780412075)
failed before admission/activation/game startup when the harness repeated fresh
venv construction. **0 physical games/rounds; no comparative result.**
[Full original ZIP parts, log, preparation and runtime evidence](preflight/failed-screen-preparation-41780412075/)
are retained. `reassemble.py` restores the exact ZIP20619a387320f7e88549b96404cf7b8a66d49cebd96ffdd170e97a1b44c20fc2.
The old freeze is preserved under `registration/`; preparation repair requires a
new tooling freeze, hosted zero-game path check and independent screen clearance.

Collection/verification repair independently reviewed:21/21 ordinary harness tests
pass; [source review and logs](preparation/collect-once-repair/). Only tooling,
protocol bookkeeping and preparation provenance changed in the new freeze above.
All25 participant assets, schedule, runtime family, packages and success criteria
match the preserved original registration. A fresh zero-game hosted profile must
exercise collect-once followed by two read-only verifications before any renewed
exact-head screen clearance. The screen and confirmation receipts remain unused.

Repaired hosted zero-game [preflight41782792500](https://github.com/stephanelee/robocode/actions/runs/41782792500)
passed at `05b3abdd3ce742d72de943e7a8a00d2fe1ba42f2`, allowed image
`20261026.180.1`: one fresh collection and two strict read-only verifications.
[Full original ZIP, log, environment proof and preparation evidence](preflight/run-41782792500/)
are retained. ZIP SHA256: `41ad46e0fe5704449ce4d7bf2b10d861410dcc269b2694bf5a3c7a971399a2da`.
This validates preparation, not strength. Physical games/rounds remain0; both
permanent acquisition receipts remain unused pending renewed exact-head clearance.

## Completed screen outcome

[Full result report](../../docs/improvements/meridian-peak-v1-results.md),
[all400 original acquisition files](screen-evidence/),
[exact original artifact ZIP parts and reassembly](screen-artifact-41783652109/),
[independent audit](analysis/original-screen-audit.json),
[provenance/variation/cost audit](analysis/independent-outcome-provenance.json),
[games.csv](analysis/games.csv), [all25rounds.csv](analysis/rounds.csv).

Diamond share12.397820% control→25.505051% Peak, wins0/5→1/5. Peak fails
required>35% and lower incoming uncapped damage/100enemy shots (98.924016→128.732704).
All transfer guards passed, but do not rescue the rejection. No new confirmation,
rerun, tuning or replacement is authorized. Permanent screen receipt object
`5c4e19bb19b32ee47ec186a608a1b376f1102693` remains consumed forever.

To reconstruct the original ZIP and verify the portable raw copy without starting
any bot or game:

```sh
python3 benchmarks/meridian-peak-v1/screen-artifact-41783652109/reassemble.py
python3 benchmarks/meridian-peak-v1/analysis/independent_audit.py \
  --raw benchmarks/meridian-peak-v1/screen-evidence \
  --artifact-zip benchmarks/meridian-peak-v1/screen-artifact-41783652109/artifact.zip \
  --artifact-sha256 9479829d95903329fc12e17ab4164943f4940f5be38940636956d69a52940afa \
  --expected-head c5239a624426b88740445f2a94675c9db995ee20 \
  --expected-freeze 07e72b257dea6f9c81217ff1143bb87addb300fa650bf0d7374b216ac24e973c \
  --output /tmp/meridian-peak-screen-audit.json
```

The independently verified live receipt is also preserved in
[github-screen-receipt.json](analysis/github-screen-receipt.json). The portable
archive cannot rehash unarchived live JDK/venv bytes or prove absence of external
unrecorded games; retain the original hosted provenance and remote-ref audit.
