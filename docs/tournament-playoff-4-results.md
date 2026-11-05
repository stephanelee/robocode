# Playoff wave 4 — measured results

The user authorized wave 4 only on minimind. Four valid games / eighty rounds
completed with frozen bots, source/JAR/runtime pins, cold starts and recordings.
No tie-break game or wave 5 game started. These are descriptive tournament
results, without a gameplay improvement claim or promotion.

| Match | Pairing | Combined valid score | Winner |
| --- | --- | --- | --- |
| U7 | KeelHelm vs Keel | 2,833–2,564 | KeelHelm |
| L5 | Diamond Port vs Prism | 3,904–1,325 | Diamond Port |

KeelHelm reaches the grand final unbeaten. Keel receives its first playoff loss
and retains the lower-final path. Diamond Port retains one loss; Prism is
eliminated with two. Keel vs Diamond Port is the next fixed lower final, requiring
another user authorization before execution.

## Voided game, explicit replacement and budget

The original U7/game 1 engine result was Keel 1,413–1,222 KeelHelm, but its replay
was substantively corrupted. Independent review rejected a footer-only exception:
CRC/decoded length mismatch, malformed event data during round 18 and no intact
end-of-game event. The user explicitly chose to void it and play one replacement
after restoring space. Its score is excluded; all fourteen original files and
exact failed report remain retained and unchanged. The replacement score alone
counts. See the [prospective void ruling](tournament-playoff-4-void-ruling.md).

The initial disk assertion failed but a non-fail-fast shell still launched the
original attempt; this protocol deviation is recorded in the ruling. Subsequent
cleanup included the user-approved npm download cache only, recovering about
19 GB. A single-process 2 GiB guard passed before `os.execv` launched the unchanged
runner for the replacement and three unplayed fixtures. No other game was rerun.
No source, installed dependency, original tournament recording or current private
replay movie was removed. Temporary Git caches were reduced without altering
published history.

Measured physical budget: five completed engine games / one hundred rounds,
including the voided twenty; four validated games / eighty competitive rounds.
Totals: **94 valid games / 1,880 competitive rounds**, plus twenty earlier excluded
solo startup rounds and twenty voided wave-4 rounds: **1,920 known physical rounds**.
The committed index includes all 94 validated official hashes, preserves the
prior 90 and identifies the excluded original separately.

## Retained evidence

Canonical directory: `benchmarks/tournament-2026-11-04/playoff-4/`, including
`voided-attempts/p01-g1-recording-corrupt-1/`. Exact failed report:
`playoff-4-recording-stop-1.json`; user ruling: `playoff-4-void-ruling-1.json`.
Independent game/CRC/20-round/final-score audit:
`local-evidence/tournament-2026-11-04/playoff-4-verification.json`.
Corruption and score-only audits: `local-evidence/wave4audit/`.

Full archive: `local-evidence/tournament-2026-11-04/playoff-4-full-artifacts.tar.gz`,
237,624,319 bytes, SHA256
`a1737129a8d611ee6128d9ae6837270eb73e675b0d7cbbd1c73644b87a882721`.
It preserves all valid and voided raw files, metadata, execution logs, guarded
recovery script and independent audit evidence. Raw recordings/models/logs remain
local; repository exports contain approved scores and evidence metadata.

## Replay storage and publication

The user authorized archiving older lower-bracket live videos when needed.
Eight videos from L1/L2/L3/L4 were removed only from the public catalog/assets;
all private original movies and raw recordings remain. Round 4 stays archived.
The current catalog command uses `--omit-swiss4 --archive-lower-before-wave 4`.
Archived result rows state “Video archived · recording retained.”

Verified scores were published immediately (Site version 37) while all four
new wave-4 videos rendered. The original voided game has no valid replay and is
never substituted into the new movie. Movies use valid replacement/remaining
recordings, with live energy/official scores, twenty chapters and 0.25× / 0.125×
controls. Large files use lossless hash-verified 4 MB pieces.

[Protocol](tournament-playoff-4-protocol.md), [running guide](tournament-running.md),
[PR #120](https://github.com/stephanelee/robocode/pull/120),
[public results/replays](https://robocode-elimination-cup.stephane-lee.chatgpt.site/matches.html?stage=playoff-4).
