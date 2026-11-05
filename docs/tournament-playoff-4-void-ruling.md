# Wave 4 — user-authorized void and replacement

The user explicitly chose: **“Void Game 1 and play a replacement after restoring
disk space.”** Exclude the original U7/game 1 from match scores. Preserve its
engine output (Keel 1,413; KeelHelm 1,222), bot metrics, recording and failed report
unaltered. Reopen only that fixture for one cold replacement with the same launch
order, participants and registered settings, then finish the three unplayed
fixtures in wave 4. Do not start wave 5 or rerun a completed validated game.

The original engine completed twenty rounds, but its gzip replay is substantively
corrupted: computed CRC `0xee7e5687` differs from stored `0x232c7523`; decoded size
463,411,242 differs from stored 463,414,507. Only seventeen valid round-end events
survive, with malformed data during round 18 and no intact game-end event.
Independent review rejected footer-only repair or automatic original-score
acceptance. Both bots completed twenty textual telemetry triplets, but their
internal targeting `score=` values do not independently establish engine totals.
No replacement or acceptance occurred before the user's explicit decision.

The bounded recovery checks the exact failed report, official output, recording
and all fourteen retained file hashes. It snapshots the failed report as
`playoff-4-recording-stop-1.json`, records a separate `playoff-4-void-ruling-1.json`,
and relocates the original directory to
`playoff-4/voided-attempts/p01-g1-recording-corrupt-1/`. Its original score never
enters the pairing total. Failure history remains under `resolvedFailures` and
`voidedGames`; the twenty original physical rounds remain counted separately as
voided, with zero validated rounds. Repeated ruling application is refused.
The hash-guarded script is retained as
`local-evidence/tournament-2026-11-04/void-wave4-game1.py`.

## Storage and launch correction

An initial standalone disk assertion failed, but its shell continued into game
execution because it lacked fail-fast handling. This violated the planned launch
guard. The immediate filesystem report showed 652 MiB available; it does not prove
the cause of replay corruption. Record this deviation rather than claiming a
successful preflight. After further cleanup the user explicitly authorized clearing
only the 19 GB npm download cache. No game artifacts, source, installed packages
or unrelated application caches were deleted.

The replacement launch uses a single Python process: verify at least 2 GiB free,
then `os.execv` the unchanged runner only if that assertion succeeds. A failed
guard cannot fall through to a shell command. Preserve the runner/bracket/harness
pins and all original bot/runtime inputs. The revised maximum physical budget is
five scheduled engine games / one hundred rounds, including the voided twenty,
with at most two registered extra tie games (seven games / 140 rounds maximum).
The competitive budget remains four scheduled valid games / eighty rounds.
