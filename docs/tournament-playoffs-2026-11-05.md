# Minimind playoffs — prospective execution registration

The user requested the next rounds after the four Swiss rounds completed, then
selected **the opening playoff round only**. Execution in this request is limited
to wave 1; no wave 2 game starts without a new user instruction. This is
the previously agreed top-eight double-elimination bracket. No bot sources,
parameters, learning policy, dependencies, battle settings or runtime pins change.
Execution remains sequential on minimind, never GitHub Actions. Every game starts
cold, plays 20 rounds and retains its official output, logs and observer recording.

Seeds from the final cumulative Swiss table:
1. KeelHelm
2. Keel
3. Diamond Port (guest)
4. Breakwater
5. Adaptive Prime (guest)
6. Prism
7. Meridian
8. AISurvivor

## Fixed bracket, before any playoff game

| Wave | Matches | Sources |
| --- | --- | --- |
| 1 | U1, U2, U3, U4 | seeds 1–8, 4–5, 2–7, 3–6 |
| 2 | U5, U6; L1, L2 | winners U1–U2, U3–U4; losers U1–U2, U3–U4 |
| 3 | L3, L4 | winner L1 vs loser U6; winner L2 vs loser U5 |
| 4 | U7; L5 | winners U5–U6; winners L3–L4 |
| 5 | L6 | winner L5 vs loser U7 |
| 6 | G1 | winner U7 vs winner L6 |
| 7, only if needed | G2 | same finalists, if the lower-bracket finalist wins G1 |

The first playoff loss sends a bot to the lower bracket; the second eliminates it.
Swiss/practice losses do not count. There is no reseeding. Crossed lower-bracket
feeds are fixed above, not chosen after seeing results. Grand-final reset is
mandatory if the unbeaten finalist loses G1. A completed bracket has exactly one
remaining bot with fewer than two playoff losses.

Every pairing uses two cold 20-round games with reversed launch order. The sum of
official totalScore selects the winner. If tied, the sum of round first places
breaks the tie. If both tie, play exactly one extra cold 20-round game in the
pairing's original a/b launch order. Its official totalScore alone decides. If
that game also ties, choose the entrant at index
`int(sha256('minimind-playoff-20261105|MATCH_ID|a|b'), 16) % 2` in `[a, b]`.
This deterministic draw of lots is registered now; no result-dependent seed search
or further sudden-death games. All tie resolutions and full game scores are kept.
Playoff advancement uses match wins/losses, without changing the Swiss 3/1/0 table.

Budget: 14 pairings / 28 games / 560 rounds without a reset; 15 pairings / 30 games /
600 rounds with a reset. At most one extra 20-round tie-break game per pairing:
absolute maximum 45 games / 900 rounds. Opening wave alone is 8 games / 160 rounds
(maximum 12 games / 240 rounds if every pairing needs its extra game). Failed or
incomplete games stop execution; no automatic reruns, replacement or forfeits.

Freeze the four canonical Swiss reports and final seeds in
`playoff-registration.json` before play. Recheck the existing entry, shared-input
and recording-harness freezes through the same runner. Register each wave's exact
draw and source dependencies before any game. Record the new orchestration hashes
separately; preserve all earlier evidence hashes and published history.

The public Site shows the bracket, completed game scores, explicit match winners,
playoff losses and eliminations. Publish verified completed-game snapshots as
available, retaining the blue palette, guest highlights and noindex/nofollow.
Recordings remain local; newly rendered videos are added only when ready. Existing
Round 4 replay playlist and slower viewer settings remain available. This event
is descriptive evidence; it does not establish broad bot strength or improvements.
