# Playoff wave 3 — measured results

Four original games / 80 rounds completed on minimind; no extra tie games,
completed-game reruns or wave 4 starts. Frozen bots, source/JARs, runtimes and
battle harness remain unchanged. These are descriptive event results, without
an improvement claim or gameplay promotion.

| Match | Pairing | Combined score | Winner |
| --- | --- | --- | --- |
| L3 | Breakwater vs Diamond Port | 60–7,027 | Diamond Port |
| L4 | Prism vs Adaptive Prime | 4,592–1,516 | Prism |

Breakwater and Adaptive Prime are eliminated by their second playoff losses.
Diamond Port and Prism retain one loss. KeelHelm and Keel remain unbeaten and
will meet in the upper final when the user authorizes the next wave.

One zero-game startup attempt failed extracting the booter JAR because the disk
was full. Its exact report, directory, logs and hashes are retained. Independent
review confirmed the first completed game was valid and the second never began;
a narrowly guarded recovery reopened only that never-started fixture. The
unchanged runner then executed the three unplayed fixtures. See the
[startup recovery ruling](tournament-playoff-3-startup-recovery.md). No additional
physical rounds resulted from that startup failure.

Totals: **90 validated games / 1,800 rounds**, plus 20 earlier excluded solo
startup rounds: **1,820 known completed physical rounds**. All four raw observer
recordings pass independent CRC, hash, twenty-round coverage and official-final
score checks. No subsequent wave has started.

## Replay publication and storage

All 34 recorded games now have verified private movies and original raw
recordings. The user authorized removing Round 4 public replays if space was
needed. With this wave, assets reached 286,863,254 bytes, exceeding the hosting
archive limit of 256 MiB. Only the 14 derived Round 4 public videos/posters/chapters
were archived; their private movies, catalog, raw recordings and scores remain.
The live site now offers **all 20 playoff games**, including this wave's four,
beside their match results. Public assets total 188,351,669 bytes.

Large movies use hash-verified 4,000,000-byte transfer pieces that reconstruct the
exact original MP4; they retain quality, live scores, all chapters and 0.25× /
0.125× controls. Public browser QA verified a previously failing large opening
playoff movie playing without error, 0.125× speed, chapter seeking, Blob download
availability and inline placement on mobile without horizontal overflow.

## Evidence

Canonical report/draw: `benchmarks/tournament-2026-11-04/playoff-3.json` and
`playoff-3-prospective-draw.json`. Exact failed report and ruling:
`playoff-3-startup-stop-1.json`, `playoff-3-startup-ruling-1.json`. The failed
startup directory is preserved under `playoff-3/failed-attempts/`.

Full raw archive: `local-evidence/tournament-2026-11-04/playoff-3-full-artifacts.tar.gz`,
40,879,794 bytes, SHA256
`f45e4c892371a00a9a6137ed265a343e60027680b0ec63bca8b537a976080803`.
Independent verification: `playoff-3-verification.json`. All-34 movie proof:
`replay-catalog-original34-verification.json`; current public catalog proof:
`video/replay-catalog-verification.json`. The raw recordings, models and private
logs remain local. Rendering and playback checks start no games. All 21 tournament tests pass, including a publication-index coverage check against every completed stage game.

[Prospective protocol](tournament-playoff-3-protocol.md),
[running/replay workflow](tournament-running.md),
[PR #120](https://github.com/stephanelee/robocode/pull/120),
[public results/replays](https://robocode-elimination-cup.stephane-lee.chatgpt.site/matches.html?stage=playoff-3).
