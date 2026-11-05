# Post-tournament top-four melee — measured results

**Diamond Port wins the two-game exhibition on combined official score.**
Prism wins game 2; KeelHelm remains the completed 1v1 tournament champion.

| Exhibition rank | Bot | Game 1 | Game 2 | Combined | First places |
| --- | --- | --- | --- | --- | --- |
| 1 | Diamond Port | 5,432 | 3,755 | 9,187 | 20 |
| 2 | Prism | 3,498 | 4,477 | 7,975 | 10 |
| 3 | Keel | 2,038 | 1,914 | 3,952 | 6 |
| 4 | KeelHelm | 1,187 | 1,641 | 2,828 | 4 |

Exactly two original four-bot games / forty rounds completed on minimind. The
registered finishers came from the final bracket, not Swiss seeding. Game 1 used
KeelHelm, Diamond Port, Keel, Prism launch order; game 2 reversed it. Connection IDs
and actual arena positions remain recorded rather than assumed to equal launch
order. Both games used the official melee setup: 1000 × 1000 arena, min/max 4 bots,
twenty rounds, 0.1 gun cooling, 450 max-inactivity turns and 100,000 microsecond turn
timeout. No extra game, failure, rerun, elimination or championship points.

Rank is summed official score, then summed first places; exact ties share a rank.
This is a small descriptive exhibition, not a bot-strength improvement claim.
All source/JAR/guest/shared-runtime pins and cold per-game starts remained
unchanged; no descriptors or gameplay were modified to accommodate melee.

## Championship and accounting

The championship index is byte-identical to its merged record: KeelHelm champion,
98 valid games / 1,960 valid rounds / 2,000 known physical rounds. Every canonical
playoff report hash is unchanged. The exhibition adds two games / forty rounds
separately: 2,040 known physical rounds across both activities. The website's
championship totals, points, profiles and lives do not include exhibition scores.

## Validation and retained evidence

Independent review confirms four frozen identities, gzip CRC, twenty rounds,
all four bots at each round's first tick, game boundaries and final official
scores in both recordings. The new runner directly freezes its own script,
imported tournament helper, runtime helper hashes, manifests and Java-adapter
hash before play. A zero-game registration was reissued to add direct helper
provenance after review; the original registration and exact original script
are preserved. This consumed no games or rounds.

Canonical private directory:
benchmarks/tournament-2026-11-04/post-tournament-melee/. It includes report.json,
both g1/g2 directories, isolated bot copies, invocations, metrics, recordings,
models and registration history. Public score/derived-replay metadata:
benchmarks/tournament-2026-11-04/post-tournament-melee.json.
Independent proof:
local-evidence/tournament-2026-11-04/post-melee-verification.json.

Full private archive:
local-evidence/tournament-2026-11-04/post-melee-full-artifacts.tar.gz,
48,758,600 bytes, SHA256
f49014da936019ea1ec2bc51407b3d38b889bc94f1c77d4167fdf33c42620821.

Validation: all 34 tournament unit tests pass, JavaScript syntax and whitespace
checks pass. The cached archival integration verifies independent archival flags
and restoration without running games. Public browser checks confirm both videos
load, extra-slow speed survives next-game navigation, round-20 seek works and the
320-pixel layout has no horizontal overflow.

## Website and replays

Site version 44 adds a separate exhibition page linked from Home and Matches.
Aligned four-bot tables show each game's scores and the aggregate standings;
guest names and winners remain highlighted. Replay buttons sit beside the games,
with all four energy/score rows, correct arena aspect ratio, twenty chapters,
0.25x/0.125x controls and lossless multipart downloads.

The two new videos are verified against their recordings and all four official
scores. All 42 championship original movies and both exhibition originals are
retained privately (44 total). To fit the hosting limit, the user's existing
permission was used to archive four more older lower-bracket public movies:
the two L5 and two L6 games. Nothing was removed from private originals or score
history. The live site now has sixteen championship and two exhibition replays.
Use --archive-lower-before-wave 6 when rebuilding this edition's public catalog.

The new exhibition URL is resolved through TOURNAMENT_SITE_URL and the existing
link helper; no production hostname is embedded in reusable scripts/templates.
The current event manifest tracks exhibition results separately. The initializer
still creates only isolated drafts, not another running event.

Run from the repository root using the separate video environment for rendering:

    python3 tournament/scripts/post_tournament_melee.py --help
    python3 tournament/scripts/render_post_tournament_melee.py --help
    python3 tournament/scripts/manage.py site-url --stage exhibition

[Prospective protocol](tournament-post-melee-protocol.md) and
[current event hub](../tournament/events/2026-11-04/README.md).

[Source and documentation PR #122](https://github.com/stephanelee/robocode/pull/122).
