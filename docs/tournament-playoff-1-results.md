# Opening playoffs — measured results

User authorized **opening wave only**; no lower-bracket or later-wave game was started. All games ran sequentially on minimind with the unchanged frozen main roster, Java 25 runtime and observer harness. The prospective bracket and tie protocol were registered before play in [the protocol](tournament-playoffs-2026-11-05.md). No gameplay change, tuning or improvement claim.

| Match | Pairing | Combined score | Winner |
| --- | --- | --- | --- |
| U1 | KeelHelm vs AISurvivor | 4,978–586 | KeelHelm |
| U2 | Breakwater vs Adaptive Prime | 1,092–5,039 | Adaptive Prime |
| U3 | Keel vs Meridian | 3,815–254 | Keel |
| U4 | Diamond Port vs Prism | 4,099–1,350 | Diamond Port |

KeelHelm, Adaptive Prime, Keel and Diamond Port remain in the upper bracket. AISurvivor, Breakwater, Meridian and Prism move to the lower bracket with one loss each. **Nobody is eliminated.** Swiss points and seed order remain unchanged.

Measured budget: **8 validated games / 160 rounds**, zero extra tie-break games and zero failed games. All eight recordings passed gzip, SHA256, single game-start/game-end and battle-round 1–20 coverage checks; every retained battle.json matches the score report. Tournament totals are **78 validated games / 1,560 competitive rounds**, plus the previously retained 20 excluded solo rounds: **1,580 known completed physical rounds**. Recordings exist for Swiss 4 and the opening playoffs; practice and Swiss 1–3 remain unrecorded.

## Per-game official scores

| Match | Game | Bot A | Score A | Bot B | Score B |
| --- | --- | --- | --- | --- | --- |
| U1 | 1 | KeelHelm | 2,457 | AISurvivor | 302 |
| U1 | 2 | KeelHelm | 2,521 | AISurvivor | 284 |
| U2 | 1 | Breakwater | 606 | Adaptive Prime | 2,460 |
| U2 | 2 | Breakwater | 486 | Adaptive Prime | 2,579 |
| U3 | 1 | Keel | 1,895 | Meridian | 174 |
| U3 | 2 | Keel | 1,920 | Meridian | 80 |
| U4 | 1 | Diamond Port | 2,113 | Prism | 685 |
| U4 | 2 | Diamond Port | 1,986 | Prism | 665 |

## Evidence and verification

Canonical local evidence: `benchmarks/tournament-2026-11-04/playoff-1/` and `playoff-1.json`; prospective draw `playoff-1-prospective-draw.json`; source/seed registration `playoff-registration.json`. Local verification: `local-evidence/tournament-2026-11-04/playoff-1-verification.json`. The public score-only export is [in the repository](../benchmarks/tournament-2026-11-04/playoff-1.json). Full raw recordings, metrics, models, bot copies, invocations and logs remain local.

Full opening-wave archive: `local-evidence/tournament-2026-11-04/playoff-1-full-artifacts.tar.gz`, 60,440,730 bytes, SHA256 `7a9b8126500025fbab3b43929b78117d301aa401276009457186b4fcc31fde25`. Earlier archives and report hashes are preserved.

Executed orchestration hashes: `tournament.py`: `58856360ac83d64ab23a0c841fd3de0eb113504ffa11ef7b839faf63e9bf2304`, `tournament_playoffs.py`: `7b1b58726d713840b822c2727f17385ea3a24eaaae6befe849072e3fdf2740b5`. These matched the registered sources at launch; all earlier entry/shared-input/harness freezes passed. No source/JAR or runtime pins were replaced.

Independent pre-play review approved the seeded draw, tie rules, stop-on-failure behavior and frozen hashes. An exhaustive 32,768-outcome bracket check preserved second chances, prevented reentry after two losses and enforced the grand-final reset. Twenty local tournament tests pass. The earlier test-only missing-numberOfRounds fixture failure was corrected; it consumed no battles.

The public Site now has separate Home, Matches, Bots and Rules pages. Fourteen Swiss Round 4 replay buttons sit beside their game scores, with the player alongside the result cards, slower speed controls and no game-selection dropdown. New playoff recordings are retained locally; no new playoff MP4 is claimed. All four pages passed static link/data checks and nonbrowser render/player-state checks. Browser layout/playback inspection remains unavailable in the current Work settings.

[PR #117](https://github.com/stephanelee/robocode/pull/117) contains the bracket runner, tests, running guide, website template and score-only evidence. [Public matches](https://robocode-elimination-cup.stephane-lee.chatgpt.site/matches.html). No merge by the implementation agent.
