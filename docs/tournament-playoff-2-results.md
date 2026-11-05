# Playoff wave 2 — measured results

The user authorized the next wave only. All eight scheduled games completed on minimind; no wave 3 game started. Frozen bots, models-at-start, source/JARs, runtime pins and the battle harness remain unchanged. This is descriptive tournament evidence, without a gameplay change or strength-improvement claim.

| Match | Pairing | Combined score | Winner |
| --- | --- | --- | --- |
| U5 | KeelHelm vs Adaptive Prime | 3,949–1,635 | KeelHelm |
| U6 | Keel vs Diamond Port | 2,754–2,312 | Keel |
| L1 | AISurvivor vs Breakwater | 939–2,410 | Breakwater |
| L2 | Meridian vs Prism | 121–6,180 | Prism |

KeelHelm and Keel reach the upper final unbeaten. Adaptive Prime and Diamond Port retain their lower-bracket path. Breakwater and Prism advance in the lower bracket. **AISurvivor and Meridian are eliminated by their second playoff losses.**

Measured budget: **8 original games / 160 rounds**, zero extra tie-break games and no reruns. One telemetry validation stop was resolved by an independently reviewed ruling accepting its original completed engine game. The original failed stage snapshot, error, raw bytes and runner log remain retained. The resumed unchanged runner played only the three unattempted fixtures. See [the explicit ruling and evidence hashes](tournament-playoff-2-ruling.md). There were no engine-game failures; one validator rejection remains recorded in resolvedFailures/adjudications.

All eight official outputs match their recordings, each with game-start/game-end and battle rounds 1–20. Overall totals: **86 validated games / 1,720 competitive rounds**, plus the 20 earlier excluded solo startup rounds: **1,740 known completed physical rounds**.

## Replays and mobile review

All **30 recorded games / 600 battle rounds from Swiss 4 onward** have playable derived videos, including both playoff waves. The original 14 Round 4 movies are unchanged. All videos passed source/video hashes, official scores, chapter 1–20, codec/frame-count/duration and per-asset size checks. The catalog tool reuses verified movies and rebuilds stage/draw/game ordering without running battles or publishing raw recordings. Previous/next and play-all stay inside the selected round; 0.25× and 0.125× speeds remain.

Browser QA is now available. Actual checks covered phone widths 320 and 390, tablet width 768 and desktop width 1,280. Matches played at 0.25×, switched to 0.125× and retained speed on the next game with no media error. The player sits below its selected card on phones and alongside scores on desktop; crossing the breakpoint retained playback. Navigation/replay controls have 44px targets. The narrow bot listing uses one readable column, active round tabs scroll into view, and the 320px profile dialog/results fit without horizontal overflow. Tablet navigation and small-phone hero buttons were adjusted. Earlier reports’ lack-of-browser-access limitation describes their initial publication; it is superseded for these newly checked layouts.

## Retained evidence

Canonical local directory: `benchmarks/tournament-2026-11-04/playoff-2/`. Stage/registration: `playoff-2.json` and `playoff-2-prospective-draw.json`. Original failed report: `playoff-2-validation-stop-1.json`; hash-bound ruling: `playoff-2-ruling-1.json`. Logs, recordings, metrics, models and bot copies remain local. The public repository contains sanitized score snapshots, ruling metadata and the complete official score index.

Full wave archive: `local-evidence/tournament-2026-11-04/playoff-2-full-artifacts.tar.gz`, 98,544,162 bytes, SHA256 `aef60828578eef2083054786c25d1f787ba228a35535b5fa9b39b232ff276fa6`. Verification: `playoff-2-verification.json`; pinned engine bytecode/custody: `adjudication-1/`; media provenance: `video/replay-catalog-verification.json` (232,348,142 video bytes).

Twenty tournament tests pass. Independent reviews approved the prospective wave and the original-score ruling, with an isolated recovery test confirming exact failed-snapshot preservation, single counting and repeated-application rejection. Render/provenance checks consumed no additional battles.

[Prospective rules](tournament-playoff-2-protocol.md), [running and replay workflow](tournament-running.md), [PR #120](https://github.com/stephanelee/robocode/pull/120), [public results/replays](https://robocode-elimination-cup.stephane-lee.chatgpt.site/matches.html?stage=playoff-2). No merge by the implementation agent.
