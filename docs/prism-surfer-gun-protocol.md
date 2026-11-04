# Prism v2 prospective screen: anti-surfer gun on confirmed bullets

**Status: registered, not run.** No v2 game has been played. The screen runs only after an independent premeasurement review of the exact head. The review URL goes in `approvedReview` and the 40-character reviewed commit in `approvedHead`, both in [`docs/prism/screen-v2-plan.json`](prism/screen-v2-plan.json), and in the tracking issue. The implementer must not approve their own protocol. This protocol is frozen as written.

## Motivation (from the rejected v1 screen)

The [v1 screen](prism-screen-protocol.md) (`Prism.jar 8c045f80…`, 200 rounds, [evidence](../benchmarks/prism-screen-v1/README.md)) failed only the Diamond Port gate, with a mean share of 28.61%. The analysis below used Prism-side outputs only: its own telemetry and the official results.

- Prism hit 7.5–8.4% of its confirmed shots against Diamond, against 13.2–16.5% against Retcon and Adaptive Prime.
- That rate did not improve within a game. By 5-round block it was 8.8%, 7.8%, 7.8%, 8.0%, while the learned guns were selected for 60–75% of aims.
- Diamond's inferred hit rate on Prism stayed at 12.1–13.8% across the same blocks.

Hypothesis: virtual-wave training may overrepresent motion that differs from responses to confirmed outgoing bullets. The Prism-only aggregates motivate testing confirmed-wave training, but do not establish this as the cause of the Diamond result, prove that a target reacts to every inferred bullet, or show the mismatch harmless against other opponents. Existing outcomes are not an isolated causal comparison. The registered screen tests the complete unchanged profile plus this gun; the effect remains unknown.

## Single change (profile `prism-profile-2-surfer-gun`, Prism 0.2.0)

Prism adds one gun, `surfer`. It is a three-window recent nearest-neighbour gun with capacities 32, 128 and 512 and up to 5 neighbours per window. It uses the recent gun's movement features and precise-GF labels, and is trained **only** on engine-confirmed (`OUTGOING`, physical) waves. Virtual crossings never enter it. It becomes eligible after 8 confirmed-wave labels. It is rated on the same frozen-prediction fire-opportunity cohort as every other gun, and the unchanged selector rule (EMA 0.05, margin 0.02, at least 8 resolved for both guns) decides whether it is ever used. No other parameter changes: movement, power, radar, inference and the existing guns and their training are untouched.

## Hypothesis

Profile 2 passes the same specification §13.3 engineering screen that profile 1 failed: all 13 games valid; no failures; skips < 0.5% of Prism turns per cohort; collisions < 1% of turns; mean equal-game share > 40% against Diamond Port and against Retcon, and ≥ 50% against Adaptive Prime, SpinBot and Walls.

No improvement is claimed in advance. The magnitude is unknown: no Diamond data were used to fit or tune the change, and no offline estimate is possible from Prism's aggregate telemetry. A Diamond share moving from 28.6% toward 40% would need a substantially higher hit rate. That is plausible but unproven.

## Pins and structure

| Item | Pin |
| --- | --- |
| Candidate | `experiments/Prism/Prism.jar` SHA-256 `f9c75a86e21aafa3dd889b626a2d9c4e321d69d4d3917e0483ea324360e27263`, byte-identical across two clean builds; source tree `3de9c39c42d36d08e37ee78a86a99b6e34488d4f` |
| Baseline (not re-run) | v1, `8c045f80…`, rejected screen |
| Engine, opponents, runtime | Identical to the v1 freeze fields copied into the v2 plan: Tank Royale 1.4.0 runner, strong-melee and Retcon lock blobs, sample ZIP, `ubuntu-26.04`, Temurin 17 for build/parity then Temurin 25.0.5+8, Python 3.13 |
| Games | 13 independent cold games, 200 rounds: Diamond Port, Retcon and Adaptive Prime 3×20 each; SpinBot and Walls 2×5 each. Seats alternate within each opponent. Fresh Prism seeds 4201–4213, not overlapping v1. Server start positions are not seeded |
| Replay protection | Receipt `refs/tags/prism-screen-v2-f9c75a86e21aafa3`, created atomically at the execution head before the first game and never deleted. Exact-head clearance compares the plan at `approvedHead` with the executed plan, ignoring only `approvedReview` and `approvedHead`. The harness runs with `--plan docs/prism/screen-v2-plan.json` |

## Metrics and decision rules

The metrics are those of v1: official totals, first places, damage, share, shots requested/confirmed/hit, collisions, skips, failures, latency and every counter. They now include `gun.aim.surfer` and the selector's choices. Per-round snapshots give learning curves.

1. Any invalid or incomplete game: **reject**. It is preserved and never rerun.
2. Any failed gate: **reject** profile 2 for promotion. There is no subset, rerun or retune.
3. All gates passed: record "passed screen" only. Superiority over Diamond Port or Retcon needs a separately registered fresh confirmation (10 independent cold 20-round games per opponent; majority of first places and a 95% game-cluster bootstrap lower bound above 50%). A v1-versus-v2 comparison is descriptive only, because v1 and v2 used different seeds and unseeded start positions.

After the screen, `docs/improvements.md` gets the budget, outcome, decision, PR and raw-evidence links.


## Review readiness corrections before any v2 game

User-authorized administrative continuation by a separate implementation agent fixes only the three reviewed blockers: guarded same-repository PR label run-prism-screen-v2 on branch prism/surfer-gun-v2, attempt1/exact PR-head execution and permanent v2 receipt; approval fixtures accept both pending and well-formed recorded approval; confirmed-wave surfer stores now enter existing payload accounting and FIFO cap eviction after existing virtual-wave priority. A small-budget synthetic regression exercises saturation/accounting/eviction without allocating128MiB. This is estimated learned payload, not JVM heap/CPU certification. JAR/source pins below are refrozen after Java17 tests/parity; the13-game/200-round4201-4213 schedule, opponents, gates, strategy and all gun parameters are unchanged. No v1 evidence is altered. These readiness corrections are additional provenance and are not claimed as the original author's exclusive work. No games until a separate exact-head review followed only by two approval metadata values and one explicit label. No retries, replacements, tuning or automatic confirmation.

Prospective prelaunch clarification: the driver now stops only on an invalid acquisition/integrity row and explicitly lists unplayedGameIds. Valid score/safety losses still complete the registered budget; focused actual-main-loop fixtures prove both paths. This was fixed before any v2 outcomes, without changing JAR, games, gates or opponent population.


## Results: registered screen rejected, no confirmation

Run41892189167 completed all13 games/200 rounds. Frozen profile2 was **rejected**, solely Diamond mean30.339588% missing>40%; Retcon99.335769%, Adaptive63.434299%, samples100% met their gates. Official Prism first places11/60,60/60,35/60,10/10,10/10. All200 round snapshots/13 final records present; zero recorded Prism skips/collisions/failures across145767 reported decisions. SURFER selected3595 Diamond aims, which is not a confirmed-shot count or causal proof. Historicalv1-to-v2 mean differences are descriptive across fresh seeds/unseeded positions, not paired improvement evidence. No rescue, retune, replacement or confirmation.

[Original raw/source/pins/gate reproduction and limits](../benchmarks/prism-screen-v2/README.md). The measured source/JAR remain only an unpromoted standalone experimental reference; production AIS and independent Keel work are unchanged. The exact spent workflow is archived and retired.
