# Dual-aim component diagnostic v2

This tooling experiment implements the [delegated next tests](retcon-next-experiments.md) using the [audited Retcon hints](retcon-review-response.md). It runs no game and changes no bot/JAR. Results are geometric command feasibility, not competitive scores. The prospective controller remains a hypothesis.

## Frozen matrix and command family

Before the first full run, freeze `dual-aim-v2` and its nine constant commands: target speed −8/0/+8 crossed with body turn −10/0/+10 degrees per turn. Baseline holds full speed in its current travel sign and zero turn. Both use legal engine acceleration/turn/wall mechanics. The selector maximizes the worst signed clearance, preferring baseline on exact ties, then fewer wall turns, smaller turn, and smaller target speed. Selection is in-sample optimization from this fixed command family, not an estimate of generalization or the accepted AIS behavior.

Development: 816 states, ranges 150/250/350/450/550/650, every integer speed −8..8, headings 0/90 degrees, layouts open/wall-gap0/wall-gap32/corner. Withheld: 90 predetermined distinct states, ranges 200/400/600, speeds −7/−3/0/3/7, headings45/135, layouts open/wall-gap16/corner. Both use a 1000×1000 arena, a stationary shooter directly west, and delay0/1/3 after the hypothetical shot decision. The fixed withheld set was first evaluated after development in v1 and is rerun unchanged after the v2 conformance repair. Its outputs have now been seen: it is not fresh untouched generalization evidence.

For each state, test both Retcon model requests and powers1.2/2/3 simultaneously. These discrete power scenarios do not cover every legal power or finishing cap. Four uncertainty profiles: fresh scan plus Retcon alignment tolerance `atan(18/d)`; constant-heading back-extrapolated scan age2 plus5degrees; age5 plus15degrees; unrestricted heading. The age profiles are sensitivity scenarios, not proven bounds on Retcon private history. This single-body diagnostic omits bot-to-bot collision/ram responses, shooter movement, multiple shots, damage and shot-induced behavioral changes; paths approaching the hypothetical shooter are not certified for those interactions. Stationary shooter position, firing time, and observed target state are fixture inputs; the tool never accesses Retcon private variables. A live controller would need explicit legal estimates and wider origin/time uncertainty.

## Mechanics and collision meaning

The movement port follows Tank Royale v1.4.0: speed update, movement along old heading, turn limited using new speed, first-wall intersection, speed zero after contact, then the engine's second position-constraint pass using the same previous origin. In this version target speed0 stops immediately and reversal8→3; do not substitute textbook braking. A wall/corner does not independently clamp coordinates and slide the bot. Golden tests cover these source facts, positive/negative velocity, and delayed commands.

Head-on uses the hypothetical target scan. Iterative linear aim uses Retcon's own coordinate-clamped constant-velocity forecast until bullet travel reaches forecast distance; that forecast is intentionally distinct from real movement physics. Gun firing occurs before same-turn gun rotation. Unknown actual gun direction is represented by the complete continuous alignment corridor around each model request, never an exact ray.

The collision test minimizes distance analytically over the union of bullet radial segments for every angle in each corridor, against each post-move bot circle of radius18. This sweeps the bullet, matching the server's collision convention; the engine does not simultaneously sweep the moving bot. The creation turn uses bullet segment `[v,2v]`, because the server advances its tick before constructing `BulletLine`. We conservatively retain hypothetical bullets through arena walls: a removed bullet cannot invalidate positive clearance against this larger threat set. Therefore nonpositive clearance means **uncertified**, not a proven live collision. Horizon is the arena diagonal divided by minimum tested speed11 plus one turn; thereafter the extended radial threats lie outside all legal bot positions.

Report baseline/proposed safe counts, uncertified finite-command states, wall contacts, and 20-turn RMS error of the opponent's constant-velocity forecast against each actual command trajectory. The latter measures forecast mismatch, not simulator prediction error. Golden conformance fixtures separately compare predicted states with explicit engine-derived numeric expectations; passing fixtures does not imply arbitrary live state accuracy. The dense independent segment oracle checks the analytic angular geometry, including interior headings, wraparound, and endpoint-miss/swept-hit cases.

## Reproduction and decision

Run bounded tests with `python -m unittest discover -s tools -p test_dual_aim_diagnostic.py -v`. Run `python tools/dual_aim_diagnostic.py --suite development --output benchmarks/dual-aim-v2/development.json`, then the unchanged command with `--suite withheld` and `withheld.json`. Outputs retain all rows and the exact tool SHA256. Python standard library only; no server/JDK/network needed. CI fixture job uses exactly Ubuntu26.04.

Before gameplay, inspect conditional safety sensitivity, uncertified states and wall cost; unrestricted-heading results expose what cannot be guaranteed without a justified observation model. Do not implement a live exact-ray controller from a fresh-scan result alone. An isolated movement hypothesis, if justified, must use legal shot inference, uncertainty-aware reachable commands, a wall veto, and fallback; it needs an independently reviewed immutable mechanism/score protocol before performance matches. Keep firing alignment separate.

## Primary sources

- [Engine speed/turn rules](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/rules/math.kt), [turn pipeline](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/core/TurnProcessor.kt), [wall/circle collision](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/core/CollisionDetector.kt), and [BulletLine](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/model/BulletLine.kt).
- [Pinned Retcon linear aim](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/gun/LinearAimModel.java) and [gun alignment](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/framework/roles/Gun.java#L45-L67).

## Completed v2 results and decision

**Retain the diagnostic; do not promote this selector to live gameplay.** Zero performance matches or gameplay/JAR changes. Independent review found v1 omitted the engine's second position-constraint pass and used an epsilon wall-contact comparison. At a corner the first pass can leave an invalid X because Y correction overwrites it. V2 preserves exact source comparisons and both passes; a reverse-corner regression plus19,584 legal corner state/command checks now pass (13 tests total). [Archived invalid v1 report](dual-aim-v1-invalid.md), [v1 summary](../benchmarks/dual-aim-v1/summary.json) and [v1 full raw artifact run](https://github.com/stephanelee/robocode/actions/runs/41513915658) remain available. Do not use v1 as an engine-conformant baseline.

The repair was frozen at [69e1f693](https://github.com/stephanelee/robocode/commit/69e1f693a5a52edf76e0d46a9214ff5fd4e197c5) before rerunning the entire unchanged816 development/90 fixed withheld matrix. No command, profile, ranking or threshold changed in response to outputs. Exact repaired tool SHA256: `21fc67d80540ec566ddc613fa41284fd0add99f02bf1d9fbc4d99804c8d45465`. Both suites ran locally development first then withheld; CI reproduces all10,872 rows on Ubuntu26.04/Python3.12 and asserts exact summary parity. [V2 summaries](../benchmarks/dual-aim-v2/summary.json), [full-row conformance comparison](../benchmarks/dual-aim-v2/conformance-diff.json), raw artifact `dual-aim-v2-raw` on the [PR #24 component workflow](https://github.com/stephanelee/robocode/pull/24/checks).

The full comparison found167/9,792 development rows changed:167 selected wall-turn counters and2 clearances; no chosen command, safety flag, forecast error or withheld row changed. Aggregate conditional safety counts therefore remain unchanged, while wall-contact totals rise. No earlier row/artifact is discarded, and the already inspected withheld set is not presented as new holdout evidence.

Each cell is baseline → selector conditional safe states, not independent samples or collision probabilities.

| Profile | Delay | Development /816 | Fixed withheld /90 |
| --- | ---: | ---: | ---: |
| fresh-alignment | 0 | 167 → 686 | 29 → 76 |
| fresh-alignment | 1 | 167 → 663 | 29 → 70 |
| fresh-alignment | 3 | 167 → 630 | 29 → 64 |
| age2-extra5deg | 0 | 130 → 618 | 22 → 56 |
| age2-extra5deg | 1 | 130 → 606 | 22 → 54 |
| age2-extra5deg | 3 | 130 → 552 | 22 → 50 |
| age5-extra15deg | 0 | 47 → 285 | 2 → 3 |
| age5-extra15deg | 1 | 47 → 212 | 2 → 5 |
| age5-extra15deg | 3 | 47 → 103 | 2 → 4 |
| unknown-heading | 0 | 0 → 0 | 0 → 0 |
| unknown-heading | 1 | 0 → 0 | 0 → 0 |
| unknown-heading | 3 | 0 → 0 | 0 → 0 |

Fresh/delay0 still leaves130 development states uncertified, including range150/speed−8/heading0/open. Age5+15degrees/delay3 leaves713 development and86 fixed withheld states uncertified; unrestricted-heading certifies none. These finite-command, single-shot scenarios do not bound hidden scan/shot-origin histories, ramming or multiple bullets.

Across the full130-turn horizon, including post-encounter motion, the baseline contacts a wall in816/816 development and81/90 withheld paths. V2 selector contacts645/816 and71/90 in fresh/delay0,724/816 and79/90 in age5/delay3. This is not a measured battle wall rate. A live wall veto/receding-horizon policy is untested.

The20-turn linear-forecast RMSE is unchanged: development baseline29.45 vs fresh/delay0 selector73.39 units; withheld36.44 vs70.89. It measures deviation from constant velocity, not simulator error or real-game strength. Wider uncertainty sharply degrades clearance despite forecast mismatch.

Next: retain this bounded negative deployment decision; prioritize the separate behavior-preserving targeting seam and legal-scan supervised forecast baseline. No extra geometry search or live movement change follows from these outputs.
