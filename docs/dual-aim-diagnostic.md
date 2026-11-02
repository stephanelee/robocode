# Dual-aim component diagnostic v1

This tooling experiment implements the [delegated next tests](retcon-next-experiments.md) using the [audited Retcon hints](retcon-review-response.md). It runs no game and changes no bot/JAR. Results are geometric command feasibility, not competitive scores. The prospective controller remains a hypothesis.

## Frozen matrix and command family

Before the first full run, freeze `dual-aim-v1` and its nine constant commands: target speed −8/0/+8 crossed with body turn −10/0/+10 degrees per turn. Baseline holds full speed in its current travel sign and zero turn. Both use legal engine acceleration/turn/wall mechanics. The selector maximizes the worst signed clearance, preferring baseline on exact ties, then fewer wall turns, smaller turn, and smaller target speed. Selection is in-sample optimization from this fixed command family, not an estimate of generalization or the accepted AIS behavior.

Development: 816 states, ranges 150/250/350/450/550/650, every integer speed −8..8, headings 0/90 degrees, layouts open/wall-gap0/wall-gap32/corner. Withheld: 90 predetermined distinct states, ranges 200/400/600, speeds −7/−3/0/3/7, headings45/135, layouts open/wall-gap16/corner. Both use a 1000×1000 arena, a stationary shooter directly west, and delay0/1/3 after the hypothetical shot decision. Withheld outputs must be recorded after development without altering commands or thresholds; they are component holdout cases, not fresh competitive confirmation.

For each state, test both Retcon model requests and powers1.2/2/3 simultaneously. These discrete power scenarios do not cover every legal power or finishing cap. Four uncertainty profiles: fresh scan plus Retcon alignment tolerance `atan(18/d)`; constant-heading back-extrapolated scan age2 plus5degrees; age5 plus15degrees; unrestricted heading. The age profiles are sensitivity scenarios, not proven bounds on Retcon private history. This single-body diagnostic omits bot-to-bot collision/ram responses, shooter movement, multiple shots, damage and shot-induced behavioral changes; paths approaching the hypothetical shooter are not certified for those interactions. Stationary shooter position, firing time, and observed target state are fixture inputs; the tool never accesses Retcon private variables. A live controller would need explicit legal estimates and wider origin/time uncertainty.

## Mechanics and collision meaning

The movement port follows Tank Royale v1.4.0: speed update, movement along old heading, turn limited using new speed, first-wall intersection, speed zero after contact. In this version target speed0 stops immediately and reversal8→3; do not substitute textbook braking. A wall/corner does not independently clamp coordinates and slide the bot. Golden tests cover these source facts, positive/negative velocity, and delayed commands.

Head-on uses the hypothetical target scan. Iterative linear aim uses Retcon's own coordinate-clamped constant-velocity forecast until bullet travel reaches forecast distance; that forecast is intentionally distinct from real movement physics. Gun firing occurs before same-turn gun rotation. Unknown actual gun direction is represented by the complete continuous alignment corridor around each model request, never an exact ray.

The collision test minimizes distance analytically over the union of bullet radial segments for every angle in each corridor, against each post-move bot circle of radius18. This sweeps the bullet, matching the server's collision convention; the engine does not simultaneously sweep the moving bot. The creation turn uses bullet segment `[v,2v]`, because the server advances its tick before constructing `BulletLine`. We conservatively retain hypothetical bullets through arena walls: a removed bullet cannot invalidate positive clearance against this larger threat set. Therefore nonpositive clearance means **uncertified**, not a proven live collision. Horizon is the arena diagonal divided by minimum tested speed11 plus one turn; thereafter the extended radial threats lie outside all legal bot positions.

Report baseline/proposed safe counts, uncertified finite-command states, wall contacts, and 20-turn RMS error of the opponent's constant-velocity forecast against each actual command trajectory. The latter measures forecast mismatch, not simulator prediction error. Golden conformance fixtures separately compare predicted states with explicit engine-derived numeric expectations; passing fixtures does not imply arbitrary live state accuracy. The dense independent segment oracle checks the analytic angular geometry, including interior headings, wraparound, and endpoint-miss/swept-hit cases.

## Reproduction and decision

Run bounded tests with `python -m unittest discover -s tools -p test_dual_aim_diagnostic.py -v`. Run `python tools/dual_aim_diagnostic.py --suite development --output benchmarks/dual-aim-v1/development.json`, then the unchanged command with `--suite withheld` and `withheld.json`. Outputs retain all rows and the exact tool SHA256. Python standard library only; no server/JDK/network needed. CI fixture job uses exactly Ubuntu26.04.

Before gameplay, inspect conditional safety sensitivity, uncertified states and wall cost; unrestricted-heading results expose what cannot be guaranteed without a justified observation model. Do not implement a live exact-ray controller from a fresh-scan result alone. An isolated movement hypothesis, if justified, must use legal shot inference, uncertainty-aware reachable commands, a wall veto, and fallback; it needs an independently reviewed immutable mechanism/score protocol before performance matches. Keep firing alignment separate.

## Primary sources

- [Engine speed/turn rules](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/rules/math.kt), [turn pipeline](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/core/TurnProcessor.kt), [wall/circle collision](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/core/CollisionDetector.kt), and [BulletLine](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/model/BulletLine.kt).
- [Pinned Retcon linear aim](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/retcon/gun/LinearAimModel.java) and [gun alignment](https://github.com/robocode-dev/rumble-bots/blob/28981c7a469ac064194b41001cd7183ec5ecf189/bots/java/Retcon/src/fnl/bots/framework/roles/Gun.java#L45-L67).

## Completed offline results

Decision: **retain the diagnostic, do not promote this selector to live gameplay**. The finite commands can evade the narrowly modeled shots, but sensitivity to unobserved history and wall contact prevents a deployable safety claim. This is not a rejection based on competitive scores: zero score matches were run.

Commands/matrix were published at [1dc69bd](https://github.com/stephanelee/robocode/commit/1dc69bd0a05f1d96a028eaede7c1d993219a9e0d) before either full suite. The unchanged tool ran development first, then withheld. Exact tool SHA256: `fab7093fd8294dadfbaccbde651a270a20b573955c30b04e1563c55fcc5cdc3e`. Initial execution used local Windows/Python, without Java/server/network; CI reproduces on Ubuntu26.04/Python3.12. [Machine-readable summaries](../benchmarks/dual-aim-v1/summary.json) retain all profile/delay counts and forecast-error measures. Full row artifacts are produced by [Component diagnostics](../.github/workflows/component-diagnostics.yml), artifact `dual-aim-v1-raw`; [PR #24](https://github.com/stephanelee/robocode/pull/24) links the exact-head workflow runs.

Each cell below is baseline → selector conditional safe states. There are 816 distinct development states and90 withheld states, each evaluated under four profiles and three delays: 9,792 +1,080 condition rows. Counts are deterministic coverage, not independent random samples or collision probabilities.

| Uncertainty profile | Delay | Development /816 | Withheld /90 |
| --- | ---: | ---: | ---: |
| Fresh +alignment tolerance | 0 | 167 →686 | 29 →76 |
| Fresh +alignment tolerance | 1 | 167 →663 | 29 →70 |
| Fresh +alignment tolerance | 3 | 167 →630 | 29 →64 |
| Age2 +5degrees | 0 | 130 →618 | 22 →56 |
| Age2 +5degrees | 1 | 130 →606 | 22 →54 |
| Age2 +5degrees | 3 | 130 →552 | 22 →50 |
| Age5 +15degrees | 0 | 47 →285 | 2 →3 |
| Age5 +15degrees | 1 | 47 →212 | 2 →5 |
| Age5 +15degrees | 3 | 47 →103 | 2 →4 |
| Unrestricted heading | 0/1/3 | 0 →0 | 0 →0 |

Even the fresh/delay0 case leaves130 development states uncertified. For example, range150/speed−8/heading0/open arena remains uncertified at clearance−17.84 despite selecting reversal+turn; stopping/turning/reversing is not instantly attainable sideways displacement. The unknown-heading profile certifies no command. It is an intentionally broad sensitivity limit, not evidence Retcon chooses arbitrary headings.

Wall contacts are counted across the full130-turn horizon, including time after the modeled encounter. Thus all816 straight development paths eventually contact a wall; selector contacts633 in fresh/delay0 and709 in age5/delay3. Withheld straight paths contact81/90, selector71/90 and79/90 respectively. These counts cannot be called collision-rate improvement during a battle. A future live wall veto and receding-horizon controller remain untested.

The20-turn opponent linear-forecast RMSE averages29.45 units for the development baseline and73.39 for fresh/delay0 selector; corresponding withheld values36.44 and70.89. This shows the selected motions deviate from constant velocity, not that our simulator is inaccurate or that shots will miss in real games. Under age5/delay3 the selector RMSE falls40.70/38.35, while conditional clearance collapses: forecast error alone is an inadequate promotion metric.

Next action: preserve these limitations, finish the behavior-preserving targeting seam independently, and prioritize a legal-scan supervised forecast dataset/component baseline. Revisit movement only with a justified legal observation model and separately preregistered wall-safe mechanism. Do not turn this best-of-nine fixture search into a strength claim.

## Conformance repair and rerun protocol

Independent review found the v1 simulator omitted the engine's second position-constraint pass after applying a wall outcome. Because the Y correction can overwrite X at a corner, one pass can leave an invalid position. The13-test v2 repair includes the reverse-corner regression and19,584 legal corner/speed/heading/target-speed in-bounds checks. V1 remains preserved at the frozen source commit and its `dual-aim-v1-raw` artifact and summary; its full-trajectory results are invalid exploratory evidence, not an accepted engine-conformant baseline.

Before v2 outcomes, freeze this repair and rerun the unchanged complete development816/withheld90 matrix and nine-command family. No uncertainty profiles, command ranking, thresholds or selected cases are changed in response to results. Previously withheld outputs have now been seen: the v2 withheld set is a fixed holdout rerun after a conformance repair, not fresh untouched generalization evidence. Compare full rows and disclose affected fields/counts. No competitive matches or gameplay gates are involved.
