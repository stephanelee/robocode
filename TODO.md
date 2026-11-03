# Robocode TODO

- [x] Measure separate legal-wave defense with fixed context gun. [PR61](https://github.com/stephanelee/robocode/pull/61) / [report](docs/improvements/diamond-surf-v3.md):20-round pilot rejected (0wins/1.52% share);265 waves/200 matched hits/planner active, no fresh60. Production unchanged; Diamond remains unbeaten.

- [x] Test the fixed bounded online context gun in the separate challenger. [PR60](https://github.com/stephanelee/robocode/pull/60) / [report](docs/improvements/diamond-context-v2.md):20-round pilot rejected (0wins/10.80% share), despite185 learned shots/227 admitted passages; no fresh60. All raw/memory retained, production AIS unchanged. Beating Diamond remains open; new hypotheses require fresh registration.

- [x] Build and measure the separate first Diamond pressure challenger. [PR59](https://github.com/stephanelee/robocode/pull/59) / [report](docs/improvements/diamond-pressure-v1.md):20-round pilot rejected (0wins/2.59% share), no fresh60; ownsource/JAR/state/raw retained, production AIS unchanged. Beating Diamond remains open; no automatic rescue of this cohort.

- [x] Measure the fixed analytic-prior danger blend against advanced opponents and Retcon. [PR55](https://github.com/stephanelee/robocode/pull/55) / [report](docs/improvements/blended-duel-danger.md):330 rounds, screen passed but fresh Diamond+0.29pp/Adaptive−0.89pp failed gains; Retcon+4.62pp/all safety and sample guards passed. Accepted AIS source/JAR restored; all24 models/raw retained. Subsequent work follows the separate DiamondChallenger direction, not another AIS parameter trial.

- [x] Benchmark the requested stronger native melee roster including Retcon and user-authorized Python opponents. [PR52](https://github.com/stephanelee/robocode/pull/52) / [results](docs/improvements/strong-melee-baseline.md):60 rounds,AIS rank4 each,12.21% share,1/60 first places; all losses/collisions retained,zero skips. No gameplay edit; zero-game acquisition failure/correction separately preserved.

- [x] Measure the requested frozen default four-bot sample melee baseline. [PR49](https://github.com/stephanelee/robocode/pull/49) / [results](docs/improvements/melee-default-baseline.md):60 rounds,rank1 each,60/60 first places,66.37% share,zero skips;184 wall/169 bot collisions retained. Descriptive only; no optimization or source change.

- [x] Complete the user-authorized fresh frozen-default confirmation redo with unchanged criteria. [PR46](https://github.com/stephanelee/robocode/pull/46) / [results](docs/improvements/final-default-confirmation-redo.md):200 valid rounds,86.74% mean share,187/200 first places,82.27% binding t lower bound,zero skips; all gates pass. Original44 failure remains separate/cause unknown. Finite confirmation goal met for pinned Retcon; no further automatic games.

- [x] Attempt the final frozen default200-round confirmation once. [PR44](https://github.com/stephanelee/robocode/pull/44) / [outcome](docs/improvements/final-default-confirmation.md): invalid after140 physical rounds (120 strictly validated), one skipped AIS turn,60 unplayed; reliability not confirmed. No retry or automatic optimization continuation; accepted v2 unchanged.

- [x] Build and benchmark the coherent GF/circular gun and strict energy-finishing revision. [PR43](https://github.com/stephanelee/robocode/pull/43) / [results](docs/improvements/coherent-duel-v2.md): all168-round gates pass, Retcon55/60 wins and self-match30/30; hit rate0.72â†’5.22%. The bundled result does not isolate a learning effect or establish broad superiority.

- [x] Build and benchmark one coherent default duel composition. [PR42](https://github.com/stephanelee/robocode/pull/42) / [results](docs/improvements/coherent-duel-v1.md): all registered168-round whole-product gates pass; Retcon39/60 wins and self-match30/30. Gun accuracy fell and broad superiority is not established.

Repository: https://github.com/stephanelee/robocode

Based on a source review of `master` at commit `fb735035811e60bf30366a0952b5dfe1b8e079ac`. The Java 17 build has since been verified locally and in GitHub Actions (PR #1); headless battle integration and a small reference experiment are recorded in [the baseline report](benchmarks/reference-2026-11-02/README.md). These results do not establish competitive strength.

## First milestone: reliable baseline

- [x] Fix coordinate calculations across movement, targeting, bullet projection, and wall avoidance to follow Tank Royale's coordinate conventions.
- [x] Fix initial targeting so empty GuessFactor statistics default to the centre bin.
- [x] Align learning with movement by executing the action the DQN actually selects.
- [x] Correct enemy-bullet handling: the API exposes the bot's own bullets in tick events; use inferred enemy waves for enemy threats.
- [x] Fix round-end learning to preserve pending rewards and record terminal transitions for both death and survival.
- [x] Correct enemy turn-rate estimates for elapsed turns between scans.
- [x] Add focused tests for geometry, targeting defaults, and learning transitions.
- [x] Add a Gradle wrapper and CI for reproducible builds. (PR #1; local Java 17 build and GitHub Actions verified.)
- [x] Write a README and update stale architecture and bot metadata documentation.
- [x] Run baseline battles and record the results. (PR #3; 140 reference rounds across four scenarios.)

## Validate and improve learning

- [x] Repair wave-surfing statistics using bullet-hit evidence rather than only the bot's position when a wave passes.
- [x] Create automated battle benchmarks against a fixed opponent roster.
- [x] Record score, placement, hit rate, wall collisions, and skipped turns.
- [x] Separate training and evaluation modes, with reproducible random seeds.
- [x] Compare a fixed minimum-risk baseline against untrained and trained DQN policies before expanding the model. (Five evaluation rounds per policy/scenario; no strength conclusion.)
- [x] Improve saved-model handling with version checks and atomic writes.

## Later improvements

- [x] Add a standalone results dashboard for scores, hit rates, survival, and training/evaluation comparisons.

First milestone is complete when the build is reproducible, movement and targeting checks pass, and baseline battle results are recorded.

## Stronger 1v1 challenge and measured improvement cycles

- [x] Integrate pinned external Java Retcon 0.2.0 with source/dependency hash checks and actual server 1.4.0 compatibility.
- [x] Record the 35-round Retcon 1v1 baseline and organize methodology/reports under docs/.
- [x] Cycle 1: compare the first targeted gameplay candidate with the previous accepted version. (120 rounds; candidate rejected and gameplay reverted. [Report](docs/improvements/cycle-1.md); [issue #10](https://github.com/stephanelee/robocode/issues/10) remains open.)
- [x] Cycle 2: record the next candidate's measured gains/regressions. (120 rounds; candidate rejected and gameplay reverted. [Report](docs/improvements/cycle-2.md); [issue #11](https://github.com/stephanelee/robocode/issues/11) remains open.)
- [x] Cycle 3: record the third controlled hypothesis and final accept/reject decision. (60 training + 120 evaluation rounds; candidate rejected and gameplay reverted. [Report](docs/improvements/cycle-3.md); [issue #12](https://github.com/stephanelee/robocode/issues/12) remains open.)

See [docs/improvements.md](docs/improvements.md) and [tracker #8](https://github.com/stephanelee/robocode/issues/8). Only candidates that meet their preregistered criteria are retained.

## Source-grounded Retcon counterstrategy

- [x] Publish the opponent-source-grounded [strategy and prospective gate categories](docs/retcon-strategy.md).
- [x] Capture unchanged Retcon native statistics as offline diagnostics, with actual Ubuntu26.04 provenance. [Report](docs/retcon-observability.md).
- [x] Test the isolated bullet-energy scan-lag correction with an actual pinned-engine trace and 120 fixed-policy rounds. [Report](docs/improvements/foundation-scan-lag.md): rejected by mean score-share and wall-rate guards; online stage skipped, gameplay/JAR reverted.
- [x] Measure fixed-policy Retcon noise with a 400-round A/A calibration. [Report](docs/aa-calibration.md): block-delta SD 5.59 pp. Review correction: with a +5 pp observed floor, acceptance at a true +5 pp is about 50% at any budget; 8 blocks with a Student-t bound gives about 89% acceptance at +7.5 pp. The wall/wins guards are noise-dominated (exploratory).
- [x] Test shot-reactive 1v1 duel movement against Retcon's two guns. [Report](docs/improvements/duel-surf.md): v1 passed (share 14.8%â†’58.6%, Retcon hit rate 18.9%â†’5.3%); the review found simulation-physics defects, so gameplay is not merged on v1 evidence.
- [x] Measure the engine-conformant duel controller in its registered fresh comparison. [v2 report](docs/improvements/duel-surf-v2.md): **retained**; fixed-policy share 17.4%â†’52.8%, wins 8â†’80/160, Retcon hit rate 18.6%â†’7.3%; samples and online veto passed.
- [x] Run the fresh frozen fixed-duel confirmation. [PR #40](https://github.com/stephanelee/robocode/pull/40) / [report](docs/improvements/frozen-duel-confirmation.md):200rounds,55.90%mean share,107wins,6/10games passed registered observation gates. Descriptive intervals cross50%; statistical superiority/default benefit remain unestablished.
- [x] Audit foundations and add turn-latency telemetry plus a pre-game warm-up. [Audit](docs/foundations-audit.md) / [check](docs/improvements/warmup-reliability.md): retain-neutral; round-1 melee skips also hit a do-nothing probe (start-up hypothesis, not established).
- [ ] Attribute the #44 1v1 round-4 skip using `maxTurnMicros` / `slowTurns` telemetry in the next confirmation run.
- [x] Add a command-replay harness (`SurvivorBrain`/`BotPort` seam, `GameTrace`, `TraceReplay`, three live golden traces). [Report](docs/replay-harness.md).
- [x] Prove the extraction decision-identical to master on all golden traces (`MasterParityTest`); the original registered battle margin failed (cause unestablished). [Report](docs/improvements/replay-harness-parity.md).
- [x] Complete separately authorized fresh replay-extraction parity confirmation. [PR48](https://github.com/stephanelee/robocode/pull/48) / [results](docs/improvements/replay-harness-parity-redo.md):1,260 rounds/66 cold models, all registered NI guards pass (t lower -2.3054 pp > -5 pp); original380-round failure unchanged, no strength gain claimed.
- [ ] Restructure the brain into observation, movement, gun and radar modules behind the replay goldens ([audit](docs/foundations-audit.md)).
- [ ] Investigate the weak default-training-mode effect (about 15â€“18% share). First record action frequencies; whether the DQN rarely picks action 0 is an untested hypothesis. Any change is a separate registered experiment.
- [x] Test head-on/linear aim-corridor danger in minimum-risk movement. [Report](docs/improvements/aim-corridor.md): rejected (âˆ’0.58 pp score share, 5.5Ã— wall rate, Retcon hit rate rose); online stage skipped, gameplay/JAR reverted.
- [x] Test aim-corridor v2 with two-way wall smoothing. [Report](docs/improvements/aim-corridor-v2.md): rejected (âˆ’5.04 pp score share, 0 wins); the bundle recorded fewer wall collisions, without establishing the isolated wall repair's benefit; online stage skipped, gameplay/JAR reverted.
- [x] Test two-way nearest-safe wall smoothing alone. [Report](docs/improvements/wall-search.md): rejected under its correctness gate (âˆ’3.76 pp score share, wall rate 2.68Ã—); online stage skipped, gameplay/JAR reverted.
- [ ] Resolve [bullet scan visibility issue #18](https://github.com/stephanelee/robocode/issues/18) under a new approved design; the actual engine contract is proven, but the tested repair was not retained.
- [x] Extract a minimal pure targeting-geometry boundary while preserving accepted command/state behavior. [PR #25](https://github.com/stephanelee/robocode/pull/25) / [report](docs/targeting-refactor.md): 288 exact replay cases and five compatibility rounds pass; no strength claim.
- [x] Collect legal scan histories and evaluate the first separately learned motion predictor with whole-game withheld data. [PR #28](https://github.com/stephanelee/robocode/pull/28) / [report](docs/ai-motion-pilot.md): 45 rounds, 26,962 legal records; learned model rejected by offline error/containment/transfer gates and not deployed.
- [x] Evaluate the isolated two-feature old-own-wave context model. [PR #31](https://github.com/stephanelee/robocode/pull/31) / [report](docs/ai-old-wave-context-v3.md): dev passed;18 fresh valid rounds improved Retcon proxy by2.734066pp but failed SpinBot/Walls transfer. No model deployed; original invalid6-round attempt retained.
- [x] Test mixed-opponent training without changing v3 features/math. [PR #33](https://github.com/stephanelee/robocode/pull/33) / [report](docs/ai-mixed-population-v4.md): existing sample games explicitly reused for training; DEV gain1.243387pp missed2pp gate, no fresh rounds or deployment.
- [x] Test one fixed nonlinear gain-tree family with current fallback. [PR #36](https://github.com/stephanelee/robocode/pull/36) / [report](docs/ai-nonlinear-gain-v5.md): DEV gain0.964221pp missed2pp gate; no fresh rounds, retuning or deployment.
- [ ] Establish a useful learned targeting mechanism on fresh confirmation data, then independently benchmark a live gun-only candidate before promotion.
- [x] Screen legal matched-hit danger in the default duel controller. [PR54](https://github.com/stephanelee/robocode/pull/54) / [report](docs/improvements/learned-duel-danger.md): rejected after60 rounds; Diamond/Adaptive gains +4.20/+3.45pp, Retcon âˆ’16.01pp failed âˆ’5pp guard. Conditional270 rounds not run; accepted gameplay/JAR restored, all raw/models retained. Advanced-opponent strength remains open in issue53.
