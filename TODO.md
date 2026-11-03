# Robocode TODO

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
- [x] Test shot-reactive 1v1 duel movement against Retcon's two guns. [Report](docs/improvements/duel-surf.md): v1 passed (share 14.8%→58.6%, Retcon hit rate 18.9%→5.3%); the review found simulation-physics defects, so gameplay is not merged on v1 evidence.
- [x] Measure the engine-conformant duel controller in its registered fresh comparison. [v2 report](docs/improvements/duel-surf-v2.md): **retained**; fixed-policy share 17.4%→52.8%, wins 8→80/160, Retcon hit rate 18.6%→7.3%; samples and online veto passed.
- [ ] Register a fresh frozen-version confirmation of beating Retcon (score share > 50%, majority of rounds and blocks).
- [ ] Investigate the weak default-training-mode effect (about 15–18% share). First record action frequencies; whether the DQN rarely picks action 0 is an untested hypothesis. Any change is a separate registered experiment.
- [x] Test head-on/linear aim-corridor danger in minimum-risk movement. [Report](docs/improvements/aim-corridor.md): rejected (−0.58 pp score share, 5.5× wall rate, Retcon hit rate rose); online stage skipped, gameplay/JAR reverted.
- [x] Test aim-corridor v2 with two-way wall smoothing. [Report](docs/improvements/aim-corridor-v2.md): rejected (−5.04 pp score share, 0 wins); the bundle recorded fewer wall collisions, without establishing the isolated wall repair's benefit; online stage skipped, gameplay/JAR reverted.
- [x] Test two-way nearest-safe wall smoothing alone. [Report](docs/improvements/wall-search.md): rejected under its correctness gate (−3.76 pp score share, wall rate 2.68×); online stage skipped, gameplay/JAR reverted.
- [ ] Resolve [bullet scan visibility issue #18](https://github.com/stephanelee/robocode/issues/18) under a new approved design; the actual engine contract is proven, but the tested repair was not retained.
- [x] Extract a minimal pure targeting-geometry boundary while preserving accepted command/state behavior. [PR #25](https://github.com/stephanelee/robocode/pull/25) / [report](docs/targeting-refactor.md): 288 exact replay cases and five compatibility rounds pass; no strength claim.
- [x] Collect legal scan histories and evaluate the first separately learned motion predictor with whole-game withheld data. [PR #28](https://github.com/stephanelee/robocode/pull/28) / [report](docs/ai-motion-pilot.md): 45 rounds, 26,962 legal records; learned model rejected by offline error/containment/transfer gates and not deployed.
- [x] Evaluate the isolated two-feature old-own-wave context model. [PR #31](https://github.com/stephanelee/robocode/pull/31) / [report](docs/ai-old-wave-context-v3.md): dev passed;18 fresh valid rounds improved Retcon proxy by2.734066pp but failed SpinBot/Walls transfer. No model deployed; original invalid6-round attempt retained.
- [x] Test mixed-opponent training without changing v3 features/math. [PR #33](https://github.com/stephanelee/robocode/pull/33) / [report](docs/ai-mixed-population-v4.md): existing sample games explicitly reused for training; DEV gain1.243387pp missed2pp gate, no fresh rounds or deployment.
- [x] Test one fixed nonlinear gain-tree family with current fallback. [PR #36](https://github.com/stephanelee/robocode/pull/36) / [report](docs/ai-nonlinear-gain-v5.md): DEV gain0.964221pp missed2pp gate; no fresh rounds, retuning or deployment.
- [ ] Establish a useful learned targeting mechanism on fresh confirmation data, then independently benchmark a live gun-only candidate before promotion.
