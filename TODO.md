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
