# Robocode TODO

Repository: https://github.com/stephanelee/robocode

Based on a source review of `master` at commit `fb735035811e60bf30366a0952b5dfe1b8e079ac`. The Java 17 build has since been verified locally and in GitHub Actions (PR #1); battle performance has not yet been verified.

## First milestone: reliable baseline

- [ ] Fix coordinate calculations across movement, targeting, bullet projection, and wall avoidance to follow Tank Royale's coordinate conventions.
- [ ] Fix initial targeting so empty GuessFactor statistics default to the centre bin.
- [ ] Align learning with movement by executing the action the DQN actually selects.
- [ ] Correct enemy-bullet handling: the pinned API exposes the bot's own bullets in tick events; use inferred enemy waves for enemy threats.
- [ ] Fix round-end learning to preserve pending rewards and record terminal transitions for both death and survival.
- [ ] Correct enemy turn-rate estimates for elapsed turns between scans.
- [ ] Add focused tests for geometry, targeting defaults, and learning transitions.
- [x] Add a Gradle wrapper and CI for reproducible builds. (PR #1; local Java 17 build and GitHub Actions verified.)
- [ ] Write a README and update stale architecture and bot metadata documentation.
- [ ] Run baseline battles and record the results.

## Validate and improve learning

- [ ] Repair wave-surfing statistics using bullet-hit evidence rather than only the bot's position when a wave passes.
- [ ] Create automated battle benchmarks against a fixed opponent roster.
- [ ] Record score, placement, hit rate, wall collisions, and skipped turns.
- [ ] Separate training and evaluation modes, with reproducible random seeds.
- [ ] Compare fixed strategies against untrained and trained DQN policies before expanding the model.
- [ ] Improve saved-model handling with version checks and atomic writes.

## Later improvements

- [ ] Add a results dashboard for scores, hit rates, survival, and learning progress.

First milestone is complete when the build is reproducible, movement and targeting checks pass, and baseline battle results are recorded.
