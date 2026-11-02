# Cycle 3: use saved targeting statistics during frozen evaluation

Issue: [#12](https://github.com/stephanelee/robocode/issues/12). This plan is registered before candidate outcomes. Preparation starts from accepted master `18f2659f6f55f10a7a30b1ec54460eb7159ce101`; the actual previous version will be the accepted master after cycle 2, recorded by source revision and JAR hash before measuring. The candidate will inherit any accepted wall-smoothing change and differ only in targeting-statistics confidence.

## Hypothesis and sole gameplay change

The current GuessFactor blend trusts per-enemy statistics according to scan count: `min(1, scanCount / 30)`. Frozen evaluation deliberately never updates its fresh, local per-enemy bins. After 30 scans, their zero values receive all the weight and the saved class statistics are discarded.

The candidate measures confidence from actual evidence in the selected per-enemy segment: `min(1, sum(segment bins) / 30)`, with absent/empty bins giving zero confidence. The existing threshold and blend shape remain; only their evidence source changes. This allows a frozen, loaded policy to keep using its saved class targeting statistics. Evaluation remains frozen: no DQN weights, class statistics, local targeting/surfing bins, temperature, or checkpoint may change. No movement, firepower, rewards, neural architecture, or exploration change is included.

## Preregistered experiment

Use the accepted previous JAR to train **20 rounds separately for each bot seed 20261102, 20261103, 20261104** against the same pinned Retcon opponent. Each training game starts without a checkpoint. Snapshot each resulting model and record its SHA256 and persisted training counter of 20. Copy that seed's exact checkpoint independently into both version arms; neither arm trains during comparison. The candidate never creates its own training model.

Then evaluate **20 rounds per seed per version**, with previous/candidate order alternating between seed blocks. Planned budget: **60 incumbent training rounds + 120 frozen evaluation rounds = 180 rounds**. Training scores are retained as provenance but excluded from every primary score comparison. Checkpoint hashes, loaded status, and unchanged training counters must be verified in both evaluation arms. Every game uses separate directories and a fresh opponent process.

The primary outcome is the equal-weight mean of the three evaluation-game AISurvivor score shares, `AIS score / (AIS score + Retcon score)`. Retain the candidate only if every criterion holds:

- Mean score share improves by at least **5 percentage points**.
- At least **two of three** seed blocks improve.
- No block loses more than **5 percentage points**.
- Candidate aggregate wins are no more than **three below** previous aggregate wins.
- Candidate records **zero skipped turns**.
- Candidate wall collisions per 1,000 bot turns are at most **1.2 times** previous; if previous has zero collisions, candidate must also have zero.

The comparator may reject early after the first completed evaluation block only if candidate score share drops by at least 15 percentage points, candidate has zero wins, and previous has at least four. Report the actual reduced budget if this happens. Process failure, model mutation, wrong counter/seed, or invalid telemetry stops the experiment and is not scored as a loss. Retain any failed attempt separately before a corrected retry.

No additional fixed-policy diagnostic is required in this cycle: the hypothesis concerns saved targeting statistics, and fixed mode does not load the trained model. Report actual bullet hit rate, damage dealt/taken, wins and descriptive Wilson intervals, wall rate, and skipped turns alongside the primary outcome.

## Evidence and limits

Retain immutable previous/candidate bundles and all three trained checkpoint snapshots in the run artifact, with exact source/JAR/runtime/opponent/runner hashes, registered plan copy/hash, launch order, round budgets, raw training/evaluation battle results and per-round telemetry, and runner logs. The harness must enforce the exact 20-round training and evaluation budgets for each of three distinct seeds and exclude training games from the decision.

Bot seeds control incumbent learning randomness. Matching checkpoints ensure equal starting weights/statistics within each seed's two arms, but arena starts and server scheduling are unpaired/unseeded. These are Retcon adaptation trials, not a held-out evaluation. Three blocks give limited uncertainty information, and rounds within a game share an adaptive opponent, so Wilson win intervals are descriptive. Report every block, including regressions; do not change thresholds after seeing outcomes.

## Results

**Decision: reject the gameplay candidate.** All **60 incumbent training + 120 frozen evaluation rounds** completed. Mean evaluation score share rose from 12.294% to 13.440% (+1.146 percentage points), below the registered 5-point minimum. Seed 20261102 lost 5.037142 points, beyond the registered 5-point maximum loss. These two failed criteria were applied without changing or rounding the thresholds. Every other guard passed.

| Seed | Previous AIS / Retcon score | Candidate AIS / Retcon score | Previous share | Candidate share | Change | Wins previous / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261102 | 338 / 2993 | 158 / 2934 | 10.147% | 5.110% | -5.037 pp | 0 / 0 |
| 20261103 | 390 / 3139 | 598 / 3151 | 11.051% | 15.951% | +4.900 pp | 0 / 0 |
| 20261104 | 603 / 3242 | 758 / 3178 | 15.683% | 19.258% | +3.575 pp | 0 / 1 |

Each score pair covers 20 evaluation rounds; the primary mean weights the three games equally. Training scores are excluded. Previous won 0/60 rounds, candidate 1/60; descriptive 95% Wilson intervals are 0%–6.017% and 0.295%–8.855%. The registered dependence and sample-size limitations apply.

Actual hit rate fell from 154/1,121 (13.74%) to 116/1,096 (10.58%). Damage dealt fell from 1,209.74 to 1,093.42; damage taken fell from 4,755.44 to 4,549.07. Wall collisions fell from 493 in 22,334 turns to 256 in 23,018 turns (22.07 → 11.12 per 1,000 turns). Both versions recorded zero skipped turns. These secondary measurements do not override the primary decision.

### Shared-model diagnostics

The unchanged incumbent trained once for 20 rounds per seed. Every frozen evaluation loaded its seed's exact model, kept a training counter of 20, and left its SHA256 unchanged. No model was discarded, substituted, or retrained based on targeting evidence.

| Seed | Saved class-GF mass | Nonzero bins | Checkpoint SHA256 |
| --- | ---: | ---: | --- |
| 20261102 | 372 | 153 | `8e0be07f656c58bac56b9c0b8889d414d0ca86da0f649775780b8c7d54a9b3fe` |
| 20261103 | 311 | 137 | `2fbaf2b750d44ff367e9c9c733c53c4b1cd7bf32cdda468163ccfbbbf3384995` |
| 20261104 | 295 | 132 | `5a8300d9c8f5c88893515784fddf91dcd98ecc089a319cc1bbf61ee8ec7d580d` |

All three checkpoints contained saved evidence, mostly in the linear class; full per-class counts are in the JSON. This establishes that saved statistics existed for the mechanism under test, not that their learned aim offsets were accurate. Mixed per-seed performance and the lower aggregate hit rate do not support retaining the candidate under the registered rule.

### Provenance and final disposition

- [Successful measured run](https://github.com/stephanelee/robocode/actions/runs/41495051167), job `125447677502`, checked out candidate `3b47803f7b2321b84abecd678d2c22fd65459ebb` exactly. Previous source: accepted cycle-2 master `5ef03cf74f8709e5b86385a598cde8def0f7abdf`.
- Original plan `39af977116d27e0792fb1d653f8a743cf81568ab` and prospective amendment `21d4a9622c9013aa42af2b57dd0a4ffa1aed86e4` were loaded from exact Git blobs before matches. [Both immutable copies](../../benchmarks/cycle-3/) and hashes are retained.
- [Canonical comparison JSON](../../benchmarks/cycle-3/comparison.json), all nine games' official results/telemetry/logs, candidate source patch, and all three original [numeric checkpoints](../../benchmarks/cycle-3/checkpoints/) are checked in for durable verification. These research models live outside the bot directory and are not automatically loaded by normal launches.
- [Original complete artifact](https://github.com/stephanelee/robocode/actions/runs/41495051167/artifacts/13165233666) includes both immutable version bundles and every per-arm model copy. Artifacts may expire; canonical checkpoint bytes remain in Git. Previous JAR SHA256: `0d120a0461a7bf32b1c6d6277da34f9480309f255caa44e168e6bcdf8636bff2`; candidate: `712283e805f947dd6bced62e7af95eb91b28fd76bf94567804f088ea5baa945d`.
- Build: Java 17. Matches: Temurin 25.0.5+8-LTS, Tank Royale 1.4.0 runner/server, unchanged Retcon 0.2.0 source `28981c7a469ac064194b41001cd7183ec5ecf189` and pinned bot API 1.3.1. Exact dependency hashes are recorded in the JSON.

The final PR restores gameplay source and the tracked JAR to accepted master, and removes the temporary workflow and candidate-only targeting tests. The shared-checkpoint harness, diagnostics, raw evidence, and report remain. Issue #12 stays open; this targeting change is not retained. All three planned cycles have measured outcomes; no additional candidate was tested here.

## Prospective diagnostic addendum

The original registration was published at `39af977116d27e0792fb1d653f8a743cf81568ab`. Before any actual cycle-3 matches, independent review requested explicit saved-targeting-evidence diagnostics and precise wording about trained-policy seeds. This addendum changes neither the candidate nor budgets, seeds, acceptance thresholds, or stopping rules. The match workflow will read this complete amended plan from its newly published immutable commit; both registration revisions remain in Git history.

Record total saved class-GF mass, number of nonzero bins, and mass/nonzero counts per opponent class for each trained checkpoint. Report these as mechanism diagnostics only. Zero mass means the proposed saved-statistics mechanism is inactive for that seed; retain that seed and its matches in the primary analysis. Do not retrain, discard, substitute, or select checkpoints according to these values. The comparator records diagnostics without mutating the checkpoint, and tests cover an inactive seed still receiving its full budget.

Seeds control the incumbent's training randomness. Each evaluation arm loads that seed's byte-identical trained weights/statistics; a frozen greedy policy does not generally use RNG. Matching seed labels do not pair arena starts or scheduling, and the report must not describe trained comparison models as cold networks.
