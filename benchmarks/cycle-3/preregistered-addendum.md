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

Pending. No cycle-3 matches start before cycle 2's final accept/reject decision and accepted-master integration. If rejected, revert gameplay/JAR while retaining the harness extension and measured evidence.

## Prospective diagnostic addendum

The original registration was published at `39af977116d27e0792fb1d653f8a743cf81568ab`. Before any actual cycle-3 matches, independent review requested explicit saved-targeting-evidence diagnostics and precise wording about trained-policy seeds. This addendum changes neither the candidate nor budgets, seeds, acceptance thresholds, or stopping rules. The match workflow will read this complete amended plan from its newly published immutable commit; both registration revisions remain in Git history.

Record total saved class-GF mass, number of nonzero bins, and mass/nonzero counts per opponent class for each trained checkpoint. Report these as mechanism diagnostics only. Zero mass means the proposed saved-statistics mechanism is inactive for that seed; retain that seed and its matches in the primary analysis. Do not retrain, discard, substitute, or select checkpoints according to these values. The comparator records diagnostics without mutating the checkpoint, and tests cover an inactive seed still receiving its full budget.

Seeds control the incumbent's training randomness. Each evaluation arm loads that seed's byte-identical trained weights/statistics; a frozen greedy policy does not generally use RNG. Matching seed labels do not pair arena starts or scheduling, and the report must not describe trained comparison models as cold networks.
