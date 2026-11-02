# Comparing two bot versions

`tools/compare.py` evaluates immutable copies of two AISurvivor bundles against the pinned Retcon opponent. Build both versions first and retain their separate directories containing the JAR, descriptor, and launchers. Descriptors and launchers must match between versions. Evaluation never trains; an explicit `--train-previous` option can prepare per-seed incumbent checkpoints before frozen evaluation.

Example primary experiment, using JDK 25 on `PATH`:

```sh
python tools/compare.py \
  --previous-bot-dir /path/to/previous/AISurvivor \
  --candidate-bot-dir AISurvivor \
  --previous-source PREVIOUS_COMMIT --candidate-source CANDIDATE_COMMIT \
  --plan docs/improvements/cycle-1.md --plan-commit PREREGISTRATION_COMMIT \
  --policy untrained --seeds 20261102,20261103,20261104 \
  --rounds 20 --timeout 300 --output build/comparison-primary
```

Policies:

- `untrained`: fresh seeded networks and frozen evaluation; no checkpoint is loaded or created.
- `fixed-minrisk`: fixed minimum-risk action, no network/checkpoint learning. Seeds label game replicates but do not alter the fixed strategy.
- `trained`: add `--checkpoint /path/to/model.dat --training-rounds 20`, where the round count must equal the checkpoint's training counter. The input model is snapshotted and copied separately for every arm/game. All input and per-game copies must remain unchanged.

For a controlled targeting-statistics experiment, use `--policy trained --train-previous --training-rounds 20` instead of importing a checkpoint. The harness trains only the previous version for exactly 20 rounds separately for each of the three seeds, snapshots each resulting model, and gives both arms byte-identical copies of that seed's checkpoint. It validates the persisted training counter and evaluation telemetry, records all training battle results and model hashes, and excludes training scores from the primary decision. Planned budgets are 60 training rounds plus 120 evaluation rounds; early rejection records the actual smaller budget explicitly. Changing the training budget or combining this option with an imported checkpoint is rejected.

Checkpoint diagnostics report saved class-targeting mass and nonzero bins, including per-class counts. Zero evidence is recorded as an inactive mechanism; it never triggers extra training, model replacement, or exclusion of a seed. Trained comparisons share checkpoint bytes within a seed rather than comparing cold networks, and frozen-policy seed labels do not imply paired game states.

Exactly three distinct Java-compatible integer seeds are required. Primary experiments enforce exactly 20 rounds per version per seed; short runs must be diagnostic and cannot yield a retain decision. Each seed runs both arms, alternating launch order between blocks. The plan path must be inside this repository; the harness reads its immutable Git blob at the complete `--plan-commit` SHA, rather than trusting the working-tree file or an arbitrary label. The output directory must be empty. The artifact contains both complete bundle snapshots and hashes, input checkpoint when applicable, registered plan copy/hash, source identifiers, opponent provenance, runtime, official scores, raw per-round telemetry, and runner logs. `comparison.json` is updated after each arm so failures retain partial evidence. A failed or invalid battle is not a measured loss.

Primary decision thresholds are defined in the preregistered plan and encoded in the harness; valid experiments can finish successfully with a `reject` decision. CI success means the experiment ran and validated, not that the candidate improved. An early catastrophic rejection is recorded explicitly with its smaller actual round budget.

Use `--purpose diagnostic` for a secondary comparison. It reports the same measurements but does not issue an independent retain/reject decision. The temporary cycle-1 measurement workflow was configured to run a three-block, five-round fixed-policy diagnostic only after the primary experiment passed; cycle 1 was rejected, so that diagnostic did not run. Cycle-specific measurement jobs are removed after evidence is retained to prevent accidental repeats.

Compare score shares at equal budgets, rather than raw score across different round counts. Report every game block and normalize wall collisions by bot turns. Win-rate Wilson intervals are descriptive because rounds share an adaptive opponent. Bot seeds do not seed arena positions or engine scheduling, and a frozen loaded model may not use RNG at all. These are unpaired Retcon adaptation experiments, with no held-out generalization claim.
