# User-authorized fresh default confirmation redo

After [PR44's invalid first attempt](final-default-confirmation.md), the user explicitly authorized **"You can redo it"**. This registers exactly one NEW attempt before its outcomes. The original140 physical rounds, skipped-turn failure and unknown cause remain immutable; no original game, model, subset or score is pooled, substituted or rehabilitated. This is a disclosed user-authorized redo, not an automatic retry.

## Identical bot, setup and binding rules

- Freeze accepted source `94640028a1b5bf072713bc19ea1b608d14f9d26d`, JAR SHA256 `9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6`. No production edits, instrumentation, extra diagnostics, timeout changes or optimization.
- Exactly200 fresh rounds: ten20-round games, two independent five-game batches. [Immutable fresh layout](../../tools/fixtures/final-default-confirmation-redo/games.json): `dc946r2-b1-g01`–`g05`, `dc946r2-b2-g01`–`g05`; AIS seeds20262401–20262405 and20262411–20262415, unused in previous performance games.
- Actual default TRAIN/controller, duel override unset; ten independent missing checkpoints/private model and metrics paths/new bot processes/RNGs. Score all20 online-learning rounds per game. Every model endpoint must have finite numeric state, counter20, sequential telemetry1..20 and checkpointLoaded=false. Positive forced-controller mechanism and both official identities must validate.
- Same pinned Tank Royale1.4.0 runner/800×600 1v1 setup/100000µs turn timeout, unchanged Retcon0.2.0 source `28981c7a469ac064194b41001cd7183ec5ecf189`. Ubuntu26.04, Java17 fresh build/tests/normalized JAR parity then Java25 runtime. Reuse the unchanged v2 acquisition helper with isolated launchers; no historical module globals are changed. Record exact source/runtime/asset/bundle/launcher hashes.

ALL original confirmation gates remain binding, using unrounded official equal-game score shares:

1. Overall ten-game mean share **>50%**.
2. Official first-place round count **>100/200**.
3. **>5/10 games** individually above50% share.
4. **Both** five-game batch means **>50%**.
5. One-sided95% Student-t(df9) game-mean lower bound **>50%**, critical `1.8331129326536335` times sample SD/sqrt(10).
6. All200 rounds/ten official/native arrays/ten independent cold models/default mechanism validate, with **zero AIS skipped turns**.

No performance early stop; acquisition invalidity may stop a batch as before. No replacement seed/game, rescue run, second retry after this attempt, favorable subset, gate waiver or tuning. Preserve every partial outcome. Invalid/incomplete/failed rules mean **not confirmed**, without reversing accepted PR43. No aggregate or binding t gate is evaluated for an incomplete population. Both jobs use fail-fast=false so a separate batch already running may finish after the other becomes invalid.

A pass completes this finite fresh confirmation; either outcome ends the automatic benchmark sequence with no further run scheduled. Game-level t assumptions remain approximate; within-game rounds adapt and are dependent. Any round-win Wilson interval is descriptive, not200 independent trials. This tests only this frozen bot against pinned Retcon/default20-round regime, not all opponents or tournaments. Seeds initialize AIS RNG, not engine starts.

## Publication and evidence

Independent exact-source/workflow preclear precedes launch. Preserve the immutable Git plan/layout, executed source/workflow, both original batch reports, all actual per-round metrics, official/native results, models and partial failures, plus original runtime/launcher/source pins. Do not duplicate runnable JARs. Update `docs/improvements.md` with measured results, confirmed/not-confirmed disposition, PR link and evidence before independently reviewed merge and required routine CI. The previous attempt and its NOTCONFIRMED result remain unchanged alongside this separate record.
