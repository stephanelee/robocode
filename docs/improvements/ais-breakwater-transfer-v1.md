# AIS transfer of Breakwater's movement: prospective whole-product comparison

The user requested transfer of Breakwater's strengths into AISurvivor and fresh benchmarks. This supersedes the earlier requirement to beat Diamond with a separate challenger before attempting AIS integration. Track under issue53. Do not merge or modify the held external Breakwater PRs.

## Intervention and limitations

Transfer the movement policy from exact Breakwater0.3 source b9f61e536c0ff5860f6aabc0f6d3f4a11685a3d9: 430px spacing, 100px wall stick, four full-speed/reversal paths, a held path with 16-turn/wave-pair replanning, seeded near-equal-risk selection within0.08, and a bounded32-entry signed incoming-hit GF memory. Inputs are legal scans, own state and legally inferred/matched incoming waves. Preserve pinned engine movement order and reset semantics. Restrict the intervention to the existing default TRAIN single-opponent controller action, preserving executed-action learning credit. Other policies and melee retain their accepted behavior.

Keep AIS's current GF/virtual gun, power, firing gates, radar, energy bookkeeping and DQN unchanged. Do not copy Breakwater's shielding or the v0.3 envelope gun. That gun fired only3shots against Diamond, and its17zero-shot wins against AIS do not establish its causal contribution. Breakwater0.2 had the same movement but a different gun and a different one-game AIS outcome. Thus this transfer is a source-grounded hypothesis about the complete movement intervention, not a proven explanation of the20/20 transfer sweep or a claim that energy conservation is isolated.

The baseline is accepted AIS source9b1624da2d9b09ec9cc4f580b51bc82edd2c1485/JARb41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800. The candidate source/JAR will be frozen in the machine protocol after implementation/tests, before independent premeasurement clearance or play. Breakwater opponent is frozen0.3 JAR e6c98feef83e083149df782d643496fb6b6fe3fab33c95356ea2634981c6baf0 from the exact source above, not its older variants.

## Fixed population and execution

Exact new game IDs, unique own seeds, stages and arm order are in tools/fixtures/ais-breakwater-transfer/protocol.json. No old games or checkpoints enter this comparison. Both AIS arms use real default TRAIN with duel-policy/action overrides unset, absent private model/metrics paths, and fresh processes per game. Self games have distinct official descriptor names, seeds and paths, bound in each bot's launcher; no shared checkpoint. Opponents reset process state between games, with within-game learning retained as implemented. Arm blocks are arithmetic comparison groups, not paired initial positions or controlled opponent random seeds.

Use official Runner/server1.4.0, unchanged oneVsOne defaults at800x600 with100000microsecond turn timeout;900second per-game result timeout and990second owned-process bound. Java17 build/tests/committed-JAR parity precede Temurin25.0.5+8 runtime. Native Diamond/Adaptive use the existing source/wheel lock and Python3.13.17. Reuse the six exact JDK/Python file hashes already retained with Breakwater's hosted runtime; compare them before play. All GitHub jobs run exactly ubuntu-26.04. Keep the interactive game/server7654 untouched.

One90-physical-round development screen:

- Breakwater0.3, Retcon, Diamond Port and Adaptive Prime:10rounds per arm/opponent =80.
- Candidate head-to-head against frozen accepted AIS:10rounds =10.

Complete every registered screen game unless validity fails. If and only if all screen gates pass, run a genuinely fresh confirmation:

- Three20-round blocks per arm against each of the four external opponents =480.
- Three10-round candidate-versus-accepted self games =30.
- Target, SpinBot and Walls:5rounds per arm =30.

Maximum630physical rounds,42games and46private AIS checkpoints, plus8private Breakwater endpoints. Screen data are developmental and are never pooled into fresh confirmation. Stop on invalid acquisition and preserve the attempt/unplayed budget; no replacement games, favorable reruns, seed selection, threshold relaxation or tuning after outcomes. A score failure completes its registered stage; it does not cancel remaining games in that stage. No additional games are authorized by an invalid result.

## Gates frozen before outcomes

Score share is official own totalScore divided by both official scores. External gains are candidate minus baseline score share in percentage-point units, averaged equally across game blocks. Official first places determine wins; retain and disclose raw callback differences rather than rewriting rows.

Screen must satisfy all:

- Breakwater: gain>=5percentage points, candidate share>50%, candidate wins>5/10.
- Retcon: gain>=-5points, candidate share>50%, candidate wins>5/10.
- Diamond and Adaptive: each gain>=-5points and their equal-opponent mean gain>=0.
- Self: candidate share>50%, candidate wins>5/10.
- For each opponent and self comparison, candidate wall contacts per1000reported turns<=baseline rate+10. Reported turn counters are not exact alive time.
- Zero skipped turns in BOTH AIS arms. Breakwater must also report zero skips in its comparisons. Complete valid source/identity/runtime/official/telemetry/checkpoint provenance.

Fresh confirmation must satisfy all:

- Breakwater: equal-game mean gain>=5points, all3block gains>0, candidate mean share>50%, candidate wins>30/60.
- Retcon: equal-game mean gain>=-5points AND its one-sided95% Student-t game-difference lower bound>=-5points. Use exactly3differences, df2 critical2.919985580355516 and sample standard deviation divided by sqrt(3). Every candidate Retcon game share>50%, candidate wins>30/60.
- Diamond and Adaptive: each equal-game mean gain>=-5points and their equal-opponent mean gain>=0. This preserves this bounded roster comparison; it does not establish Diamond/Adaptive wins.
- Self: equal-game candidate mean share>50%, wins>15/30, at least2/3games above50%.
- Each sample candidate share>=baseline share-5points.
- The same per-opponent/self/sample wall-rate guard, zero skips in BOTH AIS arms, Breakwater zero-skips and complete provenance/validity.

The Retcon t bound relies on sparse game-level assumptions; no IID-round inference or universal ranking follows. This is a bounded transfer test aimed at improvement against Breakwater while preserving existing strength, not final proof of beating Diamond. Report all opponents, game variation, shooting efficiency, damage, walls, collisions, skips and policy activation regardless of decision. No isolated movement subcomponent, neural learning benefit or energy-gate causality claim.

## Review, evidence and promotion

Publish exact source/JAR and immutable registration before games; independent reviewer clearance of the exact implementation/driver/workflow authorizes execution. Implementers never self-merge. Record all raw rows, official outcomes, logs, private endpoint models, executed tooling, runtime/source/opponent/asset hashes, failed attempts and decision output. Update docs/improvements.md, this report and issue53 after the experiment. A reject restores production gameplay/JAR and preserves the candidate patch/artifact/evidence. A retain still needs independent result audit and all protected-master required checks before merge. No force push, evidence rewrite or AI/Codex author trailers.
