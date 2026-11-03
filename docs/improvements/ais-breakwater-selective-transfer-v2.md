# AIS movement plus Breakwater-style selective fire: new prospective comparison

The user requested transfer of Breakwater's strengths into AISurvivor and fresh benchmarks. This supersedes the earlier requirement to beat Diamond with a separate challenger before attempting AIS integration. Track under issue53. Do not merge or modify the held external Breakwater PRs.

## Intervention and limitations

Transfer the movement policy from exact Breakwater0.3 source b9f61e536c0ff5860f6aabc0f6d3f4a11685a3d9: 430px spacing, 100px wall stick, four full-speed/reversal paths, a held path with 16-turn/wave-pair replanning, seeded near-equal-risk selection within0.08, and a bounded32-entry signed incoming-hit GF memory. Inputs are legal scans, own state and legally inferred/matched incoming waves. Preserve pinned engine movement order and reset semantics. Restrict the intervention to the existing default TRAIN single-opponent controller action, preserving executed-action learning credit. Other policies and melee retain their accepted behavior.

The new policy has a separate legal wave ledger so legacy surfing's wave removal cannot consume its learning event first. Bridge AIS's prior scan/aim turn to physical creation at aimTurn+1, freezing launch reference and lateral velocity from the corresponding legal own-history frame. Queue actual HitByBullet observations until inference for that tick is processed, then require a unique owner/power/actual-ray/along-radius match. This first transfer learns from matched direct hits only; it deliberately omits Breakwater's additional bullet-collision labels because AIS does not currently forward that event. Terminal queued hits with no subsequent decision are unknown and cleared at next round start. Thus exact pure movement comparisons do not establish complete-bot equivalence to Breakwater. Preserve this distinction in results.

Keep AIS's current GF/virtual gun direction, power, radar, energy bookkeeping and DQN unchanged. Preserve its existing shouldFire gate as a required prerequisite, then add the selective-fire veto described below. Do not copy shielding or replace/retarget the AIS gun. Breakwater's original envelope gun fired only3shots against Diamond, and its17zero-shot wins against AIS do not establish its causal contribution. Breakwater0.2 had the same movement but a different gun and a different one-game AIS outcome. This complete movement-plus-selectivity intervention is an unproven source-grounded hypothesis; energy conservation and movement effects are not isolated.

The baseline is accepted AIS source9b1624da2d9b09ec9cc4f580b51bc82edd2c1485/JARb41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800. The candidate is source1409f16497de7a82c11ae55d8f38d69e10dfeeb6/JARaf94f4eeaadbd2a4a9738a9ec3349de9438d044ad2747f3789bdf47015ecbbf7, frozen after a Java17 build with112tests,0failures and1existing skip. The machine protocol records these exact identities before play. Breakwater opponent is frozen0.3 JAR e6c98feef83e083149df782d643496fb6b6fe3fab33c95356ea2634981c6baf0 from the exact source above, not its older variants.


## New hypothesis after the V1 rejection

The prior movement-only candidate remains REJECTED after all90screen rounds: Breakwater gain+1.939pp/0of10wins failed all its primary gates, despite encouraging Adaptive/self/Retcon secondary observations. Production baseline was explicitly restored and all956original file payloads, source/JAR/patch and rejection are retained. No prior game is pooled into this experiment or called confirmation.

This new whole-program candidate adds the previously omitted source-grounded shot-selectivity idea. Against Breakwater, V1 still fired837shots for42hits and lost every round while recording only6.1incoming damage. That motivates a firing-economy hypothesis, without establishing why each round ended. The new source and registered population are different; every acceptance threshold and budget per experiment is unchanged. Maximum additional630rounds; combined V1+V2 maximum720because V1 stopped after90.

The fixed guard rolls the three source69 STOP/CW/CCW motion hypotheses with its650px orbital prior,160px wall stick,90-turn horizon and exact pinned movement/wall/annulus geometry. At the actual current AIS gun heading and actual proposed shot power, at least two distinct hypothetical path footprints must be intersected. These are three legal motion hypotheses, NOT an exhaustive reachable set, calibrated hit probability, or Diamond private state. There is no angle maximization, new gun, changed power, tuning or model fit. Only the initial1v1defaultTRAIN scope uses the veto; all older policy/melee branches stay accepted.

Every shot must still satisfy the old AIS shouldFire gate. The zero-energy finisher bypasses this new veto while retaining that old gate. Actual setFire acceptance and confirmed BulletFired callbacks still govern all shot cohorts; a veto cannot create a learned shot label. Four per-round int counters record old-fire eligible, guard passed, blocked and zero-energy finisher proposals; eligible=passed+blocked, finisher<=passed, eligible<=newmovementturns. Proposal counts are not confirmed shot counts.

Starvation is an explicit risk: Breakwater0.3 fired only3shots against Diamond and lost20/20. This guard is deliberately more selective than AIS and could remove useful offence or create draws. Its combined candidate must pass the same Breakwater, Retcon, advanced, self, sample and safety gates; no useful V1 secondary result can waive a failure. New trace tag breakwater-v2-selective-fire distinguishes the composition from V1 movement-only and untagged historical replay, with all strict assertions retained.

The new registered source is1409f16497de7a82c11ae55d8f38d69e10dfeeb6/JARaf94f4eeaadbd2a4a9738a9ec3349de9438d044ad2747f3789bdf47015ecbbf7. It is byte-identical in all main/test/bundle files to independently tested local e395397:112Java17testcases,0failures,1existing skip;1000geometry support comparisons match source69. Exact immutable registration and final workflow head precede play. New202641/202642/202643 own-seed population and ais-bw-v2 IDs are disjoint from V1; this is not a favorable rerun.

Frozen Breakwater JAR/descriptor are staged from their lossless original V1 payloads, with length/SHA verification and exact e6c98 JAR pin. Only executable static assets are reused; no old opponent or AIS checkpoint enters any game. The original69endpoint validation fixture is now vendored byte-for-byte for offline tests, not used as controller data.

## Fixed population and execution

Exact new game IDs, unique own seeds, stages and arm order are in tools/fixtures/ais-breakwater-selective-transfer/protocol.json. No old games or checkpoints enter this comparison. Both AIS arms use real default TRAIN with duel-policy/action overrides unset, absent private model/metrics paths, and fresh processes per game. Self games have distinct official descriptor names, seeds and paths, bound in each bot's launcher; no shared checkpoint. Opponents reset process state between games, with within-game learning retained as implemented. Arm blocks are arithmetic comparison groups, not paired initial positions or controlled opponent random seeds.

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


## Measured outcome: reject at the 90-round screen

The sole [run41730684973](https://github.com/stephanelee/robocode/actions/runs/41730684973), execution9c76456ec85216e38b0b0ef57f22709e34764f70, completed all9screen games/90physical rounds. Both AIS arms produced100raw rows and10distinct cold finite counter10 checkpoints; two private Breakwater endpoints and two complete Retcon exports are preserved. All AIS and Breakwater recorded skips were zero. The frozen decision is **reject-at-screen**, with **zero of540conditional confirmation rounds played**. The original canonical report and all gates remain unchanged.

| Opponent | Accepted share / official wins | Candidate share / official wins | Share gain |
| --- | --- | --- | --- |
| Breakwater0.3 |27.007299% /0of10 |35.714286% /3of10 |+8.706986pp |
| Retcon |77.376426% /9of10 |67.830882% /9of10 |-9.545544pp |
| Diamond Port |4.942966% /0of10 |0% /0of10 |-4.942966pp |
| Adaptive Prime |25.383707% /1of10 |15.574302% /2of10 |-9.809405pp |
| Direct head-to-head |25% /0of10 |75% /10of10 |Same-game candidate share75% |

Exactly five checks failed: Breakwater winning share and wins, Retcon relative noninferiority, advanced-opponent individual noninferiority, and advanced mean noninferiority. Breakwater's gain alone passed; its35.7%share/3wins did not meet the registered winning requirements. Retcon's absolute share/win guard passed but its-9.55pp relative guard failed. Diamond narrowly remained within its-5pp individual floor, but Adaptive did not; the equal-opponent mean gain was negative. Self, wall-rate and zero-skip checks passed. These positive checks cannot waive the failures.

The guard was active and severely restrictive:

| Candidate game | Old-fire eligible | Passed / blocked | Confirmed shots / hits |
| --- | --- | --- | --- |
| Breakwater |9526 |5 /9521 |5 /0 |
| Retcon |7723 |15 /7708 |15 /9 |
| Diamond |8607 |0 /8607 |0 /0 |
| Adaptive |6788 |1 /6787 |1 /0 |
| Self |10739 |0 /10739 |0 /0 |

No zero-energy finisher proposals were recorded. Every row's counters reconcile; proposal passes are not generally confirmed shots, although totals coincide here. Candidate damage dealt/taken was0/19.95 against Breakwater,89.200001/157.8 against Retcon,0/1008.741826 against Diamond,0/860.8 against Adaptive and0/200.4 in self. These are raw callback telemetry, not a complete official damage ledger. Source and observed counts establish starvation in this cohort, not why every round ended or that any untested relaxation would help.

Official first places remain primary as registered. Raw AIS won flags differ from official counts: Breakwater candidate2vs3; Retcon accepted7vs9 and candidate7vs9; self candidate6vs10. The Breakwater opponent against accepted AIS records7raw won flags versus10official first places; its independent endpoint/round official increments reconcile. Raw records are preserved without correction. No unrecorded callback order is inferred from these aggregates.

All956original file payloads are retained losslessly as550SHA-addressed unique payloads, including every endpoint, staged JAR/source/script/cache and native export. [Archive and offline reproduction](../../benchmarks/ais-breakwater-selective-transfer-v2/README.md). Original service ZIP SHA256f953c44b769d89f0721916c5ac87a83e650afaa4a2af389fcc4ceea579e7a0e0 matches artifact13276183049. The exact executed workflow/tooling and candidate binary-capable patch remain available after workflow retirement.

Production source/tests/bundle are restored byte-for-byte to accepted9b1624/JARb41b; no failed gameplay is promoted. V1's separate90-round rejection and all original bytes remain unchanged. No game pooling, favorable replacement, threshold waiver, new tuning or further candidate is authorized by this result. One unpaired10-round game per arm/opponent and one self game are development screening only, not universal rankings or an isolated causal test of movement, selectivity or learning. No fresh confirmation occurred.
