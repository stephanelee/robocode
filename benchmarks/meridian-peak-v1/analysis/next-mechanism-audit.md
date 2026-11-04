# Existing Peak screen: exploratory mechanism audit and next hypothesis

2026-11-04. **Read-only, post-outcome exploration of all five games/all 25 rounds.** No new game, replayed controller, alternate rollout, matrix, probe, fit, parameter search, source/JAR/tool/protocol/freeze change, or strength claim. These games are spent development evidence. The registered screen failed; no conditional confirmation is justified. Root is separately archiving canonical results and independent outcome review; this note does not replace that review.

## Evidence and reduction boundary

Sole acquisition: [GitHub run 41783652109](https://github.com/stephanelee/robocode/actions/runs/41783652109), local original `build/screen-41783652109/build/meridian-peak/screen/acquisition/`. Read every `raw/*/*.battle.gz`, all candidate `state/round-001..005.json` and endpoint/cold DTOs, marker streams and label journals. Official observer events supply shots, hits, uncapped damage, positions, score endpoints and contacts. Legal private DTOs supply model/support and planner counters. No opponent-private data was used to evaluate an alternative policy.

Counts use the semantics of [the frozen reducer](../../../tools/meridian_peak_common.py): BulletFiredEvent owner identifies shots; BulletHitBotEvent owner/victim identifies damage and hits; per-round scores/first places are differences of cumulative round endpoints. A game's share is its total own score divided by both total scores; it is **not** the mean of its round shares. Speed summaries use ticks containing own state; ranges require both bots; signed velocity reversals count sign changes between successive nonzero own velocities, ignoring intervening zeroes. They do **not** identify orbit-side switches. Fresh-scan fraction means an own ScannedBotEvent is at most three ticks old, inclusive of the current observer tick; callback ordering can differ from decision-time freshness. Longest alive no-fire run is consecutive ticks with own bot state present and no confirmed own shot, including warmup/heat/energy/alignment/terminal periods: it is **not** a common-opportunity liveness diagnostic. Mean power is weighted by confirmed shots, not rounds.

The original five official recording SHA256 values are:

| Game | Compressed recording SHA256 |
|---|---|
| mp1-01-control-diamond | `3961dcb1257685fa9b2414714d2e65dbf424c04aa783462d50b157d66bc1519b` |
| mp1-02-peak-diamond | `74f4e42119f7c329d49110f9e0f25c2378d141e9993ddde3a1367448190b8462` |
| mp1-03-peak-retcon | `f213787d9a6a0b588af8d8c22f675dca67c0fedc0c899ee1f6d67b868b713f25` |
| mp1-04-peak-ais | `90d7caa4a4f85aff10c9ab51c4d773fa04fe0493b5da1c0ce077e162056a6533` |
| mp1-05-peak-walls | `dc6667a6f474a87bb0f61707497ff6cdae71084a0d69ef86e701105a7114be2b` |

## All five games

| Game | Share | Wins /5 | Own hits/shots | Incoming hits/shots | Uncapped dealt/taken | Incoming damage /100 shots | Walls /bot contacts |
|---|---:|---:|---:|---:|---:|---:|---:|
| mp1-01-control-diamond | 12.398% | 0 | 17/205 | 52/294 | 91.467/290.837 | 98.924 | 0/0 |
| mp1-02-peak-diamond | 25.505% | 1 | 21/197 | 38/232 | 134.467/298.660 | 128.733 | 2/0 |
| mp1-03-peak-retcon | 93.128% | 5 | 25/236 | 5/242 | 188.327/38.000 | 15.702 | 1/0 |
| mp1-04-peak-ais | 93.631% | 5 | 45/190 | 7/170 | 337.400/47.200 | 27.765 | 0/0 |
| mp1-05-peak-walls | 100.000% | 5 | 50/95 | 0/25 | 413.467/0.000 | 0.000 | 0/0 |

All 25 rounds completed with zero own skipped turns. The transfers passed the absolute guards in this one screen; they are not evidence of superiority or an excuse to waive Diamond guards. Diamond candidate share 25.505% fails the strict >35% guard; incoming damage/100 enemy shots 128.733 exceeds control 98.924. The one candidate Diamond win does not establish a winning bot.

## Every round, including the outlier and late loss

`H/S` denotes own confirmed hits/shots; `in H/S` enemy hits/shots; power columns are mean own/enemy confirmed-shot power. Ranges are mean observed centre distances. No row is excluded.

| Game | R | Turns | Share | Win | H/S | In H/S | Dealt/taken | Own/enemy power | Range | Velocity reversals | Longest alive no-fire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mp1-01-control-diamond | 1 | 709 | 5.926% | 0 | 1/33 | 6/46 | 7.900/58.200 | 1.497/1.945 | 458.9 | 40 | 61 |
| mp1-01-control-diamond | 2 | 863 | 16.875% | 0 | 5/49 | 9/58 | 27.545/61.430 | 1.157/1.509 | 400.2 | 57 | 77 |
| mp1-01-control-diamond | 3 | 1407 | 29.496% | 0 | 9/76 | 19/107 | 40.222/32.013 | 1.238/0.645 | 431.1 | 76 | 124 |
| mp1-01-control-diamond | 4 | 645 | 9.494% | 0 | 2/28 | 9/43 | 15.800/70.144 | 1.460/1.561 | 456.7 | 41 | 89 |
| mp1-01-control-diamond | 5 | 607 | 0.000% | 0 | 0/19 | 9/40 | 0.000/69.050 | 1.650/1.593 | 419.7 | 22 | 187 |
| mp1-02-peak-diamond | 1 | 656 | 17.514% | 0 | 4/37 | 8/42 | 31.050/72.616 | 1.294/1.823 | 425.4 | 52 | 33 |
| mp1-02-peak-diamond | 2 | 1016 | 19.079% | 0 | 6/59 | 11/72 | 29.253/52.525 | 1.134/1.215 | 428.1 | 55 | 57 |
| mp1-02-peak-diamond | 3 | 788 | 79.699% | 1 | 5/44 | 3/53 | 38.174/27.280 | 1.639/1.484 | 502.3 | 40 | 33 |
| mp1-02-peak-diamond | 4 | 562 | 17.801% | 0 | 5/32 | 9/36 | 34.117/82.315 | 1.260/1.813 | 452.3 | 32 | 39 |
| mp1-02-peak-diamond | 5 | 459 | 1.439% | 0 | 1/25 | 7/29 | 1.873/63.924 | 1.504/1.858 | 388.8 | 18 | 48 |
| mp1-03-peak-retcon | 1 | 673 | 100.000% | 1 | 5/39 | 0/39 | 39.500/0.000 | 1.679/1.554 | 350.4 | 45 | 30 |
| mp1-03-peak-retcon | 2 | 784 | 87.097% | 1 | 6/46 | 2/51 | 40.927/16.400 | 1.445/1.384 | 410.4 | 53 | 31 |
| mp1-03-peak-retcon | 3 | 654 | 100.000% | 1 | 4/38 | 0/44 | 31.600/0.000 | 1.649/1.557 | 359.2 | 49 | 47 |
| mp1-03-peak-retcon | 4 | 816 | 86.441% | 1 | 5/42 | 2/57 | 36.800/15.200 | 1.613/1.298 | 383.1 | 44 | 39 |
| mp1-03-peak-retcon | 5 | 1151 | 94.340% | 1 | 5/71 | 1/51 | 39.500/6.400 | 1.111/1.266 | 369.8 | 70 | 32 |
| mp1-04-peak-ais | 1 | 864 | 85.000% | 1 | 9/51 | 4/44 | 64.767/24.400 | 1.426/1.197 | 410.7 | 94 | 31 |
| mp1-04-peak-ais | 2 | 511 | 95.973% | 1 | 9/29 | 1/30 | 69.600/5.200 | 1.624/1.160 | 446.4 | 38 | 33 |
| mp1-04-peak-ais | 3 | 490 | 89.571% | 1 | 10/30 | 2/25 | 79.000/17.600 | 1.609/1.574 | 383.2 | 48 | 30 |
| mp1-04-peak-ais | 4 | 643 | 100.000% | 1 | 9/38 | 0/34 | 64.800/0.000 | 1.495/1.071 | 424.6 | 62 | 35 |
| mp1-04-peak-ais | 5 | 738 | 100.000% | 1 | 8/42 | 0/37 | 59.233/0.000 | 1.574/1.124 | 433.6 | 53 | 33 |
| mp1-05-peak-walls | 1 | 270 | 100.000% | 1 | 9/16 | 0/5 | 81.000/0.000 | 1.693/2.000 | 369.9 | 19 | 30 |
| mp1-05-peak-walls | 2 | 448 | 100.000% | 1 | 11/29 | 0/5 | 79.333/0.000 | 1.270/2.000 | 453.7 | 50 | 30 |
| mp1-05-peak-walls | 3 | 377 | 100.000% | 1 | 9/21 | 0/6 | 80.333/0.000 | 1.579/2.000 | 345.0 | 20 | 39 |
| mp1-05-peak-walls | 4 | 260 | 100.000% | 1 | 11/14 | 0/5 | 86.900/0.000 | 1.633/2.000 | 391.4 | 15 | 31 |
| mp1-05-peak-walls | 5 | 285 | 100.000% | 1 | 10/15 | 0/4 | 85.900/0.000 | 1.727/2.000 | 371.1 | 10 | 30 |

## What the failure does and does not show

**The damage guard fails without a higher observed Diamond hit frequency.** Control incoming hit rate is 52/294 =17.687%; Peak is 38/232 =16.379%. Mean uncapped damage per received hit rises from 5.593 to 7.859. The identity `100 × hits/enemyShots × damage/hit` gives 98.924 versus128.733. Control round3 is a material composition difference:107 enemy shots/19 hits, mean incoming-hit power0.397,32.013 damage. Peak round3 has53 enemy shots/3 hits, mean incoming-hit power1.849,27.280 damage. That candidate round supplies the sole win/79.699% share, followed by17.801% and1.439% shares. Different random starts, opponent learning, shot power, exposure duration and the changed full movement path prevent causal attribution. Do not say the Peak objective raised Diamond's hit probability, or normalize away the failed registered metric. Do not use this decomposition as a post-hoc replacement gate.

Aggregate Diamond own firing remains low-accuracy:17/205=8.293% control versus21/197=10.660% candidate; own mean power1.329 versus1.344. Candidate dealt134.467 damage while spending264.793 shot energy (control91.467/272.423). Shooter hit bonuses, opponent energy/power adaptation and score caps mean those ratios alone do not predict wins. The unchanged gun has not solved Diamond targeting. Candidate round5 has only1/25 hits and1.873 damage despite substantial learned support. A different movement interaction can change targeting even with source-identical gun arithmetic.

**There is no observed data-starvation or long-stall rescue here.** Peak Diamond markers contain166 learned/31 fallback requests;191 accepted outgoing crossing labels reach a191/192 example bank. Its five incoming accepted counts are8,10,3,9,7 (37 total of38 hits; one UNMATCHED). Effective decayed histogram mass grows6.732,12.056,13.189,15.707,17.002. All candidate crossing exclusion counters and missing-scan trial counters are zero; Diamond has832 scored virtual trials. Pending shots at each Diamond seal are1,0,2,1,2 and pending trials5,6,7,5,5; these are unresolved, not failures or misses. A label that matches the legal inference does not establish the opponent's true aiming context. A supported learned gun is not necessarily an accurate one.

The candidate Diamond longest alive no-fire runs are33,57,33,39,48 versus61,77,124,89,187 control. Firing rates are56.593 versus48.452 shots/1000 observer turns; these are descriptive, not a rerun of PR92's stronger common-opportunity test. Fresh observer scans cover97.343–99.746% of candidate Diamond own-state ticks. No basis is shown for repeating held-plan liveness work or treating mask advancement as the primary next experiment.

**Neither zero speed nor absent incoming learning explains the transfer pattern.** Candidate mean absolute observed speed is5.720 Diamond,5.172 Retcon,4.983 AIS,5.617 Walls. Velocity reversal counts are197/261/295/114 respectively; actual zero speed occupies less than1% of every round. Control Diamond has236 reversals, mean absolute speed5.636. Signed reversals can arise from back-as-front orbit control and cannot recover selected side, switching margins or an action-tree counterfactual. Candidate Diamond mean range443.1 versus control431.7; transfers375.4/420.1/390.7. These are consequences of whole interactions, not independently assigned range treatments.

Retcon/AIS/Walls final accepted-hit totals are5/7/0, effective histogram masses4.524/6.033/0, outgoing accepted totals230/182/87 and bank counts192/182/87. Their learned/fallback request totals are217/19,174/16,57/38. Thus Walls' perfect screen uses the cold incoming prior throughout; it cannot validate the Diamond-trained histogram. None of these five-round processes share learning.

## Concrete next hypothesis: allow one planned continuation change between waves

**Proposed research direction, not a registered candidate:** the current planner may reject a useful immediate orbit because its50-tick scoring requires the same orbit sign for the entire forecast, including later waves. A bounded two-leg plan that may change orbit after the first relevant wave clears could represent useful escapes unavailable to either held-sign forecast. Keep this separate from selecting a smaller numeric kernel, prior weight, range, horizon or switch penalty after viewing this result.

Source basis: [PeakRisk.evaluate](../../../experiments/MeridianPeak/src/PeakRisk.java#L22) fixes `sign` throughout its50-step orbit loop and sums all waves' per-ray maxima. [performTurn](../../../experiments/MeridianPeak/src/MeridianPeak.java#L179) instead recalculates two choices on every active fresh turn. Frozen private counters show3–4 active inferred waves in Diamond rounds, and3 in every transfer game at some point. The deployed controller therefore replans among multiple waves while each value estimate assumes one sign throughout. This is a concrete representational limitation, **not** evidence that any recorded loss was caused by it. The current archive has no per-decision selected wave identities, alternate scores, peak times, chosen signs or predicted routes. It cannot establish which wave dominated a decision, whether the actual controller switched signs, or that an alternate continuation would have survived. No alternate paths were reconstructed here.

A future prospectively reviewed experiment could compare exactly one bounded continuation design against unchanged Peak, retaining its gun/radar/power, wave inference, kernel, ray mixtures, finite sweeps and lifecycle. Before any implementation/game, specify the legal first-wave identity/order, per-path wave-clear boundary and handling of unresolved/tied waves, when switching is permitted, tie/switch accounting, fixed path/CPU budget and complete own-state trace joins. Contract tests would need to distinguish a real two-leg path from simply terminating a horizon or changing a mask, and establish that the first issued command and next-state forecast match the public engine. Fresh bounded games would still need the original Diamond damage/share and transfer safeguards plus independent review; these spent25 rounds cannot confirm it. This report chooses no boundary constants, freezes no protocol and authorizes no game.

### Prior work and rejected shortcuts

- [PR92 held gun](../../../docs/improvements/meridian-gun-plan-results.md) reduced an observed stall but lost competitive share; changing the gun to chase this screen's liveness is not the proposed mechanism.
- [Peak's prior-art registration](../../../docs/improvements/meridian-peak-v1.md#L48) already identifies Breakwater's failed per-wave mixture peak. Repeating MAX/mixture aggregation or claiming peak danger is novel would duplicate spent work.
- [Challenger v4](../../../docs/improvements/diamond-wave-horizon-v4.md) corrected irrelevant late-wall vetoes after waves finish and still failed its20-round screen. The proposal is a conditional continuation while later waves remain, rather than another horizon termination or STOP option. Challenger's [source](../../../experiments/DiamondChallenger/src/WaveSurfMove.java#L34) scores held CW/CCW/STOP policies, not a two-leg action tree. Its [trace analysis](../../../docs/improvements/challenger-defense-trace-analysis.md) already found selected hit-wave coverage and lowest-stored-score dispatch, so missing-wave/minimum-choice claims require new actual evidence.
- The [spent PR94 mask-only work](../../../docs/improvements/meridian-peak-v1.md#L1) is not fresh confirmation or a reason to rescan a diagnostic matrix. No mask change is proposed.
- Merely adding shot-damage weights is not a new discovery: Challenger's score already multiplies damage/time. The present power decomposition warns about exposure composition; it does not identify damage weighting as the cause or justify tuning it.
- [Aegis](../../../docs/improvements/aegis-shield-v1.md) already tested an interception/replay strategy and failed despite high blocking. This audit does not repackage that whole-bot design or claim ordinary shots provide safe bullet shadows.

## Runtime, liveness and remaining uncertainty

Candidate Diamond recorded3437 planner decisions/3,493,896 kernel evaluations, mean153.409 microseconds and max3.125ms per instrumented decision. Retcon/AIS/Walls means178.002/116.534/123.069 microseconds, maxima9.852/3.567/4.384ms. These are adapter measurements including its recorded scope, not full process scheduling or a guarantee of headroom; zero skips occurred in this screen. A larger action set must independently satisfy the pinned100ms limit, source/JAR parity and retained failure rules. The sparse cumulative DTOs cannot measure hypothetical branch cost.

There is one independent engine-started game for each screen slot; rounds learn within that process and are correlated. Apparent late-learning changes, one large winning round, transfer wins, lower observed hit frequency, support activation and zero skips cannot establish general strength, calibrated danger or the cause of failure. Keep all failed outcomes and exact artifacts. Do not confirm, tune or rerun MeridianPeak0.1 under this failed registration.
