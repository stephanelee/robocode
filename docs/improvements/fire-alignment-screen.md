# Distance-aware current-heading fire alignment: prospective screen

Baseline source `204ad4c9` (accepted first pilot rejection), runnable JAR SHA256 `1b99fa0ac303e5e5f89ed91bb3803841ded54910628f335fe1aeaeba65e34c9a`. This candidate changes only the fire-alignment gate from strict absolute angular error<10degrees to error<min(10,degrees(atan2(18,predictedRange))). The range is the distance from current shooter coordinates to the existing predicted target point. The error remains the existing normalized aim-minus-CURRENT-gun-heading value, including current GF offset. Target selection, projected point, requested gun turn, firepower, target-wave lifecycle, movement, DQN and recorder are unchanged. No learned model is deployed.

Pinned1.4.0 engine [fires before executing new intents](https://github.com/robocode-dev/tank-royale/blob/v1.4.0/server/src/main/kotlin/dev/robocode/tankroyale/server/core/TurnProcessor.kt#L65), so the gate must use current gun heading. At500pixels a9degree heading error misses a point by78.22pixels while radius is18. At100pixels9degrees misses by15.64 and the old10degree cap stays. This is a conservative geometric alignment repair hypothesis; prediction error, GF offset and discrete timing still make the point uncertain. Withholding misaligned shots may also lower useful shot volume and worsen score.

Deterministic fixtures before matches: distant9degree error blocked; distant2degree error allowed; near9degree error remains allowed; error exactly at the computed tolerance is blocked; sign symmetry; zero range retains10degree cap; invalid/nonfinite inputs fail closed. Actual aim-handler fixtures verify that gun commands/power requests remain unchanged for allowed shots and blocked requests leave prior pending-wave state intact. Historical accepted recordings remain immutable and are replayed with only the historical gate substituted explicitly; separate candidate fixtures exercise the real new gate.

## Proposed fast screen, before outcomes

Independent review must clear this plan before any performance matches. Three independent Retcon games per arm, five rounds each, fixed minimum-risk action0, absent models, seeds2026110401/2026110402/2026110403. Seeds label independent unpaired random starts, not matched arenas or fixed-mode bot randomness. Within blocks order previous/candidate, candidate/previous, previous/candidate. Full immutable bot bundles are snapshotted before launch. Retcon remains pinned `28981c7a469ac064194b41001cd7183ec5ecf189`, engine1.4.0; build Java17, runtime Java25, exact ubuntu-26.04. Run with existing comparator purpose=diagnostic, rounds=5; its ordinary complete20-round tactical gate is intentionally not a screen-retention decision.

Then one independent three-round game per arm against each official1.4.0 Target, SpinBot and Walls (previous/candidate for Target, candidate/previous for SpinBot, previous/candidate for Walls), fixed minimum-risk and isolated absent models, seed2026110404. Total48scored rounds:30Retcon+18sample. No training. No performance early stopping, favorable replacements or hidden reruns. Invalid acquisition retains all partial artifacts and stops for review.

Proposed screen advance criteria: mean of three Retcon game score-share deltas>=0; at least2/3 nonnegative blocks; worst block>=-10percentage points; zero AIS skipped turns; aggregate shot hit rate OR damage per shot strictly improves; candidate shots per1000 own turns>=50% baseline. Samples each lose no more than5score-share points and have zero AIS skipped turns. All ratios use aggregate counts, not averages of round rates. Report official score shares, wins, shots, turns, hit rate, damage/shot, wall rate and skipped turns for both arms and each game. Wall rate is descriptive at this short screen; record rather than reinterpret it afterwards.

This is a limited-power screening decision, not a strength or retention claim. Failure rejects/reverts the gameplay source/JAR while preserving patch, fixtures and raw evidence. Success only advances to separately preregistered120fixed-policy rounds under the existing+5point tactical gate and mandatory120default-online-training veto; that stronger confirmation must precede gameplay merge. Passing this small screen does not close the strategy tracker or establish beating Retcon.

## Measured result: reject

Independent review cleared the exact source/workflow before the single[48-round run41525955712](https://github.com/stephanelee/robocode/actions/runs/41525955712). All12games and48rounds completed; actualUbuntu26.04 provenance, Java17 source/JAR parity and Java25 runtime were verified. No rescue rerun or confirmation followed. Canonical summaries and raw data are[retained with hashes](../../benchmarks/fire-alignment-screen/README.md). [PR30](https://github.com/stephanelee/robocode/pull/30) records the candidate and restoration.

| Retcon metric,15rounds per arm | Accepted | Candidate |
| --- | ---: | ---: |
| Mean game score share | 24.9990% | 17.3112% |
| Wins | 2 | 1 |
| Shots | 547 | 540 |
| Hits / shots | 11.7002% | 7.2222% |
| Bullet damage / shot | 0.808131 | 0.578223 |
| Wall collisions /1000turns | 7.9497 | 0.5672 |
| AIS skipped turns | 0 | 0 |

Mean score-share delta−7.687809pp fails the nonnegative rule. Seed-block deltas are−33.674033,+10.266221,+0.344386pp: two blocks are nonnegative but the worst violates−10pp. Shot volume per1000turns remains86.54%of accepted and passes the50%floor; neither hit rate nor damage/shot improves. The sample score-share losses are0pp(Target),3.088803pp(SpinBot),10.321489pp(Walls), so Walls fails the5ppguard. Zero-shot rates were prospectively treated as inconclusive; both arms here fired hundreds of shots and the rejection is defined.

The narrower gate therefore **fails this bounded screen**. The geometric observation remains valid, but it does not imply a useful combat change under uncertain prediction and unpaired starts. Lower wall counts do not override failed combat gates. Source, tests and JAR are restored exactly to204; rejected patch and actual outcomes remain preserved. No240-round confirmation or gameplay promotion is authorized by this result.
