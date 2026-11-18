# Cooldown retargeting cost 11 shot opportunities

**Reject this policy under the registered offline gate.** Across the 12 retained candidate games the simulated gun created 228 shot opportunities and displaced 239. No game had a positive net count. The control fired 4,042 physical shots. On those same melee trajectories the proposed rule produced 4,031 eligible shot proxies. This spent zero new games and rounds. The conditional live Diamond Port duel and four-bot melee stage did not qualify.

| Retained game | Created | Displaced | Net |
| --- | ---: | ---: | ---: |
| 1 | 63 | 67 | −4 |
| 2 | 31 | 32 | −1 |
| 3 | 14 | 15 | −1 |
| 4 | 5 | 5 | 0 |
| 5 | 4 | 5 | −1 |
| 6 | 14 | 15 | −1 |
| 7 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 |
| 9 | 15 | 15 | 0 |
| 10 | 11 | 11 | 0 |
| 11 | 51 | 53 | −2 |
| 12 | 20 | 21 | −1 |
| **Total** | **228** | **239** | **−11** |

## Where the shots went

Of 59,825 matched decision turns, both policies fired at the same intended opponent on 3,799 and at different opponents on four. Neither fired on 55,555. Every decision had an observable following tick. The gun committed to an alternate 69 times and cancelled 35 commitments before the first cold decision. Changing one shot can shift several later shots in time through heat. A shifted shot appears once on each side of the comparison.

| Intended opponent | Shadow only | Baseline only | Shared, same target |
| --- | ---: | ---: | ---: |
| Diamond Port | 75 | 81 | 1,029 |
| Keel | 128 | 135 | 1,975 |
| Prism | 25 | 23 | 795 |

Three shared shots redirected from Keel to Prism and one from Prism to Diamond Port. These labels describe the intended target when firing. The hot gun protected its incumbent on 33,632 decisions. For the remaining hot decisions that searched for an alternate, 19,045 opponent checks failed on age, 13,355 on range and two on turn budget. No opponent failed the energy check after passing age and range. These are opponent checks, so multiple failures may belong to the same decision.

## Calibration and interpretation

The adapter checked all 12 candidate recording and telemetry hashes and the measured candidate JAR against the published independent audit. Across all 60 original rounds, the ordinary selector reproduced every physical melee shot by turn, owner, power and ray. The unmodified simulated gun also passed every live melee heat/ray check and matched the selector on shot presence, target and power. Own death ticks were excluded. A shot requested in melee and created on the first duel tick remained eligible. The observed cold gun used the pre-rotation ray. It acquired full shot heat at creation, then cooled on subsequent ticks.

The simulation used the original movement, radar, opponent behavior and environmental energy changes, plus a running adjustment for ammunition expenditure. It gave no additional energy credit for hypothetical hits or net saved power above observed energy. On nonfiring turns, private callback order and intended target are reconstructed. Counterfactual hit rate, damage, survival, 1v1 performance and game score are unmeasured.

[Registered protocol](keelhelm-cooldown-protocol.md) · [Original issue and gate](https://github.com/stephanelee/robocode/issues/141) · [Exact pre-run clearance](https://github.com/stephanelee/robocode/issues/141#issuecomment-6984780707) · [Per-game evidence](../../benchmarks/keelhelm-cooldown-shadow/games.json) · [Input hashes](../../benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json) · [Raw recording custody](../../benchmarks/keelhelm-crossfire/stopped-attempt/artifact-proof.json) · [PR #142](https://github.com/stephanelee/robocode/pull/142)

A separate reviewer compiled the cleared head and reproduced the scored output byte-for-byte. They verified per-game shot totals, next-tick coverage and conservation across all five categories. The per-game JSON SHA-256 is `85c225241279c6cbd442e5b9bf4463bc975ab8b477d0766ceb3a7145ec5ef616`. The private diagnostic archive `cooldown-shadow-141-offline.zip` is 50,151 bytes with SHA-256 `4e44d080230a5ceeef1797a5dac9a208b2154bcdd202fa86829444866a9f2c7e`.

## Reproduction

The execution head was `052bd5ac78bd97f2db379c8f0e3dc88c7bedfa5b` with Homebrew OpenJDK 17.0.21.0.3 on minimind. The input JAR SHA-256 was `fe6498d461ebc6c92e5d2d03e047c91771166c3d339e49ed647cf47966164d96`. The Tank Royale 1.4.0 server SHA-256 was `16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c`. The original field comprised Keel, Prism, Diamond Port and KeelHelm Crossfire; exact opponent and runtime pins remain in the [original preflight](https://github.com/stephanelee/robocode/tree/8b580d9fb5c9c5b2ddded43f785ee1ed1c615508/benchmarks/keelhelm-crossfire/minimind-preflight).

From the repository root, set EVIDENCE_DIR to the extracted original evidence directory containing games.

```sh
jar=experiments/KeelHelmCrossfire/KeelHelmCrossfire.jar
classes=$(mktemp -d)
javac -cp "$jar" -d "$classes" tournament/tools/CrossfireReplay.java tournament/tools/CooldownShadow.java tournament/tools/CooldownShadowFixtures.java
java -cp "$classes:$jar" CooldownShadowFixtures
java -Xmx256m -cp "$classes:$jar" CooldownShadow "$EVIDENCE_DIR" > "$classes/games.json"
cmp "$classes/games.json" benchmarks/keelhelm-cooldown-shadow/games.json
```
