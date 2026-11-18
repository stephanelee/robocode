# Why Crossfire never requested an alternate shot

In the reconstruction of the 12 recorded Crossfire games, a fresh alternate was within range on only 11 turns when the gun was cool and the incumbent could not fire. On all 11 turns the alternate intercept was at least 45.70° away from the gun. Crossfire keeps turning the gun toward the incumbent, so it could not use any of these opportunities.

## What narrowed the opportunities

Each row counts turns once, even if two alternate opponents pass a check. Each row after the first is a subset of the preceding row.

| Condition | Turns |
| --- | ---: |
| Bot alive and playing melee | 59,825 |
| Gun cool, with an incumbent target | 5,564 |
| Incumbent could not fire on the current gun direction | 1,522 |
| Another living opponent known | 1,522 |
| At least one alternate scanned in the last 0–3 turns | 1,003 |
| At least one such alternate strictly within 500 units | 11 |
| Energy sufficient for at least one such alternate | 11 |
| Alternate intercept aligned with the current gun direction | 0 |

The incumbent fired on the other 4,042 cool turns. Of the 1,003 turns with a fresh alternate, 992 had none in range. On the remaining 11, the smallest absolute error per turn ranged from 45.70° to 178.46°, with a median of 133.52°. A small relaxation of the alignment tolerance would still produce zero alternate shots on those turns. The next diagnostic worth testing is whether a **deliberate retarget while the gun is cooling** can finish before the next firing opportunity, and how often that would displace a viable incumbent shot. This replay did not simulate those changed trajectories.

## Sensitivity to scan delivery

The original same-turn reconstruction was retrospective. The one-tick delay and its interpretation were registered before its first run. Both use current own state and public deaths. The delayed model applies each own scan after deciding on that turn, retaining its original timestamp. [Issue registration](https://github.com/stephanelee/robocode/issues/138#issuecomment-6984472110)

| Distinct turns or shots | Scans through current tick | Scans through previous tick |
| --- | ---: | ---: |
| Live melee decisions | 59,825 | 59,825 |
| Cool with incumbent | 5,564 | 5,564 |
| Incumbent eligible to fire | 4,042 | 4,518 |
| Incumbent ineligible, another opponent known | 1,522 | 1,046 |
| Fresh alternate | 1,003 | 570 |
| In range | 11 | 8 |
| Affordable | 11 | 8 |
| Aligned | 0 | 0 |
| Predicted shots matched to physical fire | 4,042 | 3,988 |
| Unmatched predictions | 0 | 530 |
| Unmatched physical shots | 0 | 54 |
| Predictions missing the following tick | 0 | 0 |

Scan timing changes the intermediate counts. In the delayed reconstruction the closest eligible alternate was 113.84° off aim. Both reconstructions predict zero alternate requests, in agreement with private round counters. Same-turn scans reproduce every physical melee shot. The delayed reconstruction both invents and misses incumbent shots, so it gives a less faithful account of when the bot fired. Neither timing convention establishes private callback order on an individual nonfiring turn. [Delayed-model counts by round](../../benchmarks/keelhelm-crossfire/replay-diagnostic/previous-tick-scans.json)

## Evidence and limits

This retrospective diagnosis spent zero new games and zero new rounds. It covers all 12 candidate recordings and all 60 candidate rounds from the stopped attempt. Its decision is to retire this passive alternate-shot hypothesis for this recorded prefix. It provides no new melee or duel strength measurement. [Original comparison and private archive proof](keelhelm-crossfire-results.md)

The reader checks the measured candidate JAR and every input recording and telemetry hash against the published independent audit. It uses the measured JAR's target selection, intercept, power, and shot choice directly. Inputs to each reconstruction are the bot’s own state, its own recorded scans, and public death events through the current turn. An observer tick containing the bot's death cannot start another bot decision. The reader compares expected shots to physical fire on the next turn using owner, turn, power and gun direction. Under same-turn scan delivery, every round reproduces the private melee-turn, accepted-shot and alternate-request counts. All 4,042 expected shots match, with zero missing or extra shots. Observer events do not expose private callback order or per-turn incumbent identity. The intermediate counts on turns without a shot are reconstructed estimates. Agreement on the logged totals and physical shots validates the reconstructed firing behavior for this prefix.

[Per-round counts and eligible alternate angles](../../benchmarks/keelhelm-crossfire/replay-diagnostic/rounds.json) · [Reader](../../tournament/tools/CrossfireReplay.java) · [Input hashes and provenance](../../benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json)

An independent reviewer reproduced both modes byte-for-byte from the retained recordings. The delayed model differed from physical fire in all 60 rounds. Both modes hold the observed trajectory fixed. The private diagnostic bundle includes the reader source and classes, runtime version, input hash index and both per-round outputs. Its filename is `crossfire-replay-diagnostic-140.zip`, SHA-256 `4678edcfab07bfb57bc73ab027c6aafd97e51b1705431ca94ab08447368cd55c`. The measured reader run used Homebrew OpenJDK 17.0.21.0.3 on minimind.

## Reproduce on the retained evidence

From this repository root, use Java 17 and the extracted original private archive’s evidence directory containing the games folder.

```sh
jar=experiments/KeelHelmCrossfire/KeelHelmCrossfire.jar
classes=$(mktemp -d)
javac -cp "$jar" -d "$classes" tournament/tools/CrossfireReplay.java
java -Xmx256m -cp "$classes:$jar" CrossfireReplay "$EVIDENCE_DIR" > "$classes/rounds.json"
cmp "$classes/rounds.json" benchmarks/keelhelm-crossfire/replay-diagnostic/rounds.json
java -Xmx256m -cp "$classes:$jar" CrossfireReplay "$EVIDENCE_DIR" --previous-tick-scans > "$classes/delayed.json"
cmp "$classes/delayed.json" benchmarks/keelhelm-crossfire/replay-diagnostic/previous-tick-scans.json
```
