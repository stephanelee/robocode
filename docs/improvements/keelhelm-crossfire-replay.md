# Why Crossfire never requested an alternate shot

During the 12 recorded Crossfire games, a fresh alternate was within range on only 11 turns when the gun was cool and the incumbent could not fire. On all 11 turns the alternate intercept was at least 45.70° away from the gun. Crossfire keeps turning the gun toward the incumbent, so it could not use any of these opportunities.

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

## Evidence and limits

This retrospective diagnosis spent zero new games and zero new rounds. It covers all 12 candidate recordings and all 60 candidate rounds from the stopped attempt. Its decision is to retire this passive alternate-shot hypothesis for this recorded prefix. It provides no new melee or duel strength measurement. [Original comparison and private archive proof](keelhelm-crossfire-results.md)

The reader checks the measured candidate JAR and every input recording and telemetry hash against the published independent audit. It uses the measured JAR's target selection, intercept, power, and shot choice directly. Inputs to each reconstruction are the bot’s own state, its own recorded scans, and public death events through the current turn. An observer tick containing the bot's death cannot start another bot decision. The reader compares expected shots to physical fire on the next turn using owner, turn, power and gun direction. Every round reproduces the private melee-turn, accepted-shot and alternate-request counts. All 4,042 expected shots match, with zero missing or extra shots. Observer events do not expose private callback order or per-turn incumbent identity. Agreement on the logged totals and physical shots validates the reconstructed firing behavior for this prefix.

[Per-round counts and eligible alternate angles](../../benchmarks/keelhelm-crossfire/replay-diagnostic/rounds.json) · [Reader](../../tournament/tools/CrossfireReplay.java) · [Input hashes and provenance](../../benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json)

## Reproduce on the retained evidence

From this repository root, use Java 17 and the extracted original private archive’s evidence directory containing the games folder.

```sh
jar=experiments/KeelHelmCrossfire/KeelHelmCrossfire.jar
classes=$(mktemp -d)
javac -cp "$jar" -d "$classes" tournament/tools/CrossfireReplay.java
java -Xmx256m -cp "$classes:$jar" CrossfireReplay "$EVIDENCE_DIR" > "$classes/rounds.json"
cmp "$classes/rounds.json" benchmarks/keelhelm-crossfire/replay-diagnostic/rounds.json
```
