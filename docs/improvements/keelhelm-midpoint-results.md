# Where KeelHelm lost bullet damage to Diamond Port

Over the same 12 games and 60 rounds, Diamond Port earned **4,150** official bullet-damage points to KeelHelm Crossfire’s **3,434**. The gap arose on shots fired while at least two opponents remained. Diamond fired more often in that phase, used slightly more power and hit a bot with a larger fraction of its shots. The recorded Crossfire issued zero alternate requests. Its measured source and JAR identify the bot used in this comparison.

| Shooter and opponents remaining at fire decision | Shots | Shots per 100 live turns | Mean power | Shots that hit | Raw damage | Credited damage | Credit per shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Crossfire, two or more | 4,042 | 6.76 | 1.078 | 663 (16.40%) | 3,160.42 | 3,071.91 | 0.760 |
| Diamond, two or more | 4,148 | 7.15 | 1.149 | 770 (18.56%) | 3,953.02 | 3,819.58 | 0.921 |
| Crossfire, one | 4,319 | 7.05 | 0.197 | 405 (9.38%) | 385.86 | 360.72 | 0.084 |
| Diamond, one | 3,402 | 7.87 | 0.247 | 363 (10.67%) | 339.87 | 331.97 | 0.098 |

The two common victims account for most of Diamond’s 747.67-point melee credit lead. On Keel, Diamond earned 1,880.39 and Crossfire 1,548.92. On Prism the corresponding figures were 886.51 and 510.24. Against each other Diamond earned 1,052.69 and Crossfire 1,012.75. These totals count actual victims. Diamond’s intended target is unknown.

Of Crossfire’s 663 melee hits, 620 struck the reconstructed intended target and 43 hit another bot. Neither shooter had a multiple-victim bullet on this prefix. For Crossfire, 2,919 melee bullets hit a wall, 460 collided with another bullet and one had no recorded terminal event before round end. A bullet can both collide with another bullet and hit a bot on the same tick. Diamond had 2,964 wall events, 417 colliding bullets and one unresolved melee bullet.

Crossfire fired 3,248 shots at scan ranges of at least 500. Of these, 484 hit the intended opponent and 22 hit another. The other 794 shots yielded 136 intended and 21 collateral hits. This describes an opportunity to examine long-range aim. Range, opponent mix, power and movement all varied together.

| Game | Official Crossfire damage | Official Diamond damage |
| --- | ---: | ---: |
| 1 | 349 | 332 |
| 2 | 356 | 412 |
| 3 | 172 | 452 |
| 4 | 347 | 339 |
| 5 | 196 | 443 |
| 6 | 312 | 240 |
| 7 | 291 | 414 |
| 8 | 252 | 412 |
| 9 | 325 | 319 |
| 10 | 265 | 216 |
| 11 | 170 | 355 |
| 12 | 399 | 216 |

Raw damage uses the event's nominal bullet damage. Credited damage is capped at the victim's nonnegative remaining energy immediately before that hit. The game rounds cumulative damage for display. The reader verified all four owners at each of the 60 round ends, reproduced all 240 displayed integers, and checked every final standing against the published result. Kill bonuses are a separate score category. The largest absolute rounding residual was 0.494 points. [Per-game phase, victim, range and round attribution](../../benchmarks/keelhelm-midpoint/attribution.json)

## What happened when the long-range aim changed

**Reject the midpoint correction.** The pinned engine checks contact along the segment from a bullet’s newly advanced position to its next position. The proposed rule aimed for the midpoint of that segment at long range, while keeping the same physical firing opportunities, target, power and turn-rate limit. On the 4,042 matched slots it gained five intended contacts and lost 15. Intended hits fell from 620 to 610. Shots hitting any bot fell from 663 to 654; projected collateral contacts rose from 43 to 44.

| Intended target | Original intended hits | Projected intended hits | Gained | Lost | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diamond Port | 210 | 202 | 1 | 9 | −8 |
| Keel | 306 | 305 | 3 | 4 | −1 |
| Prism | 104 | 103 | 1 | 2 | −1 |
| **All opponents** | **620** | **610** | **5** | **15** | **−10** |

| Game | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Net intended hits | −2 | −1 | −2 | +1 | −1 | +1 | +1 | −4 | −2 | +1 | 0 | −2 |

The registered gate required at least 20 net additional intended hits, gains in at least seven games, a nonnegative net for each intended opponent and at least as many shots contacting any bot. Four of 12 games improved. All four requirements failed. No new candidate or live Diamond duel or melee comparison qualified.

The correction changed the firing direction on 999 eligible shots, and 70 original slots no longer met the alignment gate. On 38 other shots the changed path survived a collision that removed the original bullet; subsequent travel was censored because the other bullet’s counterfactual path was unavailable. One shot reached the recorded round end unresolved. Neither kind of unknown earned projected contact credit. Suppressed opportunities and unknowns stayed in the denominator. [Paired outcomes by game and intended opponent](../../benchmarks/keelhelm-midpoint/shadow.json)

## Validation and scope

Before evaluating the modified ray, the reader reproduced the observed first terminal tick and the full sets of wall, bullet and bot contacts for every baseline melee shot, or followed the original bullet to the observed end of round. It checked the 4,042 original firing decisions against the physical shots. The model uses the pinned server's intersection functions, radius 18, wall removal before contact tests and all same-tick bullet and bot contacts. Partial survival after an original collision is censored. The shooter’s current state, own radar scans and public deaths determine aim. Recorded future positions label contacts. Each changed shot is evaluated separately on the original bullet and opponent trajectories. A real opponent could respond differently. Damage, energy, survival and competitive score for this change remain unmeasured.

Physical budget was zero new games and zero new rounds. The 12 original candidate recordings, each five rounds, were the sole source. KeelHelm Crossfire 0.1.0, Diamond Port 1.8.28, Prism 0.2.0 and Keel 0.1.0 were in every game. Exact input and environment pins are in the [original preflight](https://github.com/stephanelee/robocode/tree/8b580d9fb5c9c5b2ddded43f785ee1ed1c615508/benchmarks/keelhelm-crossfire/minimind-preflight). The measured JAR SHA-256 is `fe6498d461ebc6c92e5d2d03e047c91771166c3d339e49ed647cf47966164d96`. The server SHA-256 is `16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c`. The exact reviewed execution head is `4eca5f5b0b9bb4c932a007e9189a3ecc9c11c296`. [Registered study](keelhelm-midpoint-protocol.md) · [Pre-score clearance](https://github.com/stephanelee/robocode/issues/143#issuecomment-6990052408) · [Baseline contact calibration](../../benchmarks/keelhelm-midpoint/baseline.json) · [Recording hashes](../../benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json) · [Original private archive](../../benchmarks/keelhelm-crossfire/stopped-attempt/artifact-proof.json)

An independent reviewer reproduced the descriptive attribution and the baseline and proposed-ray outputs. The latter matched byte-for-byte with SHA-256 `713d43b67a05b1c0585000419d93df54fffefa258c0c16237859f8cb6aa63cce`. They verified all four failed gates from the per-game and per-target counts. The private diagnostic bundle `melee-damage-midpoint-143.zip` is 506,796 bytes, SHA-256 `c54f28991d86ae9c668d7598ad9b298aeff353e01ba08400cabd2880c7f72883`. It retains the reconstructed intent contexts, compiled reader, source, provenance index and run outputs next to the original sealed recording archive.

## Reproduce the offline results

From the project root, set EVIDENCE_DIR to the extracted original evidence directory with games and server.jar. The intent exporter ran on minimind with OpenJDK 17.0.21.0.3, the attribution script with pinned CPython 3.12.16, and the engine geometry reader with pinned Temurin 25.0.5+8. Set JDK17, JDK25 and PYTHON to the corresponding home directories and Python executable.

```sh
classes=$(mktemp -d)
jar=experiments/KeelHelmCrossfire/KeelHelmCrossfire.jar
cp="$classes:$jar:$EVIDENCE_DIR/server.jar"
"$JDK17/bin/javac" -cp "$jar" -d "$classes" tournament/tools/CrossfireReplay.java tournament/tools/MeleeShotIntents.java
"$JDK17/bin/java" -Xmx256m -cp "$cp" MeleeShotIntents "$EVIDENCE_DIR" > "$classes/intents.json"
"$PYTHON" tournament/tools/melee_damage.py "$EVIDENCE_DIR" "$classes/intents.json" > "$classes/attribution.json"
"$JDK25/bin/javac" -cp "$cp" -d "$classes" tournament/tools/MidpointReplay.java
for mode in baseline shadow; do
  "$JDK25/bin/java" -Xmx256m -cp "$cp" MidpointReplay "$EVIDENCE_DIR" "$mode" > "$classes/$mode.json"
done
for file in attribution baseline shadow; do
  cmp "$classes/$file.json" "benchmarks/keelhelm-midpoint/$file.json"
done
```
