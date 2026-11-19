# KeelHelm full-speed steering results

**Reject the full-speed steering change and retain KeelHelm 0.2.1.** Long-range Diamond contacts rose from 36 to 40. Only the minimum-exposure criterion passed. The registered decision stopped the experiment before live play, for a budget of zero new games or rounds.

The replay covered 290 windows in the 12 original four-bot games, spanning 60 rounds. The control reproduced native position, speed, heading and incoming bullet IDs. The paired paths were assessed for 2301 intervention ticks and 4530 follow-through ticks.

| Incoming fire | Original hits | Steering hits | Original damage | Raw steering damage | Unresolved charge | Latent charge | Truncation charge | Charged damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Diamond Port / 500+ | 36 | 40 | 167.06 | 189.04 | 4.00 | 22.98 | 4.00 | 220.01 |
| Diamond Port / <500 | 8 | 8 | 49.96 | 49.96 | 0.00 | 2.69 | 0.00 | 52.65 |
| Keel | 30 | 31 | 12.00 | 12.40 | 0.80 | 0.80 | 0.00 | 14.00 |
| Prism | 11 | 12 | 64.48 | 73.43 | 2.05 | 4.00 | 0.00 | 79.48 |

Long-range Diamond means the shot originated at least 500 units from KeelHelm’s preceding recorded position. For those bullets, original damage was 68.16 during the eight steering ticks and 98.91 during follow-through. Corresponding steering-path damage was 68.16 and 120.89. There were 0 original pulse hits delayed into follow-through and 8 newly introduced hits. The [per-window trace](../../benchmarks/keelhelm-steering/steer.json) records each relevant Diamond bullet’s ID, launch range, first contact on each path and disposition. IDs are scoped to game and round. Of 32 bullets that struck both paths, 23 hit on the same tick, five arrived earlier and four arrived one or two ticks later. Of four original-only hits, two rays passed, one left the arena and one was unresolved. One of the passes occurred in a wall-truncated window and earned no damage saving for its stratum.

All shooters combined dealt 293.51 nominal damage on the original path and 324.84 on the proposed path. Conservative damage including charges and collisions was 381.15. The conservative endpoint improved in 2 of 12 games. There were 5 candidate wall contacts and 0 possible rams. Of 290 windows, 277 completed, six ended when melee ended, two at death and five at a proposed wall collision. The wall cutoffs removed 13 follow-through ticks from the paired observation, with no original contacts lost. Wall penalties contributed 15.00 damage.

| Original recording | Original damage | Charged steering damage |
| --- | ---: | ---: |
| crossfire-01-2 | 59.96 | 67.95 |
| crossfire-02-2 | 68.39 | 64.39 |
| crossfire-03-2 | 11.19 | 15.19 |
| crossfire-04-1 | 0.00 | 4.00 |
| crossfire-05-1 | 21.20 | 50.58 |
| crossfire-06-1 | 7.99 | 7.40 |
| crossfire-07-2 | 22.38 | 30.37 |
| crossfire-08-2 | 6.80 | 7.20 |
| crossfire-09-1 | 8.79 | 20.19 |
| crossfire-10-2 | 11.99 | 24.68 |
| crossfire-11-1 | 61.77 | 73.16 |
| crossfire-12-1 | 13.04 | 16.04 |

An independent reviewer reproduced the baseline and steering outputs byte-for-byte, checked the timing and disposition of all 49 primary shot records, and independently verified each decision criterion.

## Scope and reproduction

The study used Crossfire 0.1.0 with Diamond Port 1.8.28, Prism 0.2.0 and Keel 0.1.0. Crossfire’s movement code matches accepted KeelHelm 0.2.1. Enemy paths, shots and original follow-through commands stay fixed. The experiment measures incoming damage along hypothetical paths within disjoint windows. A live controller could replan after its position changes. The estimate includes pessimistic damage for newly reachable or unresolved threats. Nominal damage uses bullet power before victim energy caps.

Execution commit `20d190049dba5ef710a9ea7f332301a12e8511d0`. The 12 recording hashes, game telemetry and measured JAR are checked against the [original evidence index](../../benchmarks/keelhelm-crossfire/stopped-attempt/independent-outcome-review.json). Measured JAR SHA-256 `fe6498d461ebc6c92e5d2d03e047c91771166c3d339e49ed647cf47966164d96`. Tank Royale 1.4.0 server SHA-256 `16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c`. The analysis ran on minimind with the pinned Temurin 25.0.5+8 runtime. Source, compiled classes, original evidence index and outputs are archived on minimind. [Archive digests](../../benchmarks/keelhelm-steering/artifact-proof.json)

With the retained evidence extracted and JDK 25 on PATH, set `EVIDENCE` to the directory containing the pinned server.jar and games folder. Run from the repository root.

```sh
classes=$(mktemp -d)
cp="$classes:experiments/KeelHelmCrossfire/KeelHelmCrossfire.jar:$EVIDENCE/server.jar"
javac -d "$classes" -cp "$cp" tournament/tools/{CrossfireReplay,SurvivalPulseReplay,SteeringReplay}.java
for arm in baseline steer; do
  java -Xmx256m -cp "$cp" SteeringReplay "$EVIDENCE" "$arm" > "$classes/$arm.json"
  cmp "$classes/$arm.json" "benchmarks/keelhelm-steering/$arm.json"
done
python3 tournament/tools/keelhelm_steering_summary.py "$classes/baseline.json" "$classes/steer.json"
```

[Registered test](keelhelm-steering-protocol.md) · [Numerical results and criteria](../../benchmarks/keelhelm-steering/summary.json) · [Issue #149](https://github.com/stephanelee/robocode/issues/149) · [PR #150](https://github.com/stephanelee/robocode/pull/150)
