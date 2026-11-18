# KeelHelm long-range melee firepower

**Retain KeelHelm 0.2.1.** The higher-power bot earned 1,087.41 points from shots fired in melee, compared with 986.38 for accepted KeelHelm. Mean total-score share was 30.42% for the candidate and 31.79% for the control. The fixed budget was 16 new serial games / 68 rounds on minimind. The candidate gained damage in two of four paired layouts. Its official survival score fell by 150 and its round wins fell from 11 to 7.

## Replay evidence

Twelve retained four-bot games and 60 rounds from the original Crossfire study supplied the same opponent trajectories for both arms. The ordinary gun reproduced 4,042 actual melee shots, 663 hit events and 3,071.91 credited damage. The higher-power gun raised damage to 3,693.54 (+20.24%) across all 12 games, with every game contributing a positive delta. Damage to Diamond specifically rose from 1,012.75 to 1,206.39. The simulated higher-power gun landed 617 hit events from 3,968 shots, consumed 5,233.59 energy versus 4,358.89, and boosted 2,410 shots. New shot timing created 1,055 slots and displaced 1,129. The replay simulated aim, energy, bullet travel and later cooldown. Opponents followed their recorded movement and the counterfactual damage credit used their recorded energy. Unobserved continuations of destroyed opponent bullets are unavailable. The live engine measured how the interacting bots actually responded.

## Official live results

The first-place column counts engine round wins. Mean share weights each game equally. Bullet damage and survival are official score components, excluding their respective bonuses. Damage from melee-fired shots includes shots fired with more than one opponent alive that landed later.

| Scenario | Arm | Games / rounds | KeelHelm score | Mean share | Round firsts | Bullet damage | Survival |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| melee | control | 4 / 20 | 3,509 | 31.79% | 11 | 1,081 | 2,000 |
| melee | candidate | 4 / 20 | 3,286 | 30.42% | 7 | 1,162 | 1,850 |
| Diamond Port | control | 2 / 10 | 823 | 62.39% | 10 | 188 | 500 |
| Diamond Port | candidate | 2 / 10 | 823 | 62.39% | 10 | 188 | 500 |
| Retcon | control | 1 / 2 | 177 | 90.77% | 2 | 47 | 100 |
| Retcon | candidate | 1 / 2 | 177 | 90.77% | 2 | 47 | 100 |
| Walls | control | 1 / 2 | 295 | 93.65% | 2 | 160 | 100 |
| Walls | candidate | 1 / 2 | 295 | 93.65% | 2 | 160 | 100 |

## Top four melee

| Bot | Score with control | Score with candidate | Round firsts with control | Round firsts with candidate |
| --- | ---: | ---: | ---: | ---: |
| Diamond Port | 3,632 | 3,723 | 7 | 9 |
| KeelHelm | 3,509 | 3,286 | 11 | 7 |
| Prism | 3,060 | 3,215 | 1 | 3 |
| Keel | 831 | 613 | 1 | 1 |

During the melee firing phase the candidate dealt 228.62 damage to Diamond Port versus 235.26 for the control. Against Keel it dealt 546.63 versus 490.30, and against Prism it dealt 312.16 versus 260.82. In direct 1v1 play both versions beat Diamond 10–0 in round wins, 823–496 in total points. [Damage by actual victim and game](../../benchmarks/keelhelm-long-range-power/live-melee-victims.json)

## Decision criteria

| Registered check | Measured change | Result |
| --- | ---: | --- |
| Melee-phase damage > control | +101.03 points | Pass |
| Melee score share ≥ control | -1.36 pp | Fail |
| Diamond 1v1 share within 5 pp | +0.00 pp | Pass |
| Retcon share within 5 pp | +0.00 pp | Pass |
| Walls share within 5 pp | +0.00 pp | Pass |

## Reproduction and provenance

Execution commit `fa2ac92a9d6182fad2f8f9340602ad442e47518d`. Freeze SHA-256 `0bf4597227cb02042b90887452e8975aa1085de6d5158b4126cc9e03efe0bf4e`. Candidate `0.2.1-power145` JAR `81d89aed7a60d2a968e25922d520576f3d8c09d7ae5e5ea36cbfdb43ca274340`, accepted `0.2.1` JAR `19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b`. The candidate differs by one gameplay condition in MeleePolicy and retains the accepted duel code. Java 17 clean build, 24 tests and byte-identical JAR parity passed. Live games used the pinned native Temurin 25.0.5+8, Python 3.12.16, Tank Royale runner/server 1.4.0, Diamond Port 1.8.28, Keel 0.1.0, Prism 0.2.0, pinned Retcon and the official Walls sample. Launches used fresh processes and fixed per-pair starts; native randomness remains. Multiple rounds within a game share state and are not independent replicates.

All recordings, official scoreboards, start poses, transport, bot telemetry, live resource samples, cleanup records, exclusive receipt, the full input freeze and staged copies are retained privately on minimind under `~/.codex/artifacts/robocode-keelhelm-firepower/145/`. The per-game publication includes the recording hashes. The sealed archive also contains all 6,997 frozen inputs and passed gzip integrity verification. [Private archive digest and size](../../benchmarks/keelhelm-long-range-power/artifact-proof.json)

[Prospective replay and live protocol](keelhelm-long-range-power-protocol.md) · [Control replay](../../benchmarks/keelhelm-long-range-power/control.json) · [Higher-power replay](../../benchmarks/keelhelm-long-range-power/higher.json) · [Official scores and recording hashes](../../benchmarks/keelhelm-long-range-power/live-games.json) · [Live summary](../../benchmarks/keelhelm-long-range-power/live-summary.json) · [Issue #145](https://github.com/stephanelee/robocode/issues/145) · [PR #146](https://github.com/stephanelee/robocode/pull/146)

An independent final audit checked all 16 recordings, official results, starting poses, input hashes and cleanup records. It reproduced the live damage totals and the direct damage to Diamond. All 68 rounds finished with zero KeelHelm skipped turns. Peak logged bot-loop time was 6 ms and minimum sampled available memory was 53%.

## Reproducing the measurements

For the replay, extract the private Crossfire evidence archive and set `EVIDENCE` to the directory containing `games` and the pinned `server.jar`. With Java 25 on PATH, run these commands from the repository root. The reader validates the candidate JAR and all input recording hashes. The control mode checks the original shot count, physical contacts and damage.

```sh
classes=$(mktemp -d)
cp="$classes:experiments/KeelHelmCrossfire/KeelHelmCrossfire.jar:$EVIDENCE/server.jar"
javac -cp "$cp" -d "$classes" tournament/tools/CrossfireReplay.java tournament/tools/MidpointReplay.java tournament/tools/LongRangePowerReplay.java
for arm in control higher; do
  java -Xmx256m -cp "$cp" LongRangePowerReplay "$EVIDENCE" "$arm" > "$classes/$arm.json"
  cmp "$classes/$arm.json" "benchmarks/keelhelm-long-range-power/$arm.json"
done
```

The live runner uses `tournament/tools/keelhelm_long_range_power.py prepare` to stage a dedicated evidence directory from the pinned tournament source, entry manifests, JDK 25 and Python executable. Build the candidate project with Java 17 using `clean test assemble` first. Preparation verifies the committed and freshly built JARs. The `run` entry point takes the prepared evidence directory and its reviewed freeze digest, creates one exclusive receipt, and plays the scheduled layouts serially on localhost. The original receipt and private native recordings are preserved.
