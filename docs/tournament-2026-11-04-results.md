# Minimind invitational — Swiss stage complete

[Tournament and replay video](https://robocode-elimination-cup.stephane-lee.chatgpt.site) · [Running guide](tournament-running.md) · [Official scores and file hashes](../benchmarks/tournament-2026-11-04/official-results.json)

**70 validated games / 1,400 competitive battle rounds**, covering unranked practice and all four Swiss rounds. An excluded 20-roundsolo startup brings the completed physical total to 1,420. The top eight advance to double-elimination playoffs; no playoff game has started.

## Practice (unranked)

| Winner | Combined score | Opponent | Official first places | Points |
|---|---:|---|---:|---:|
| DiamondChallenger | 4693–939 | AISurvivor | 39/40 | Unranked |
| Walls | 5903–408 | RamFire | 38/40 | Unranked |
| Breakwater | 3364–251 | Aegis | 25/40 | Unranked |
| Adaptive Prime | 6545–160 | SpinBot | 40/40 | Unranked |
| Keel | 2892–2400 | Diamond Port | 32/40 | Unranked |
| KeelHelm | 4011–146 | Retcon | 40/40 | Unranked |
| Prism | 6153–194 | Meridian | 40/40 | Unranked |

## Swiss round 1

| Winner | Combined score | Opponent | Official first places | Points |
|---|---:|---|---:|---:|
| Adaptive Prime | 4659–1272 | Breakwater | 31/40 | 3 |
| KeelHelm | 6419–220 | Walls | 40/40 | 3 |
| DiamondChallenger | 7009–40 | Aegis | 40/40 | 3 |
| Keel | 4162–127 | Retcon | 40/40 | 3 |
| Diamond Port | 7203–17 | RamFire | 40/40 | 3 |
| Meridian | 3506–967 | AISurvivor | 32/40 | 3 |
| Prism | 6556–0 | SpinBot | 40/40 | 3 |

## Swiss round 2

| Winner | Combined score | Opponent | Official first places | Points |
|---|---:|---|---:|---:|
| Diamond Port | 3481–1935 | Prism | 30/40 | 3 |
| Keel | 4196–1025 | DiamondChallenger | 38/40 | 3 |
| KeelHelm | 3968–1781 | Adaptive Prime | 38/40 | 3 |
| Breakwater | 2400–268 | Meridian | 40/40 | 3 |
| AISurvivor | 6317–73 | Walls | 40/40 | 3 |
| Retcon | 7149–3 | Aegis | 40/40 | 3 |
| SpinBot | 7848–4513 | RamFire | 33/40 | 3 |

## Swiss round 3

| Winner | Combined score | Opponent | Official first places | Points |
|---|---:|---|---:|---:|
| KeelHelm | 3023–2288 | Keel | 26/40 | 3 |
| Diamond Port | 4586–1082 | Adaptive Prime | 36/40 | 3 |
| Prism | 6388–424 | DiamondChallenger | 38/40 | 3 |
| Breakwater | 2629–853 | Retcon | 40/40 | 3 |
| Meridian | 4579–60 | SpinBot | 39/40 | 3 |
| AISurvivor | 7213–37 | RamFire | 40/40 | 3 |
| Walls | 6652–200 | Aegis | 40/40 | 3 |

## Swiss round 4

| Winner | Combined score | Opponent | Official first places | Points |
|---|---:|---|---:|---:|
| KeelHelm | 2717–2519 | Diamond Port | 32/40 | 3 |
| Keel | 2852–2391 | Prism | 31/40 | 3 |
| Adaptive Prime | 4136–1459 | Meridian | 34/40 | 3 |
| Breakwater | 2400–777 | AISurvivor | 40/40 | 3 |
| Retcon | 6370–116 | Walls | 40/40 | 3 |
| DiamondChallenger | 6065–160 | SpinBot | 40/40 | 3 |
| RamFire | 6771–2344 | Aegis | 40/40 | 3 |

## Final Swiss standings

| Rank / seed | Bot | Points | Buchholz | Mean pairing score share |
|---:|---|---:|---:|---:|
| 1 / seed 1 | KeelHelm | 12 | 27 | 68.63% |
| 2 / seed 2 | Keel | 9 | 30 | 68.72% |
| 3 / seed 3 | Diamond Port | 9 | 27 | 73.26% |
| 4 / seed 4 | Breakwater | 9 | 24 | 65.61% |
| 5 / seed 5 | Adaptive Prime | 6 | 36 | 50.64% |
| 6 / seed 6 | Prism | 6 | 27 | 68.78% |
| 7 / seed 7 | Meridian | 6 | 24 | 53.30% |
| 8 / seed 8 | AISurvivor | 6 | 21 | 61.11% |
| 9 / out | Retcon | 6 | 21 | 56.41% |
| 10 / out | DiamondChallenger | 6 | 18 | 55.68% |
| 11 / out | Walls | 3 | 24 | 25.83% |
| 12 / out | SpinBot | 3 | 21 | 16.84% |
| 13 / out | RamFire | 3 | 18 | 27.88% |
| 14 / out | Aegis | 0 | 18 | 7.31% |

AISurvivor takes seed 8 ahead of Retcon on mean pairing share after equal points and Buchholz; no exact qualification-boundary draw was required.

## Recording and video

Round 4 added observer recording prospectively at the user’s request. All 14 replays pass gzip checks and contain one game-start, one game-end, and completed round coverage 1–20. Their paths, SHA256 s, sizes and structural audits remain local. The user also explicitly authorized the rendered MP4 on the public Site. It shows all 20 rounds of KeelHelm vs Diamond Port/game 1, live energy and official scores after each completed round. Duration 5:42.7;1280×800, H.264/yuv420p, 20 fps,240 simulationturns/sec. Final displayed scores 1,231–1,447 match the official game output. The movie is a rendered view of recorded data. Codec metadata and sampled frames were checked; browser/desktop GUI playback was not inspected.

## Provenance, failures and limitations

All eight home bots are frozen at `1f6e6a38582838fe891330a0b3e7a7d9f1668279`. Retcon 0.2.0 is pinned to `28981c7a469ac064194b41001cd7183ec5ecf189`. Diamond Port 1.8.28 and Adaptive Prime 1.0 are pinned to `67b68c22e41c18281a3fce9ef638e3c23a0b47cf`. The runner/server and official Java samples use the existing Tank Royale 1.4.0 hash pins.

Actual environment: minimind/macOS/ARM64; Java 17.0.21 builds, Temurin 25.0.5+8 battles, isolated CPython 3.12.16/API 1.3.1 native guests. The actual macOS runtime differs from the archived Linux/CPython 3.13 wheel lock. All eight home JAR parity checks pass; seven experimental suites pass 158 tests. AISurvivor retains 83 passes, 5 exact replay-value failures and 1 skip. The first gun-turn mismatch is 2.0869864576583836 versus 2.086986457658327; its precise cause remains unconfirmed. No gameplay, parameter or test changes were made.

The initial practice staging error produced 20 invalidAISurvivor-only rounds; they are excluded from every competitive result and retained. A second startup was interrupted before an official output. A Swiss-round-1 sample-launcher preflight interruption consumed zero rounds; only unplayed games continued. No valid competitive game was discarded or rerun. All original failures, source/JAR manifests, runtime data, models, telemetry, invocation records, runnable bot bundles and recordings remain local. The public repository contains official score evidence and code; the approved rendered video is on the Site.

Every game started cold, with normal learning only within its 20 rounds. Seeds do not control engine starts or native opponent randomness. Adaptive rounds are dependent. Recording in round 4 is a declared runtime change; no blanket zero-skip, timing-equivalence, broad-strength or isolated-learning claim follows. The tournament runner currently automates practice/Swiss stages; playoff execution remains documented as a separate phase.

Disposition: accept 70 complete two-bot games as descriptive event evidence; reject and retain the invalid solo startup. No gameplay candidate is promoted.
