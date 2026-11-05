# Minimind invitational results

[Public site](https://robocode-elimination-cup.stephane-lee.chatgpt.site) · [Running instructions](tournament-running.md) · [Prospective protocol](tournament-2026-11-04-protocol.md) · [Official result arrays and file hashes](../benchmarks/tournament-2026-11-04/official-results.json)

**Practice and Swiss rounds 1–2 completed: 42 validated games, 840 competitive battle rounds.** An initial invalid 20-round solo game brings the known physical budget to 860 rounds. No GitHub Actions were used for battles. No bot has been eliminated; two Swiss rounds remain.

## Round 0: unranked practice

| Winner | Combined score | Opponent | Winner’s round first places | Points |
|---|---:|---|---:|---:|
| DiamondChallenger | 4693–939 | AISurvivor | 39/40 | Unranked |
| Walls | 5903–408 | RamFire | 38/40 | Unranked |
| Breakwater | 3364–251 | Aegis | 25/40 | Unranked |
| Adaptive Prime | 6545–160 | SpinBot | 40/40 | Unranked |
| Keel | 2892–2400 | Diamond Port | 32/40 | Unranked |
| KeelHelm | 4011–146 | Retcon | 40/40 | Unranked |
| Prism | 6153–194 | Meridian | 40/40 | Unranked |

## Swiss round 1

| Winner | Combined score | Opponent | Winner’s round first places | Points |
|---|---:|---|---:|---:|
| Adaptive Prime | 4659–1272 | Breakwater | 31/40 | 3 |
| KeelHelm | 6419–220 | Walls | 40/40 | 3 |
| DiamondChallenger | 7009–40 | Aegis | 40/40 | 3 |
| Keel | 4162–127 | Retcon | 40/40 | 3 |
| Diamond Port | 7203–17 | RamFire | 40/40 | 3 |
| Meridian | 3506–967 | AISurvivor | 32/40 | 3 |
| Prism | 6556–0 | SpinBot | 40/40 | 3 |

## Swiss round 2

| Winner | Combined score | Opponent | Winner’s round first places | Points |
|---|---:|---|---:|---:|
| Diamond Port | 3481–1935 | Prism | 30/40 | 3 |
| Keel | 4196–1025 | DiamondChallenger | 38/40 | 3 |
| KeelHelm | 3968–1781 | Adaptive Prime | 38/40 | 3 |
| Breakwater | 2400–268 | Meridian | 40/40 | 3 |
| AISurvivor | 6317–73 | Walls | 40/40 | 3 |
| Retcon | 7149–3 | Aegis | 40/40 | 3 |
| SpinBot | 7848–4513 | RamFire | 33/40 | 3 |

## Cumulative Swiss standings

| Rank | Bot | Points | Buchholz | Mean pairing score share |
|---:|---|---:|---:|---:|
| 1 | Keel | 6 | 6 | 88.70% |
| 2 | KeelHelm | 6 | 3 | 82.85% |
| 3 | Diamond Port | 6 | 3 | 82.02% |
| 4 | Prism | 3 | 9 | 67.86% |
| 5 | Adaptive Prime | 3 | 9 | 54.77% |
| 6 | DiamondChallenger | 3 | 6 | 59.53% |
| 7 | Breakwater | 3 | 6 | 55.70% |
| 8 | Retcon | 3 | 6 | 51.46% |
| 9 | Meridian | 3 | 6 | 44.21% |
| 10 | AISurvivor | 3 | 3 | 60.24% |
| 11 | SpinBot | 3 | 3 | 31.75% |
| 12 | RamFire | 0 | 9 | 18.37% |
| 13 | Walls | 0 | 9 | 2.23% |
| 14 | Aegis | 0 | 6 | 0.30% |

## Provenance and retained failures

The eight home bots are frozen at main commit `1f6e6a38582838fe891330a0b3e7a7d9f1668279`. Retcon 0.2.0 uses rumble-bots@`28981c7a469ac064194b41001cd7183ec5ecf189`; Diamond Port 1.8.28 and Adaptive Prime 1.0 use robocode-bot@`67b68c22e41c18281a3fce9ef638e3c23a0b47cf`. Official runner/server/sample assets are Tank Royale 1.4.0 with the existing literal hashes.

Actual host: minimind, macOS/ARM64. Build/test: Homebrew Java 17.0.21. Battles: Temurin 25.0.5+8 LTS. Python guests: isolated CPython 3.12.16 with API 1.3.1, with exact packages in `tools/tournament-macos-requirements.txt`. This is not the Linux CPython 3.13 wheel runtime. All guest source bytes were checked against existing locks.

All eight home JARs pass source/committed-JAR content parity; AISurvivor raw JAR hashes also match. The seven experimental suites pass 158 tests. AISurvivor: 83 passed, 5 exact replay-value failures, 1 skipped (89 tests). The first mismatch is gunTurnRate 2.0869864576583836 versus 2.086986457658327. The precise platform/JDK cause is unconfirmed. No test or gameplay changes were made. The initial fixture extraction omission and final numeric failures remain in local evidence; no passing-CI claim is made.

The first practice staging attempt used the wrong DiamondChallenger folder name. The runner then completed 20 AISurvivor-only rounds, which are excluded from all results. A second startup was stopped before an official game output. The corrected attempt preserved the draw; no valid competitive game was discarded. Swiss round 1 later stopped before launching Walls because a shell-launcher assertion was incorrectly applied to official Java source samples. That preflight interruption consumed zero rounds. The staged files and log were retained and only never-launched games continued.

The full raw artifacts remain local, including runnable isolated bot bundles, official arrays, telemetry, models, runner logs, invocation records, source/JAR manifests, build/test evidence and runtime provenance. Only profiles and score results are public; no artifact archive is exposed by the Site or this PR.

Each game started cold with fresh private paths; normal learning persisted within its 20 rounds. No tuning or separate training was added. Bot seed overrides use 20261104; arena starts, native opponent randomness and scheduling are uncontrolled. Native skipped-turn telemetry is not uniformly available; no blanket zero-skip or timing-equivalence claim is made. These dependent adaptive rounds describe this frozen field and regime, not broad playing strength, isolated learning gains, or classic-Diamond equivalence.

Disposition: accept the 42 complete validated two-bot games as descriptive tournament evidence; reject and retain the invalid solo startup. No gameplay candidate is promoted. Further rounds require an explicit command; no round 3 or playoffs have started.
