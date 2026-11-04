# Held gun plan improves the observed stall but loses the Diamond comparison

**Research decision: reject the held plan as a competitive improvement; no confirmation or promotion.** The registered descriptive schedule completed all12 games/60 rounds, with0 invalid rounds and0 Meridian skipped turns. This is the registered completion rule, not a post-hoc numeric promotion gate. Diamond won all10 candidate rounds; the control won1/10. Production AISurvivor is unchanged. [PR92](https://github.com/stephanelee/robocode/pull/92) · [issue90](https://github.com/stephanelee/robocode/issues/90) · [sole acquisition](https://github.com/stephanelee/robocode/actions/runs/41766660321).

## Whole-study results

Each arm has two fresh five-round games per opponent. Share is the equal-game mean. Hits/shots and damage/energy below are totals across the ten rounds; the hit fraction is pooled, not the mean of game hit rates.

| Opponent | Arm | Wins /10 | Mean share | Hits /shots | Damage dealt /taken | Firing energy | Walls /rams /skips |
|---|---|---:|---:|---:|---:|---:|---:|
| Diamond Port | baseline | 1 | 20.7739% | 38/513 (7.407%) | 230.613/482.712 | 651.968 | 0/0/0 |
| Diamond Port | candidate | 0 | 12.2121% | 31/448 (6.920%) | 168.706/528.589 | 602.225 | 5/0/0 |
| Retcon | baseline | 10 | 98.4258% | 53/447 (11.857%) | 385.013/17.600 | 686.693 | 1/0/0 |
| Retcon | candidate | 10 | 94.2142% | 58/465 (12.473%) | 417.617/66.800 | 728.831 | 2/0/0 |
| AISurvivor | baseline | 10 | 93.4834% | 81/418 (19.378%) | 623.467/92.000 | 604.358 | 4/0/0 |
| AISurvivor | candidate | 9 | 88.8594% | 81/429 (18.881%) | 594.646/88.973 | 665.738 | 1/0/0 |

Diamond share was20.773894% for the control and12.212144% for the candidate (−8.561750 percentage points). Candidate damage dealt per100 shots was40.878 and34.659, versus46.019 and43.949 in control games. Equal-game incoming damage per100 enemy shots was103.737 for candidate versus61.523 for control. These are descriptive differences between independent engine starts, not a paired causal estimate.

## Game variation and liveness

| Game | Arm | Share | Wins /5 | Hits /shots | Shots /1,000 observer turns | Longest common-opportunity no-fire run |
|---|---|---:|---:|---:|---:|---:|
| gp60-01-diamond-s0-baseline | baseline | 25.0000% | 1 | 18/249 | 39.480 | 1152 |
| gp60-02-diamond-s0-candidate | candidate | 12.6074% | 0 | 16/216 | 58.920 | 15 |
| gp60-03-retcon-s0-baseline | baseline | 97.2325% | 5 | 27/222 | 57.468 | 17 |
| gp60-04-retcon-s0-candidate | candidate | 98.2729% | 5 | 30/233 | 62.150 | 7 |
| gp60-05-ais-s0-baseline | baseline | 94.3089% | 5 | 43/190 | 58.915 | 24 |
| gp60-06-ais-s0-candidate | candidate | 83.5244% | 4 | 41/217 | 55.655 | 21 |
| gp60-07-diamond-s1-candidate | candidate | 11.8168% | 0 | 15/232 | 59.594 | 38 |
| gp60-08-diamond-s1-baseline | baseline | 16.5478% | 0 | 20/264 | 52.853 | 29 |
| gp60-09-retcon-s1-candidate | candidate | 90.1554% | 5 | 28/232 | 62.399 | 18 |
| gp60-10-retcon-s1-baseline | baseline | 99.6190% | 5 | 26/225 | 59.618 | 16 |
| gp60-11-ais-s1-candidate | candidate | 94.1945% | 5 | 40/212 | 58.258 | 30 |
| gp60-12-ais-s1-baseline | baseline | 92.6579% | 5 | 38/228 | 57.722 | 29 |

Against Diamond, the candidate's longest runs were15/38 turns versus control1,152/29; its firing rates were58.920/59.594 versus39.480/52.853 shots per1,000 observer turns. Total candidate shots were lower because its games were shorter. Only one of these two control games exhibited a long stall. This does not estimate how often stalls occur generally. The common-opportunity diagnostic uses fresh legal scans, a cold gun, the original current-state power/energy/reserve tests and next-tick confirmed firing. It deliberately omits aim alignment and held-plan-specific affordability and excludes terminal decisions without a following tick.

The fixture's liveness prediction transferred to the observed candidate firing behavior, but better liveness did not establish better targeting or strength. It did not earn a Diamond win. Retcon wins were10/10 for both arms; AISurvivor wins were10/10 control and9/10 candidate. All three candidate equal-game score means were lower. With only two games per arm and unseeded starts, do not infer reliable causal effect or superiority from these samples. Keep every outcome; do not pool older games or reuse them as confirmation.

Full per-game and per-round metrics, firing/hit rates, damage, energy, learned/fallback requests, crossing labels, contacts and variation are in [decision.json](../../benchmarks/meridian-gun-plan/decision.json), [results.csv](../../benchmarks/meridian-gun-plan/results.csv) and [summary.json](../../benchmarks/meridian-gun-plan/summary.json). Every official recording, private marker stream, command/environment/staged hashes, runner log and AIS checkpoint/telemetry is retained in [raw/](../../benchmarks/meridian-gun-plan/raw/).

## Frozen execution and preserved setup failures

Acquisition head04fc2c87b6f961846e755a98e93213d320af9374, tree589dd107d20d0a31eb119def2b3ac6d357b2b882; tooling manifest529acb1d84ce1f0634eed4a4608244bad47a8b070f944617a404a757c24eab52. Candidate source4a02dfd, Meridian.java5268264c49dd1150eb862b4cd59814a9e589abb3e5f284c56bf07dd07989d0d1, JAR7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe. Exact-v3 control809b880 uses JAR5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9. AIS JARb41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800.

Java17.0.21+9 built/tested all three participants and passed source/JAR parity before Java25.0.5+8 battles. Actual runtime was Python3.13.18, Ubuntu26.04 image20261026.180.1, pinned Tank Royale1.4.0 runner02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf. Diamond Port1.8.28 remains upstream67b68c22e41c18281a3fce9ef638e3c23a0b47cf with selected-source hash a9bf0b5d62b510ce3a405db9dcbe11cf2c1bce3f982e2a5d12b9fdc76b8fbc92; Retcon0.2.0 remains28981c7a469ac064194b41001cd7183ec5ecf189, tree5f7dfaaedefc39bd93d86440a2996342583d57b75b861942dc1e1899b7164bb2, API1.3.1 acb1858a7b83b45ddd153db52fcea4d855b95cc442f643f94f299f0ee2258ce2. Full exact hashes, installed package files and host/boot identity remain in the runtime/activation files.

The independent [exact-head clearance](https://github.com/stephanelee/robocode/pull/92#issuecomment-6621635267) and [external activation binding](https://github.com/stephanelee/robocode/issues/90#issuecomment-6621646489) preceded games. The fixed annotated receipt objectf6cab37f06931b730c5b6fdc68a2256f2759327d remains at refs/tags/experiment-receipts/meridian-gp60-issue90 and must never be deleted or reset. No repeat or additional acquisition is allowed under this registration.

Original zero-game failure41765322623 (root clean removed Java provenance) remains separately preserved. Corrected zero-game preflight41765933504 passed; its package hashes were frozen before acquisition. No runtime mismatch, setup failure or old games were omitted or turned into strength evidence. The full30,684,292-byte acquisition artifact has SHA2560e63d31532e2cd03657d14352f69c80b111a148ae3543457b4542a66e070facd. It includes complete staged bundles in addition to the curated raw files. Its original ZIP is retained locally and linked in summary.json; hosted artifact retention is90 days.

Independent [outcome review](https://github.com/stephanelee/robocode/pull/92#issuecomment-6621882708) reproduced all12 games using a standalone parser importing no repository reducers. It checked official scores/endpoints, shot/hit/damage/contact counts, liveness, all staged assets/opponent locks, AIS finite cold models, receipt and runtime/package identities. [Portable audit and original recomputation](../../benchmarks/meridian-gun-plan/analysis/README.md).

## Next direction

Do not retune or repeat this gun candidate. A separate source audit identified a possible omitted first-contact interval in the existing incoming-wave danger window. Before changing movement, register and independently review a zero-game swept-segment diagnostic, including neighboring forecast ticks, retirement, complete route scores and action changes. An analytic example is not evidence Diamond produces it, and fixture geometry cannot establish competitive strength. Any live candidate needs a new source/JAR freeze and fresh bounded comparison.
