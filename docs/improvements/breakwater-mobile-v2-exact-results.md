# Breakwater 0.2 results against Diamond Port and AISurvivor

**Reject against Diamond.** The authorized attempt completed 40 valid rounds. Breakwater lost the 20-round Diamond pilot and missed both progress gates. Confirmation was skipped. It then won five rounds in the registered 20-round AISurvivor transfer game. [Registration](breakwater-mobile-v2-exact.md) · [PR #67](https://github.com/stephanelee/robocode/pull/67) · [Measurement run](https://github.com/stephanelee/robocode/actions/runs/41715478408)

| Opponent | Official score, Breakwater–opponent | Breakwater first places | Breakwater share | v1 historical wins / share |
| --- | ---: | ---: | ---: | ---: |
| Diamond Port 1.8.28 | 60–2,865 | 0/20 | 2.051282% | 0/20 / 0.169062% |
| AISurvivor 1.2.1 | 389–1,168 | 5/20 | 24.983943% | 0/20 / 1.115760% |

The Diamond pilot required >20% share, at least one first place, zero own skips and complete evidence. Breakwater recorded zero skipped turns, wall collisions or bot collisions in both games. The transfer result supplies five actual wins against AISurvivor, which won the other 15. Each opponent was tested in one fresh game from cold state. The historical v1 comparisons describe different games and cannot separate movement, gun, learning and matchup variation. [Diamond scoreboard](../../benchmarks/breakwater-mobile-v2-exact/breakwater-mobile-v2-pilot/battle.json) · [AISurvivor scoreboard](../../benchmarks/breakwater-mobile-v2-exact/breakwater-mobile-v2-ais-g01/battle.json) · [v1 Diamond](breakwater-v1-results.md) · [v1 AISurvivor](breakwater-vs-ais.md)

## What the controller did

| Breakwater metric | Diamond pilot | AISurvivor transfer |
| --- | ---: | ---: |
| Observed moving turns / decisions | 21,179 / 21,203 | 24,063 / 24,086 |
| Decisions with an inferred wave | 19,185 | 22,296 |
| Confirmed hits / shots | 10 / 603 (1.66%) | 16 / 1,289 (1.24%) |
| Damage dealt / taken (own events) | 59.96 / 1,396.37 | 83.26 / 260.82 |
| Energy spent firing | 652.49 | 1,461.08 |
| Mean fresh-observed range | 443.58 px | 430.60 px |

The defensive change did put Breakwater in motion on nearly every turn and freed the gun for hundreds of attacks. Against Diamond, the linear, stopped and circular forecasts respectively hit 4/127, 2/182 and 4/294. Even with continuous fire, Breakwater landed only ten shots in the entire pilot. Against AISurvivor, Breakwater took 260.82 event damage versus 2,004.00 in v1's historical game. AISurvivor hit 46 of 1,284 shots in this game. No component was tested independently. [Original experiment summary](../../benchmarks/breakwater-mobile-v2-exact/experiment.json)

Breakwater's 20 official round-end increments reconciled with each final scoreboard. Its raw win callbacks missed three transfer wins (rounds 6, 10 and 12); AISurvivor reported 13 local win flags against 15 official first places. Scores and wins use the official results. The two incoming/outgoing event ledgers can differ in damage coverage; the table uses Breakwater's own ledger consistently.

## Provenance

The [launcher registration](https://github.com/stephanelee/robocode/blob/cb1a2b752b9933b463565f6ddca59889e8249b0a/.github/workflows/breakwater-mobile-v2-exact.yml) selected Python 3.13.17 and checked out the original execution payload at `6010a4986ccea11be910ec6cdd67940a9fb568dc`. The original candidate source is `268f62180bae853fa9d0ca11fa754c941611c1d7`, JAR SHA-256 `13768c7e72f47cbcd2c62f051b021fb2af404e28d1aa68864e18d1710fa5580a`. Java 17 parity builds preceded JDK 25 play. All six runtime hashes matched the original freeze both in the no-game hosted check and in the actual measurement job. Diamond used the same locked source, wheels and Tank Royale 1.4.0 runtime. Each bot had an initially absent private model and metrics path per game. Breakwater seeds were 532001 and 532021; AISurvivor used 532022 in default training mode. The first authorization remains separately recorded as an invalid, zero-game attempt. [Original invalid report](breakwater-mobile-v2-results.md) · [Actual runtime fingerprint](../../benchmarks/breakwater-mobile-v2-exact/retry-provenance/actual-runtime.json) · [Raw evidence and replay commands](../../benchmarks/breakwater-mobile-v2-exact/README.md)
