# Learned melee displacement gun: measured result

**Registered decision: reject.** Forty fresh cold games / 520 rounds ran serially on minimind. The isolated candidate is KeelHelm 0.5.0; control is accepted 0.2.1. No power, movement, radar, target or duel policy changed.

| Scenario | Control mean score share | Candidate mean score share | Change |
| --- | ---: | ---: | ---: |
| melee | 29.87% | 31.52% | +1.65 pp |
| Diamond Port | 52.04% | 45.18% | -6.85 pp |
| Retcon | 92.67% | 94.59% | +1.93 pp |
| Walls | 94.23% | 93.39% | -0.83 pp |
| SpinBot | 100.00% | 99.80% | -0.20 pp |

Melee mean survival-point share: 33.54% → 35.24% (+1.70 pp). Each game is normalized and weighted equally. Engine placements are unpaired; matching seeds govern bot choices only.

| Registered gate | Pass |
| --- | --- |
| meleeGain | False |
| meleeShare | True |
| Diamond Port | False |
| Retcon | True |
| Walls | True |
| SpinBot | True |
| survival | True |
| confirmedChanged | True |
| multipleNamedTargets | True |

Actual materially changed confirmed shots: 6,349. Named targets: Keel 3,156, Diamond Port 1,142, Prism 2,051. These counts match legal physical BulletFired events and the preceding-tick melee request phase; desired and actual headings both differ from the captured circular reference by more than 0.1°.

## Validation and provenance

Independent raw audit verifies 6,995 frozen inputs, all official/recorded scores, CRC/contiguous ticks/round endpoints, staged entrant/shared-support files, resource samples and owned teardown. All games finish, zero own skips or partial games. Maximum logged own loop is 8 ms (excludes callbacks/termination), minimum sampled memory availability is 56%, and all resource/cleanup checks pass. Ledger hits/walls/collisions and callback-cutoff censored counts are checked against physical events; unresolved bullets are never declared misses.

The first independent audit incorrectly assumed that one physical bullet could have only one outcome. The pinned engine can report collision and bot-hit events for the same bullet in the same turn. The corrected auditor retains all events and validates callback counters against the available outcomes. Both auditor versions and receipts are retained; no game, score or experimental input changed. The accepted 0.2.1 source/JAR stays exact after rejection.

Java 17 build: 30 tests pass, source/committed-JAR byte parity across 835 archive entries. Candidate JAR SHA256 80f19d7951dd7cb169d56720b0d6f12e219264c3aad0387d75fbefcf56dba3c1. Control JAR SHA256 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Games use pinned JDK 25, Python/API/runner and frozen opponents with the unchanged 100000 us deadline/heaps/GC.

The prospective protocol fixes eight 20-round melee games per arm, four 10-round duels per arm against each Diamond Port and Retcon, and two 5-round duels per arm against each Walls and SpinBot. Each process/home is cold. Two complete melee slot rotations, alternating arm order, no tuning, historical pooling, score-driven stop, retry or replacement. Pre-game independent review corrected one cached-horizon reference bug before any gameplay; the unused first freeze and its candidate snapshot are retained with zero games.

Fresh measured cost: 40 games / 520 completed rounds, zero partial rounds. Cumulative KeelHelm development/diagnostic spend is 2,510 completed rounds plus three historical partial rounds, separate from the completed tournament. Prior accounting includes the separate 80-round latest snapshot in [PR #128](https://github.com/stephanelee/robocode/pull/128), which is not pooled into this comparison. These bounded engineering gates do not establish statistical significance, general strength or the cause of prior timing failures.

All original recordings, physical shot traces, staged source/JAR/support, invocations, resource/cleanup data, helper/auditor source and Java build logs/XML are retained in robocode-keelhelm-displacement/2026-11-05/full-artifacts.tar.gz (666,459,982 bytes), SHA256 7c301b68229eaab957571416d2c7010722c25a83d24ccab7125e74e9ffaf7353. Public per-game data omit bulky telemetry (hashes retained) and an unused inherited zero counter; canonical actual exposure is in summary.exposure.confirmedChanged. The original acquisition results remain byte-exact in the archive.

[Protocol](keelhelm-learned-gun-protocol.md), [implementation](keelhelm-learned-gun-implementation.md), [per-game official results](../../benchmarks/keelhelm-learned-gun/results.json), [independent raw audit](../../benchmarks/keelhelm-learned-gun/independent-audit.json), [pre-game review](../../benchmarks/keelhelm-learned-gun/pre-game-review.json), [provenance](../../benchmarks/keelhelm-learned-gun/provenance.json), [archive proof](../../benchmarks/keelhelm-learned-gun/artifact-proof.json), [issue #129](https://github.com/stephanelee/robocode/issues/129), [PR #130](https://github.com/stephanelee/robocode/pull/130).
