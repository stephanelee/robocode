# Confirmed-outcome melee power: measured result

**Registered decision: reject.** Forty fresh cold games / 520 rounds ran serially on minimind. Candidate 0.6.0 forks accepted 0.2.1 with circular aim; it does not combine the rejected learned gun.

| Scenario | Control mean score share | Candidate mean score share | Change |
| --- | ---: | ---: | ---: |
| melee | 30.04% | 31.60% | +1.55 pp |
| Diamond Port | 51.88% | 56.89% | +5.01 pp |
| Retcon | 91.39% | 95.35% | +3.96 pp |
| Walls | 95.08% | 94.19% | -0.89 pp |
| SpinBot | 100.00% | 100.00% | +0.00 pp |

Mean normalized melee survival share: 33.09% → 35.21% (+2.12 pp). Means weight each game equally. Matching bot seeds do not pair engine starts. The from-start duel controller remains inherited; unpaired duel differences are guard measurements, not causal proof of a duel-policy change.

| Registered gate | Pass |
| --- | --- |
| meleeGain | False |
| meleeShare | True |
| Diamond Port | True |
| Retcon | True |
| Walls | True |
| SpinBot | True |
| survival | True |
| confirmedChanged | False |
| multipleNamedTargets | True |

Physically confirmed power changes: 85. Named intended targets: Keel 79, Diamond Port 6. The exposure gate prevents crediting random score variation when the policy is insufficiently exercised. Real feedback windows are diagnostic outcome filters, not calibrated probabilities.

The rule changed 85 of 10,459 confirmed melee shots (0.81%). All changes were 1.6→2.0 power at ranges 261–499. This is a read-only activation description with zero new games, not a competitive-strength or calibrated-probability claim.

## Validation and provenance

Independent audit replays all global F/R/E sequences, 32-outcome FIFO windows and request-context snapshots; verifies selected/baseline/actual powers and exact rule; matches legal callback masks/positive damage against all available same-tick physical events; and checks censor/eviction conservation. Observer-only events after callback cutoff are not assumed to reach the bot. All 6,995 frozen input hashes, staged entrants/guest/sample support, recordings/CRC/contiguous ticks/round endpoints, final official score components, resources and owned cleanup validate. Zero own skips or partial games; maximum logged own loop is 6 ms (excludes callbacks/termination), minimum sampled available memory is 43%, and all resource/cleanup guards pass.

Java 17: 29 tests pass (Homebrew OpenJDK 17.0.21.0.3) and source/committed-JAR parity across 833 entries, candidate SHA256 e6227d4b561754f7ef93aceb1411620f77df39bfdda758fb043afdbdaeda004b. Control SHA256 19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b. Four Python schedule/gate/replay contracts pass. Initial two unit fixture expectations were corrected before freeze; their failed log is retained and spent zero competitive games. Fixtures establish contracts only. Games use frozen JDK 25/Python/API/runner/opponents and the unchanged 100000 us deadline/heaps/GC.

Exact prospective budget: 8×20-round melee per arm, 4×10-round duels per arm against each Diamond Port and Retcon, 2×5-round duels per arm against each Walls and SpinBot. All processes/homes cold; two melee slot rotations and alternating arm order, seeds 2026112300+pairIndex for bot choices only. No warmup, tuning, historical pooling, score-driven stopping, retry/replacement or resume. All valid planned games finished.

Fresh cost: 40 games / 520 complete rounds, zero partial. Cumulative KeelHelm development/diagnostic accounting: 3,030 complete plus 3 historical partial rounds, separate from the completed tournament. Prior 80-round PR #128 snapshot and 520-round rejected gun study are not pooled. Finite unpaired engineering gates do not establish general strength, significance or causes of prior timing failures. Accepted default stays 0.2.1 on rejection.

Every original recording, F/R/E trace, source/JAR/support snapshot, invocation, runtime/opponent hash, resource/cleanup sample, build log/XML and auditor is retained in robocode-keelhelm-outcome-power/2026-11-05/full-artifacts.tar.gz (587,590,374 bytes), SHA256 b6e6d2bdce724fae58671f60b8e6cecadc477941f5466eb1c41a8eeca6eb794b. Public results omit bulky telemetry but preserve its hashes and original bytes in the archive. Previously authorized cleanup removed only older inactive owned temporary Git checkpoint stores after full layout/age/symlink/open-file checks; recent undo, source and all game artifacts stayed intact.

[Protocol](keelhelm-outcome-power-protocol.md), [implementation](keelhelm-outcome-power-implementation.md), [official per-game results](../../benchmarks/keelhelm-outcome-power/results.json), [independent audit](../../benchmarks/keelhelm-outcome-power/independent-audit.json), [pre-game review](../../benchmarks/keelhelm-outcome-power/pre-game-review.json), [provenance](../../benchmarks/keelhelm-outcome-power/provenance.json), [archive proof](../../benchmarks/keelhelm-outcome-power/artifact-proof.json), [issue #131](https://github.com/stephanelee/robocode/issues/131), [PR #130](https://github.com/stephanelee/robocode/pull/130).
