# KeelHelm A/A calibration: measurement not ready

**Valid acquisition; STOP under the registered continuation gate.** Identical accepted KeelHelm 0.2.1 played both arms in 32 cold serial games / 240 rounds on minimind. No bot policy changed and no strength improvement is claimed.

| Mode | Scenario | A mean share | B mean share | A−B | Within allowance |
| --- | --- | ---: | ---: | ---: | --- |
| random | melee | 27.98% | 28.50% | -0.5153 pp | True |
| random | Diamond Port | 47.63% | 53.11% | -5.4779 pp | False |
| fixed | melee | 34.60% | 31.05% | +3.5479 pp | False |
| fixed | Diamond Port | 56.71% | 56.71% | +0.0000 pp | True |

Fixed melee +3.5479pp exceeds the prechosen 2 pp A/A allowance, so the original movement continuation route is blocked. Fixed Diamond Port duels match exactly: all 20 paired-round nine-field normalized bot-state trajectories are identical (40 physical duel rounds). This is not equivalence of every event, intent, bullet or hidden state. Random/fixed differences are a small compound-method diagnostic, not a causal isolation or variance-distribution estimate.

All requested poses and name→ID ordering match in every fixed round. Yet fixed melee pair differences are +5.24,+10.49,+13.50,−15.04pp. Earliest round 1 state divergence is Diamond Port heading in two pairs, and Keel gun/radar directions in two others. Earlier same-turn scan lists for Keel reverse target order in those latter pairs; its first-scan-only handler is order-sensitive. Diamond also has unseeded melee movement draws. These observations identify sensitivity and candidate randomness sources, not a single established cause.

Independent audit verifies all 32 recordings/CRC/contiguous ticks/endpoints/final scores, all 7,019 unchanged frozen inputs, 768 staged entrant files, 2,816 shared files, 96 wrappers/original scripts, 192 relay connections, 32 empty probes, 32 authoritative observer ends and 96 bot ending events. All observed input/output stream hashes/counts match; zero own skips, maximum logged loop 6 ms, minavailable memory 39%, all resource waves/owned cleanup pass. Loop timing excludes callbacks/termination.

Java 17 accepted-project 23 tests and 827-entry source/committed-JAR parity pass; 15 zero-engine harness contracts pass. The initial zero-game setup failure and two unused pre-game freezes are separately retained. Official external-server/initial-position support uses exact pinned 1.4.0 server bytes, valid issued session IDs and position-only handshake edits; gameplay payloads and bot binaries stay unchanged. No engine patch, deadline/heap/GC change, candidate tuning, retry/resume or historical pooling.

Fresh physical spend: 32 games / 240 completed rounds, zero partials. Cumulative development accounting: 3,270 complete plus 3 historical partial rounds, separate from the completed tournament. Original 0.2.1 stays byte-exact and earlier rejected studies stay rejected. Any new movement evaluation must use a genuinely NEW prospectively reviewed measurement design; this failed readiness verdict cannot be called a pass or silently waived.

All original recordings, transport streams/metadata, initial-state trajectories, source/JAR/support/runtime snapshots, build/fixture failures, resources/cleanup and independent diagnostic/auditor sources are retained in robocode-keelhelm-calibration/2026-11-06/aa-full-artifacts.tar.gz (599,237,893bytes), SHA256 6338e990f4ef390358f616a172c567c4c5207af3ed698433783eca9339f718d8. Original bulky per-tick hashes and telemetry stay in the archive; public data retain hashes and all official scores/initial poses.

[Prospective protocol](keelhelm-aa-protocol.md), [fresh continuation](keelhelm-aa-continuation.md), [stopped setup](keelhelm-aa-stopped-setup.md), [results](../../benchmarks/keelhelm-aa/fresh/results.json), [independent audit](../../benchmarks/keelhelm-aa/fresh/outcome-review.json), [attestation](../../benchmarks/keelhelm-aa/fresh/audit-attestation.json), [state divergence](../../benchmarks/keelhelm-aa/fresh/first-divergence-review.json), [event ordering](../../benchmarks/keelhelm-aa/fresh/event-order-review.json), [archive proof](../../benchmarks/keelhelm-aa/fresh/artifact-proof.json), [PR #133](https://github.com/stephanelee/robocode/pull/133), [issue #132](https://github.com/stephanelee/robocode/issues/132).

The public audit omits per-tick `startAudit` arrays and retains canonical digests; the immutable full archive contains the original independent audit and complete trajectories.
