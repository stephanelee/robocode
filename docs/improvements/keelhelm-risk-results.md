# Isolated predicted-risk movement: results

**Reject the improvement claim; keep default KeelHelm 0.2.1 unchanged.** This fresh study completed all 88 serial minimind games / 520 rounds, with no partials or own skipped turns. Its movement hypothesis fails the prospectively registered melee-gain and exact-test gates. No automatic extension, retry, tuning, or promotion followed.

| Scenario | Control mean score share | Candidate mean score share | Difference |
| --- | ---: | ---: | ---: |
| melee | 32.5947% | 31.9126% | -0.6821 pp |
| Diamond Port | 54.6711% | 54.6711% | +0.0000 pp |
| Retcon | 94.3965% | 94.3965% | +0.0000 pp |
| Walls | 97.7224% | 97.7224% | +0.0000 pp |
| SpinBot | 100.0000% | 100.0000% | +0.0000 pp |

Melee first places fell from **76/160 to 71/160**. Equal-game survival share moved from 36.0920% to 35.2083% (−0.8836 pp), within its −3 pp guard. The source/fixture-backed effective movement counter records **10,131** differences, exceeding the ≥100 exposure requirement. This is not a reconstructed independent control trajectory. Every duel guard passes: all 12 pairs / 100 paired rounds have identical normalized nine-field bot-state trajectories (200 physical duel rounds) and identical official scores. This does not assert identity of every event, bullet or hidden state.

The balanced-assignment exact one-sided tail is **393,807,293 / 601,080,390 = 0.6551657641**, exceeding 0.05; measured melee gain is **−0.6821405516 pp**, below +2 pp. The test uses unlabelled first-slot minus second-slot contrasts and the original sealed uniform assignment. It is conditional on the sharp no-treatment-effect null and no cross-game interference assumptions; cold processes do not prove absence of host carryover. Native RNG and scan ordering remain uncontrolled. These results fail to establish improvement; they do not prove universal inferiority.

## Four-contestant melee totals

Each cohort covers 160 rounds over 32 fresh layouts. Aggregate scores below are descriptive; registered promotion uses equal-game mean score share.

| Bot | Control cohort score | Control wins | Candidate cohort score | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| KeelHelm | 29,247 | 76 | 28,670 | 71 |
| Diamond Port | 30,708 | 64 | 29,769 | 65 |
| Prism | 25,855 | 17 | 26,668 | 19 |
| Keel | 4,071 | 3 | 4,894 | 5 |

## Validation, failures and provenance

The independent raw auditor validates all 88 recordings/CRC, contiguous ticks, game/round endpoints, native setup/poses/IDs, official scores, own-skip absence, effective-exposure telemetry bounds, transport and teardown. All 7,010 frozen inputs, 2,784 staged entrant files, 6,336 shared guest files, eight sample-library copies and 304 wrapper/original scripts verify. All 568 connections, 88 empty probes, 88 authoritative observer ends and 304 bot ending events validate stream counts/hashes. Minimum observed available RAM is 37%; maximum logged own loop is 8 ms. Timing excludes callbacks/termination. All resource/cleanup guards pass.

Candidate Java 17 build has 24 passing tests and 827-entry / 791-file committed-JAR parity. An offline fixture compares 640 commands against exact old bit-8 movement and 640 disabled-feature commands against accepted parent movement; all 1,280 match. Twenty-two pinned-Python custody/geometry/reducer/transport contracts pass. An earlier wrong-default-Python import failure and two zero-engine preflight failures are retained, with zero physical games/rounds; no reroll of assignment rank 116078539. The initial monolithic preparation source and original assignment remain archived. None of these offline fixtures establishes competitive strength.

This is a new study after A/A failed its own continuation gate; that failure remains unchanged. The earlier 0.4.0 risk rejection also remains unchanged, and historical data were never pooled or relabelled. Candidate 0.7.0 isolates the exact risk intervention on accepted 0.2.1, with no gun/targeting/radar/power/opponent or engine-deadline change. Normal baseline project remains all 40 files unchanged.

Frozen source: `d2dc1922e5fd6ca07648d2de15505a27db2e606a`; candidate JAR: `1ff7d447d458be9f2a332d45892d1ef838018ddd50742f008abb62a558b4ea1c`; control JAR: `19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b`; freeze: `e862dc92c6586ae3c779a41b5be0337cc21e4dce59ae5d4450e05d7a7dd8dfb9`. Opponent pins/entry identities, native runner/server hashes, Java 25/Python runtime and all geometry/orders/seeds are in the sealed registration and full archive. Builds use Java 17; gameplay and Retcon source launch use pinned Java 25 after builds.

Fresh physical cost: 88 games / 520 completed rounds / zero partials. Combined A/A plus movement cost: 120 games / 760 rounds. Cumulative development accounting: **3,790 completed plus three historical partial rounds**, separate from the completed tournament. Offline aiming reanalysis spends zero new games/rounds.

Durable full archive: `robocode-keelhelm-calibration/2026-11-06/risk-full-artifacts.tar.gz` (1,220,647,019 bytes), SHA256 `1c0f6ed48aa57cf60d57d5e2a88da1143b7a3d3f585eeb14cee32303fa421141`. Includes every original game/staging/recording, telemetry/start trajectories, resources/cleanup, source/JAR/runner/server/opponent/runtime input bytes (all 7,010), independent auditor/kernel/attestation sources and acquisition logs. Public copies omit bulky telemetry/trajectories while retaining official scores, initial poses, hashes and all gates.

[Prospective protocol](keelhelm-risk-protocol.md), [sealed registration](../../benchmarks/keelhelm-risk/registration), [results](../../benchmarks/keelhelm-risk/results/results.json), [independent audit](../../benchmarks/keelhelm-risk/results/outcome-review.json), [attestation](../../benchmarks/keelhelm-risk/results/audit-attestation.json), [artifact proof](../../benchmarks/keelhelm-risk/results/artifact-proof.json), [A/A failure](keelhelm-aa-results.md), [PR #133](https://github.com/stephanelee/robocode/pull/133), [issue #134](https://github.com/stephanelee/robocode/issues/134).
