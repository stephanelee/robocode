# KeelHelm accuracy-power initial comparison — invalid timing evidence

The original parallel study stopped permanently at its second stage. No power-improvement decision is supported and stages 3/4 were not launched. The separate first-stage adaptive-gun rejection remains valid.

The unchanged control 0.2.1 skipped turns 1, 2, 3 at the start of Round 13. Its CRC-valid recording contains 12 completed rounds, then an aborted Round 13 through turn 107 and no GameEnded event. Its candidate sibling completed 20 rounds. Actual spend: 32 completed physical rounds plus one partial round, in two started games. No replacement or resume occurred. Eighteen scheduled games in this stage remain unplayed.

Resource monitoring passed: minimum memory availability 40%, no new Swapouts, no recorded resource threshold breach. These observations do not establish a CPU or memory cause for the timing failure. The candidate's complete individual score is retained but is excluded from any accepted comparison or pooled result.

Cleanup returned EPERM and masked the primary skipped-turn failure; cleanup.json was not written. The retained evidence does not prove whether SIGTERM or SIGKILL returned the permission error. A separate read-only inspection found no surviving processes in the registered groups 71279/71280; no unrelated process was signalled. Both the failure and teardown defect require explicit treatment before another acquisition plan.

The original study's follow-ups consumed 292 completed rounds plus one partial round (stage 1: 260; stage 2: 32+partial). Including the historical 260 development rounds, project accounting is 552 completed rounds plus one partial round. All source/JAR/runtime/registration bytes still verified independently at this stop. The original 80-game / 1,040-round plan is not being silently resumed.

Any continuation needs a separately prospective load plan, fresh complete populations, unchanged gameplay constants/JAR, independent exact-freeze approval and no pooling of this failed attempt. The user's concurrency preference is pending clarification after this new timing evidence. No further games are authorized by the stopped study.

All raw recordings, staged copies, masks, telemetry, resource samples, invocations, frozen snapshots and independent failure audit are retained. [Failure results](../../benchmarks/keelhelm-followups/stage-2-original-invalid/results.json), [independent audit](../../benchmarks/keelhelm-followups/stage-2-original-invalid/independent-audit.json), [archive proof](../../benchmarks/keelhelm-followups/stage-2-original-invalid/artifact-proof.json). Local archive: local-evidence/keelhelm-followups/stage-2-original-invalid/full-artifacts.tar.gz.

[Original protocol](keelhelm-melee-followups-protocol.md), [issue125](https://github.com/stephanelee/robocode/issues/125), [PR126](https://github.com/stephanelee/robocode/pull/126).
