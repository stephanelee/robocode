# KeelHelm melee v1 — invalid comparison, retained evidence

Decision: reject the improvement claim and stop the comparison. Two games completed physically: one valid control and one invalid candidate, 40 rounds total. Fourteen games remained unplayed. The preceding folder-collision setup launched zero games. No v1 continuation or replacement game occurred.

| Attempt | KeelHelm version | Official engine score | Disposition |
| --- | --- | ---: | --- |
| melee-1-control | 0.1.0 | 1,807 | Valid; 15.214% score share, 2 first places |
| melee-1-candidate | 0.2.0 | 392 | Invalid gameplay lifecycle/telemetry; not accepted competitive evidence |

The candidate's onRoundStarted called getEnemyCount before the first tick. The pinned API's getter calls getCurrentTickOrThrow, so the callback threw before resetting melee state/counters and writing its start marker. All twenty start markers were absent; melee counters accumulated across rounds. The candidate recording and engine results are complete and retained, but the required telemetry failed. This is a gameplay correctness failure, not a harmless evidence-format mismatch. Independent audit rejected it. A real unstarted-Bot callback test reproduces the BotException without launching a server or spending rounds.

The only valid control game is descriptive, not a baseline mean or evidence of candidate improvement. No comparison gates passed and no v1 improvement is claimed. The separate [v2 lifecycle protocol](keelhelm-melee-v2-protocol.md) prospectively registers the API-lifecycle repair and a completely fresh comparison; no v1 game is pooled.

Source/protocol: b20caea (candidate gameplay unchanged from85830f6), frozen v1 registration SHA256 3ed83354c2eb4c5867c54351aa5d79ad9a56aaba31714544fb0f6a41253f446e. Earlier zero-game setup freeze:5083df55da3227fd634b32991a319e69822e5fc3a34eea219d4366039388870d. Candidate/control source and JAR bytes, original protocols/harnesses, runtime manifests, isolated bot copies, recordings, official results, telemetry, staging failure, lifecycle reproduction and independent audit are retained privately. The final source/runtime audit verified all7,005 registered hashes before v2 edits.

Private evidence directories on minimind:

- /private/tmp/robocode-keelhelm-melee-evidence (zero-game staging failure).
- /private/tmp/robocode-keelhelm-melee-evidence-run2 (40-round invalid v1 comparison).

[Sanitized invalid-attempt record](../../benchmarks/keelhelm-melee-v1/invalid.json), [original protocol](keelhelm-melee-v1-protocol.md), [issue123](https://github.com/stephanelee/robocode/issues/123), [PR124](https://github.com/stephanelee/robocode/pull/124). Full artifact archive and hashes accompany the final v2 report.
