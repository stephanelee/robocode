# Prism profile2 screen: rejected

Sole registered [run41892189167](https://github.com/stephanelee/robocode/actions/runs/41892189167), [PR112](https://github.com/stephanelee/robocode/pull/112), [issue111](https://github.com/stephanelee/robocode/issues/111): all13 official game endpoints report200 rounds. The frozen decision is **reject**, solely Diamond Port30.339588% mean score share not exceeding40%. No confirmation, replacement, rerun or retune. Prism remains a standalone experimental reference; no production AIS promotion or competitive-superiority claim.

| Opponent | Games/rounds | Mean game score share | Prism official first places | Confirmed shots/hits | SURFER aims |
|---|---:|---:|---:|---:|---:|
| Python Diamond Port1.8.28 |3/60|30.339588%|11/60|3485/275|3595|
| Retcon0.2.0 |3/60|99.335769%|60/60|2850/418|4488|
| Python Adaptive Prime1.0 |3/60|63.434299%|35/60|2823/403|2067|
| Official Spin Bot1.0 |2/10|100%|10/10|166/90|91|
| Official Walls1.0 |2/10|100%|10/10|147/87|213|

All registered engineering gates passed: zero Prism recorded skips, collisions and subsystem failures across145767 reported movement decisions. All200 per-round telemetry snapshots and13 cumulative game endpoints are present. First places are the official field and can include ties. `gun.aim.surfer` counts selected aims, not confirmed shots from that gun. It confirms activation, not causality or isolated improvement. Diamond pooled hit rate275/3485=7.8910%. The historical v1 Diamond28.608206% versus v2 30.339588% uses different seeds and unseeded positions: descriptive only, no paired A/B, pooling, significance or power claim. Retcon/Adaptive success does not rescue the failed Diamond gate.

## Original preservation and zero-game reproduction

* `raw/` retains all55 original ZIP payloads:13 attempt/result/log/telemetry groups plus decision, receipt and runtime. Original artifact13353639321, ZIP56600bytes SHA256f58a4f6d000c9f87a6276df6b05b8235a43285af8910296cc355b1874ee88b7d.
* `source/` and source-provenance.json preserve exact Git execution inputs from cc6886d86b43f4b955870dd6a6d05877f9cad226, including the frozen approval plan, workflow, Prism source/tests/build settings and generic runtime loaders/locks. CandidateJAR f9c75a86e21aafa3dd889b626a2d9c4e321d69d4d3917e0483ea324360e27263/source-tree3de9c39c42d36d08e37ee78a86a99b6e34488d4f remain reachable through the permanent execution receipt.
* `audit-summary.json`, cohort-totals.json, manifest.json, original job/run/artifact metadata and job.log support exact reproduction. Manifest records every retained byte count/hash; no raw files edited.

Run `python -m unittest discover -s tools -p test_prism_v2_evidence.py -v`. It hashes every archived file and reconstructs official/cumulative metrics and all unchanged gates with the executed gate function. It launches no bots. Source snapshot is provenance; full Java build/trace fixtures can be restored from the exact Git execution commit.

## Scope and limitations

The original author supplied the single confirmed-wave gun. A separately disclosed user-authorized readiness continuation fixed the three reviewed blockers (label/exact-head route; pending/recorded-approval test; surfer payload accounting/eviction) and prospective invalid-only acquisition stopping. Java17 clean build42JUnit/855-entry parity and22harness tests passed; ordinary required CI passed before play. These maintenance fixes are additional provenance, not an exclusive-original-author claim. No opponent implementation was consulted for those fixes. No strategy, tuning, roster or criterion changed after outcomes.

Independent exact-head review6639301636 binds e5adfc4. Executioncc6886d changes only approvedHead/approvedReview and passed the actual semantic guard. One label triggered attempt1; ubuntu26.04, Temurin17 build/parity then25.0.5+8, Python3.13.17. Permanent receipt refs/tags/prism-screen-v2-f9c75a86e21aafa3 points atcc6886d and is never deleted. No invalid rows/unplayed IDs. The spent workflow is retired; exact executed bytes remain archived. GUI7654 and the independent Keel owner were untouched.

No observer .battle.gz or opponent-private skip telemetry was retained by this registered harness. No official per-round scores are reconstructed from cumulative Prism counters. Zero skips refers to Prism private telemetry only. Latency fields are bounded rolling4096-sample summaries, not a lifetime CPU/deadline certificate. Cold setup is supported by13 fresh staging paths/process launches/seeds; neither Prism nor this screen persists model checkpoints, so there are no checkpoint endpoint claims.
