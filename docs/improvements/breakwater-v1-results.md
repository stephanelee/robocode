# Breakwater v1 pilot result

**Reject.** Breakwater lost all 20 rounds and scored 6 points against Diamond Port's 3,543. Its official score share was 0.169061708%. The pilot required strictly more than 20% and at least one official first place. All 20 rounds validated and Breakwater recorded zero skipped turns. Both performance gates failed, ending the experiment after one game. The conditional 60-round confirmation was never started. [Registration](breakwater-v1.md) · [Official scores](../../benchmarks/breakwater-v1/breakwater-v1-pilot/battle.json) · [PR #66](https://github.com/stephanelee/robocode/pull/66)

| Pilot measurement | Result |
| --- | --- |
| Game and seed | `breakwater-v1-pilot`, 531801 |
| Inferred enemy shots | 410 |
| Decisions with a shield solution | 1,370 |
| Confirmed shield shots | 351 |
| Shield bullets that intercepted an enemy bullet | 13 of 351 (3.70%), across 11 rounds |
| Confirmed attack shots and attack hits | 17 shots, 1 hit, 4.0 damage |
| Shield bullets hitting Diamond | 5 hits, 2.0 damage |
| Total damage dealt / taken in bot events | 6.0 / 2,010.1 |
| Energy spent firing | 52.1 (35.1 on shields, 17.0 on attacks) |
| Mean observed range | 469.56 px across 6,957 observations |
| Wall collisions / bot collisions / own skips | 0 / 0 / 0 |

The interception mechanism operated, but most shield bullets produced no confirmed interception. Breakwater spent 95.4% of its 368 shots on defense and earned just four offensive damage points from deliberate attacks. Its ledger matched 208 incoming rays for learning. The final cold-game endpoint retained eight signed residuals and 16 jointly resolved attack passages, with one virtual hit for each attack model. Solution counts are decisions and can describe repeated planning for the same wave. The round aggregates cannot distinguish mistakes in inferred launch time, origin or heading from the cost of braking or reserving the gun. They also cannot tell how much damage each successful intercept prevented. Official scoring caps and event-level raw damage have different semantics. This game supports rejection of the combined policy against the pinned Python port. [All round rows](../../benchmarks/breakwater-v1/breakwater-v1-pilot/metrics.jsonl) · [Final model](../../benchmarks/breakwater-v1/breakwater-v1-pilot/model.json)

## Provenance and review

The design was published in issue #53 and the draft PR before competition. An initial build-only CI job failed when a release-fixture source file was missing from the draft commit. The fixture was included, the independent review's same-tick collision and inactivity-decay issues were addressed, and the release passed 12 Java tests and three runner tests before play. The reviewer independently compiled the Java 17 code, matched all 30 candidate/shared class files, checked all 805 JAR entries, and verified the runtime pins, evidence rules and exact execution head. [Initial build failure](https://github.com/stephanelee/robocode/actions/runs/41707626608) · [Passing release build](https://github.com/stephanelee/robocode/actions/runs/41707972658)

The sole benchmark used execution head `fced7e405b64de4d059de2cfa36be36be34742d5`, source `b6b4bd87e3f5610f471cf76a9c99ebbd73524352` and Breakwater JAR SHA-256 `21163f89f97c0cce0c2c90072965e233cb9559a63898f4d8d9bc8954c2c9797e`. It ran on Ubuntu 26.04 x86_64 using Tank Royale 1.4.0, Temurin 25.0.5+8, Python 3.13.17 and the 12 locked wheels. Diamond Port 1.8.28 was acquired from `fduch-stranger/robocode-bot@67b68c22e41c18281a3fce9ef638e3c23a0b47cf`. The private endpoint SHA-256 is `a8caaae5a80ab6bc6386a517da1d093a35d3a04976daa5f159cddc7d2d118481`. No previous model was loaded. [Freeze manifest](../../tools/breakwater-freeze.json) · [Executed command and raw evidence](../../benchmarks/breakwater-v1/README.md) · [Measurement run](https://github.com/stephanelee/robocode/actions/runs/41708366019)

The independent post-run audit reproduced both failed gates from the official scores, matched the 20 rows to the endpoint, and checked all 104 staged opponent hashes and the launched JAR. Breakwater recorded its death in 14 round rows. Rounds 10, 11, 14, 17, 18 and 20 had neither terminal flag set. The official result gives all 20 first places to Diamond, and Breakwater's zero win flags reconcile as registered. The evidence does not identify why those six death flags were absent.
