# Frozen fixed-duel fresh confirmation

Registered before any matches. Freeze accepted7104b65d03298caceb010a5bfd085bda65092d17 / JARd047bf99325d2ad73aedd87eac6a4baa1677170afff0c86b948f03ea93390e1d. No gameplay, default-policy or model change. Prior development and PR38's two700-round comparisons are excluded.

Budget exactly10new20-round Retcon1v1 games,200rounds, in two separate runner jobs/batches of5games. Every game starts fresh bot processes. Layout tools/fixtures/frozen-duel-confirmation/games.json fixes IDs fc710-b1-g01..05 / fc710-b2-g01..05 and seeds20261601..05 /20261611..15, not previously used. Bot RNG is seeded; starting positions are unseeded/unpaired. Retcon28981c7a/API1.3.1, engine1.4.0,800x600, fixed-minimum-risk action0, absent model path. JDK25 matches after Java17 build/JAR parity; exactlyUbuntu26.04. No interactive game connection.

All three strict gates must pass over ALL10games: (1) equal-game mean official score share>50%; (2) official firstPlaces>100/200rounds; (3)>5/10games with official share>50%. Exactly50%,100wins or5games fails. No separate per-batch pass gate. Report both batches, all game shares, game SD/range and descriptive one-sided t(df9)=1.8331129326536335 lower bound, plus descriptive round-win Wilson interval. Intervals do not decide acceptance. Retcon adapts within each game; rounds are correlated and are not200independent trials. A pass concerns this frozen fixed policy/pinned opponent, not default-mode or general strength.

Require complete20-round official/strict telemetry evidence each game, zero AIS skipped turns, no model load/write, complete native Retcon exports, original source/JAR/opponent/runtime hashes. No private Retcon data enters control. Two batches are separately executed with isolated fresh processes; independence is an experimental design goal, not a guarantee about random starts. All registered games run unless an infrastructure/evidence failure makes the attempt invalid. Preserve every failure/partial; no automatic retry, replacement, rescue, favorable subset or changed criterion. Failed confirmation means reliable beating NOT confirmed; it does not retroactively reject accepted PR38's A/B improvement.

Execute only after exact independent source/plan/workflow preclear, one two-batch run, then aggregate once. Preserve raw200rounds/10officialgames/nativeexports/source/runtime/layout hashes in benchmarks, document results/decision/PR links in docs/improvements.md and independently review/merge. Default-policy experimentation is separate and cannot replace the frozen710 bundle. Remove temporary workflow before final evidence push.

## Measured result: registered confirmation passed

The single [run41560662466](https://github.com/stephanelee/robocode/actions/runs/41560662466) completed both five-game batches and the single aggregation at execution source `185312716afc268f06ee7fb385094629c0c261d2`. All200 fresh rounds,10official games and10native exports validated; AIS skipped turns were zero, no model loaded or written, and every runnable JAR matched frozen710/d047. No retry, replacement, selection or gameplay change occurred.

**Passed the three registered observation gates:** mean score share **55.903600%**, official round wins **107/200**, and **6/10** game shares above50%. Batch1 averaged56.608816% with55wins; batch2 averaged55.198383% with52wins. Both batches contained3/5games above50%.

| Game | Share | Official round wins |
| --- | ---: | ---: |
| fc710-b1-g01 | 38.9616% | 7 /20 |
| fc710-b1-g02 | 43.6118% | 9 /20 |
| fc710-b1-g03 | 60.9486% | 11 /20 |
| fc710-b1-g04 | 72.9565% | 15 /20 |
| fc710-b1-g05 | 66.5656% | 13 /20 |
| fc710-b2-g01 | 67.2276% | 13 /20 |
| fc710-b2-g02 | 52.5822% | 10 /20 |
| fc710-b2-g03 | 70.2156% | 13 /20 |
| fc710-b2-g04 | 40.6751% | 7 /20 |
| fc710-b2-g05 | 45.2915% | 9 /20 |

**Uncertainty remains substantial.** Game-share SD was13.151564percentage points. The descriptive one-sided95% t lower bound is48.279885%; the descriptive Wilson round-win interval is46.586647%–60.281436%. These cross50%; rounds within a game adapt and are not independent. This result satisfies the registered finite-sample fixed-policy criteria; it does **not** statistically establish superiority, prove general Robocode strength or establish default-mode benefit. Default-policy work remains a separate experiment.

[PR #40](https://github.com/stephanelee/robocode/pull/40), [durable raw data and exact source/provenance](../../benchmarks/frozen-duel-confirmation/README.md), [experiment index](../improvements.md). The accepted PR38 A/B improvement is unchanged.
