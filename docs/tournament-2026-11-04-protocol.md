# Robocode Elimination Cup — prospective protocol

Roster frozen from stephanelee/robocode master1f6e6a38582838fe891330a0b3e7a7d9f1668279: AISurvivor1.2.1, Aegis0.1.0, Breakwater0.3.0, DiamondChallenger0.6.1, Keel0.1.0, KeelHelm0.1.0, Meridian0.6.0, Prism0.2.0. Guests: pinned Retcon0.2.0 and Diamond Port1.8.28; expanded with Adaptive Prime1.0 and official1.4.0 release samples Walls, SpinBot and RamFire1.0.

This is a descriptive tournament, with no gameplay changes, tuning, promotion or hypothesized improvement. Run locally on minimind, never GitHub Actions. Java17 source build/tests and committed-JAR content parity precede JDK25 battles. Tank Royale runner/server1.4.0 and existing source locks apply. Native Python guests use an isolated macOS environment; record its actual Python/packages rather than claiming the Linux wheel runtime.

## Current user-approved format

Practice round0: one opponent per bot, two20-round games per pairing, reversed launch order:14games/280physical rounds. Opening draw uses Python random.Random(202611040).shuffle on the published roster order; adjacent entries pair. Practice has no tournament points, elimination, checkpoint carryover or competitive acceptance gate.

Four Swiss rounds: all14bots play each round, no elimination. Two20-round games per pairing with reversed launch order; combined official totalScore determines3points to winner,0to loser,1each if exact draw. Rank by points, opponents' total points (Buchholz), then average pairing score share. Avoid repeat opponents and minimize point gaps. Round1 draw must be recorded before play; no draw selected from measured practice strength.

Top8 advance to double-elimination playoffs. Seed1v8,4v5,2v7,3v6. First playoff loss enters lower bracket; second eliminates. Lower-bracket winner must beat unbeaten finalist twice (final reset). Same two-game format. Playoff score tie breaks by round first-place wins, then one extra20-round game, then a documented draw of lots if that game ties. Non-playoff exact ranking ties use a documented seeded draw after the declared numeric tiebreaks. Practice does not count as a Swiss rematch.

Every game starts cold in private directories. Normal within-game learning allowed. No separate training. Fixed bot RNG seed20261104 for all seeds that bots support; engine starts/opponent randomness remain uncontrolled. Runtime and launch scheduling are not seeded. Execute sequentially. Preserve every official output/log and all failures; incomplete games get no automatic forfeit or replacement. An incomplete pairing remains unscored and prevents a final stage ranking until resolved with an explicit recorded ruling. Per-game deadline900seconds; outer process limit960seconds. Existing harness settings:1v1,100000microsecond turn timeout, default arena/rules. No gameplay modification.

Publish the public ChatGPT Site before the first practice game. Include robots noindex,nofollow and robots.txt Disallow:/. Record source/JAR hashes, descriptors, lock pins, Java/Python versions, full score arrays, round wins and runtime failures. Public results contain tournament evidence, not credentials or environment secrets. Scores/rounds in adaptive games are dependent and these small samples do not establish broad competitive superiority. Update docs/improvements.md with actual budget/results and evidence/report links. No later round starts merely because its schedule exists; current request covers practice and Swiss round1.

## Build observations before practice

All8Javaentries pass committed/rebuiltJAR content parity. Seven experimental Java17cleanbuild/test suites pass(158tests). AISurvivor Java17build produces the matchingJAR, but its final89test run has83passes,5exact replaygolden failures and1skip on macOS/ARM64: first difference is gunTurnRate2.0869864576583836 versus2.086986457658327. The initial fixture-omission packaging failure and final numeric failures remain in evidence. No tests or gameplay were altered. These baseline test failures are a recorded verification limitation; they are not claimed as passingCI. Tournament execution proceeds with frozen main binaries and recorded source parity.

## Startup correction

The initial practice invocation staged DiamondChallenger under folder`bot`, so its launcher was not discovered. Official runner continued with one bot and produced20invalidAISurvivor-only rounds; the next invocation was stopped before an official result. Preserve `practice-startup-attempt/` and its manifest; exclude the20solo rounds from all practice/Swiss scores, retain them in physical budget. Correct only staging folder names to match launchers; restart the identical published practice draw. No valid competitive game was replaced. The canonical-folder assertion is added before Swiss execution.

## Swiss round2 prospective pairing

User authorized round2 after round1 completed. Freeze14unchanged bots,2x20games/pair and3/1/0points. Compute a rematch-free perfect matching over current standings, minimizing lexicographically the sum of point differences, then sum of rank-index differences, then the ordered rank-index pair tuple. Ranking uses cumulativeSwisspoints, opponents current totalpoints, then mean pairing score share. The exact dynamic-programming matching uses all14bots once, has totalpointgap3(minimum possible with seven3-point and seven0-point bots), and totalrankgap9. Record full input standings and pairings before games in swiss-2-prospective-draw.json. Practice opponents do not count as Swiss rematches. No tuning, extra training, or bot/model carryover. User also requests a PR for runner/code and running instructions; raw game artifacts remain local.

## Swiss round 3 prospective amendment

User authorized round 3 after round 2 completed. Keep all 14 frozen entries and two cold 20-round games per pairing. Use the same exact rematch-free matching: total point gap, then rank-index gap, then deterministic pair tuple. Cumulative points/Buchholz/mean pairing share determine the input order. Record the draw before any game. The independently reviewed input freeze now also covers imported Python launch/validation helpers and opponent lock files, in addition to home inputs, shared sample libraries, Java harness, cached API and Java/Python runtimes. These are orchestration integrity guards; bot gameplay, parameters, model-reset policy and battle settings remain unchanged.

## Recording amendment before future stages

User requested visual replay recording after Swiss round3’s final game had started. No running/completed game is restarted or retroactively represented as recorded. Enable official observer .battle.gz recording by default for subsequent stages, preserving bot sources,20roundgames,100000microsecondturntimeout and900seconddeadline. Use a separately hashed recording harness, retain local files and SHA256s, and report recording overhead as a runtime change. Public exports continue to contain score results only.
