# Robocode Elimination Cup — prospective protocol

Roster frozen from stephanelee/robocode master `1f6e6a38582838fe891330a0b3e7a7d9f1668279`: AISurvivor 1.2.1, Aegis 0.1.0, Breakwater 0.3.0, DiamondChallenger 0.6.1, Keel 0.1.0, KeelHelm 0.1.0, Meridian 0.6.0, Prism 0.2.0. Guests: pinned Retcon 0.2.0 and Diamond Port 1.8.28; expanded with Adaptive Prime 1.0 and official 1.4.0 release samples Walls, SpinBot and RamFire 1.0.

This is a descriptive tournament, with no gameplay changes, tuning, promotion or hypothesized improvement. Run locally on minimind, never GitHub Actions. Java 17 source build/tests and committed-JAR content parity precede JDK 25 battles. Tank Royale runner/server 1.4.0 and existing source locks apply. Native Python guests use an isolated macOS environment; record its actual Python/packages rather than claiming the Linux wheel runtime.

## Current user-approved format

Practice round 0: one opponent per bot, two 20-round games per pairing, reversed launch order:14 games/280 physical rounds. Opening draw uses Python random.Random(202611040).shuffle on the published roster order; adjacent entries pair. Practice has no tournament points, elimination, checkpoint carryover or competitive acceptance gate.

Four Swiss rounds: all 14 bots play each round, no elimination. Two 20-round games per pairing with reversed launch order; combined official totalScore determines 3 points to winner, 0 to loser, 1 each if exact draw. Rank by points, opponents' total points (Buchholz), then average pairing score share. Avoid repeat opponents and minimize point gaps. Round 1 draw must be recorded before play; no draw selected from measured practice strength.

Top 8 advance to double-elimination playoffs. Seed 1 v 8, 4 v 5, 2 v 7, 3 v 6. First playoff loss enters lower bracket; second eliminates. Lower-bracket winner must beat unbeaten finalist twice (final reset). Same two-game format. Playoff score tie breaks by round first-place wins, then one extra 20-round game, then a documented draw of lots if that game ties. Non-playoff exact ranking ties use a documented seeded draw after the declared numeric tiebreaks. Practice does not count as a Swiss rematch.

Every game starts cold in private directories. Normal within-game learning allowed. No separate training. Fixed bot RNG seed 20261104 for all seeds that bots support; engine starts/opponent randomness remain uncontrolled. Runtime and launch scheduling are not seeded. Execute sequentially. Preserve every official output/log and all failures; incomplete games get no automatic forfeit or replacement. An incomplete pairing remains unscored and prevents a final stage ranking until resolved with an explicit recorded ruling. Per-game deadline 900 seconds; outer process limit 960 seconds. Existing harness settings:1v1,100000 microsecond turn timeout, default arena/rules. No gameplay modification.

Publish the public ChatGPT Site before the first practice game. Include robots noindex, nofollow and robots.txt Disallow:/. Record source/JAR hashes, descriptors, lock pins, Java/Python versions, full score arrays, round wins and runtime failures. Public results contain tournament evidence, not credentials or environment secrets. Scores/rounds in adaptive games are dependent and these small samples do not establish broad competitive superiority. Update docs/improvements.md with actual budget/results and evidence/report links. No later round starts merely because its schedule exists; current request covers practice and Swiss round 1.

## Build observations before practice

All 8 Javaentries pass committed/rebuiltJAR content parity. Seven experimental Java 17 cleanbuild/test suites pass(158 tests). AISurvivor Java 17 build produces the matchingJAR, but its final 89 test run has 83 passes, 5 exact replaygolden failures and 1 skip on macOS/ARM64: first difference is gunTurnRate 2.0869864576583836 versus 2.086986457658327. The initial fixture-omission packaging failure and final numeric failures remain in evidence. No tests or gameplay were altered. These baseline test failures are a recorded verification limitation; they are not claimed as passingCI. Tournament execution proceeds with frozen main binaries and recorded source parity.

## Startup correction

The initial practice invocation staged DiamondChallenger under folder`bot`, so its launcher was not discovered. Official runner continued with one bot and produced 20 invalidAISurvivor-only rounds; the next invocation was stopped before an official result. Preserve `practice-startup-attempt/` and its manifest; exclude the 20 solo rounds from all practice/Swiss scores, retain them in physical budget. Correct only staging folder names to match launchers; restart the identical published practice draw. No valid competitive game was replaced. The canonical-folder assertion is added before Swiss execution.

## Swiss round 2 prospective pairing

User authorized round 2 after round 1 completed. Freeze 14 unchanged bots, 2 x 20 games/pair and 3/1/0 points. Compute a rematch-free perfect matching over current standings, minimizing lexicographically the sum of point differences, then sum of rank-index differences, then the ordered rank-index pair tuple. Ranking uses cumulativeSwisspoints, opponents current totalpoints, then mean pairing score share. The exact dynamic-programming matching uses all 14 bots once, has totalpointgap 3(minimum possible with seven 3-point and seven 0-point bots), and totalrankgap 9. Record full input standings and pairings before games in swiss-2-prospective-draw.json. Practice opponents do not count as Swiss rematches. No tuning, extra training, or bot/model carryover. User also requests a PR for runner/code and running instructions; raw game artifacts remain local.

## Swiss round 3 prospective amendment

User authorized round 3 after round 2 completed. Keep all 14 frozen entries and two cold 20-round games per pairing. Use the same exact rematch-free matching: total point gap, then rank-index gap, then deterministic pair tuple. Cumulative points/Buchholz/mean pairing share determine the input order. Record the draw before any game. The independently reviewed input freeze now also covers imported Python launch/validation helpers and opponent lock files, in addition to home inputs, shared sample libraries, Java harness, cached API and Java/Python runtimes. These are orchestration integrity guards; bot gameplay, parameters, model-reset policy and battle settings remain unchanged.

## Recording amendment before future stages

User requested visual replay recording after Swiss round 3’s final game had started. No running/completed game is restarted or retroactively represented as recorded. Enable official observer .battle.gz recording by default for subsequent stages, preserving bot sources, 20 roundgames,100000 microsecondturntimeout and 900 seconddeadline. Use a separately hashed recording harness, retain local files and SHA256 s, and report recording overhead as a runtime change. Public exports continue to contain score results only.

## Swiss round 4 prospective registration

User authorized round 4 when recording support is ready. All 14 bots remain frozen and each pairing plays 2 cold 20-roundgames with reversed launch order. Use the same exact rematch-free matching and cumulative points/Buchholz/mean-share order. Record all input standings, cost and draw before games. Enable observer recordings for all 14 games through the separately hashed adapter; user approved this runtime amendment before the round. Java 25 adapter compilation and 12 local tests pass; an independent reviewer found no blocker and confirmed recorder shutdown behavior from official 1.4.0 bytecode. The first actual approved replay will be checked for game start/end and 20-round coverage; no extra test battle is added. Full recordings remain local. Top 8 qualification follows round 4; no playoff starts automatically.

## Approved replay-video publication

After the Swiss stage completed, the user explicitly requested a video of the first replay, its public Site publication and a live score overlay. Render all20battle rounds of Swiss4/game1 at240simulationturns/sec with live energy and official cumulative scores replaced at round end. Final movie:5:42.7,H.264/yuv420p,1280x800,20fps. Verify final scores against battle.json and inspect sample frames. Only this derived MP4/poster/chapters were published; raw recordings, models, logs and invocation artifacts remain local. Independent review confirmed all70official outputs and final qualification, all14recordings, movie hash/chapters/final scores, and public export privacy. Browser GUI playback remains unverified under Work settings.
