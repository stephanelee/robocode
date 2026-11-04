# Meridian v3 committed gun plan: prospective local protocol

Written before implementation tests, 2026-11-04. Baseline is exact PR81 commit
809b880cdcee98d7946d05e6bcd30f623f604348. This is an isolated local candidate;
production AISurvivor and all committed candidate JARs/evidence stay unchanged.
No remote publication, Actions activation, server use or fresh battle in this milestone.
The relevant-issue publication required by AGENTS.md is deferred under the explicit
local-only instruction; it remains a dependency to resolve before a later battle stage.

## Hypothesis and mechanism

Instantaneous sign/bin/model reselection can outrun a 20-degree-per-turn gun.
On the first fresh cold-gun scan, hold the v3 selected signed offset from line of
sight and the selected power. Continue moving only its live line-of-sight anchor.
Do not relax the original alignment, freshness, heat, energy or reserve guards.
Keep virtual-ray grading and all movement, radar, target learning and power formulas.
Build CrossingGun's requested launch using the firing-decision scan and own pose,
held power, and next-turn launch timestamp. A pending accepted fire suppresses a
second request until confirmation or an expired event turn. Clear aim on scan loss,
insufficient energy, round transitions and confirmed fire; target death invalidates
its scan. Confirmation bookkeeping must survive a scan-loss/affordability invalidation.

## Predeclared tests and rejection conditions

Use installed actual Java17 (no substitute). Run Meridian tests/build, unchanged
v3 source/committed-JAR parity, and production tests/source-JAR parity without
replacing either committed JAR. Check every production tracked file is untouched.
Unit/integration cases: plan stability across sign/bin changes; first cold/fresh
selection; angular wrap; pending accepted/rejected/expired fire; confirmed fire;
round reset; scan loss/reacquisition; target disappearance; affordability including
zero/equal-power energy; unchanged reserve/alignment/freshness/heat guards; actual
firing-decision context even if another scan arrives before the callback.

Fixture: materialize commit 8989201545a09acebfaed9444aee263272bf01d8 path
benchmarks/meridian-prior/raw/meridian-prior-v3-control-diamond-screen/game-2026-11-04-00-21-43.battle.gz.
Required SHA256 f0c38e81e0965b29e5c71c79ec2f2eb7a1c18232dd275172d37b1dbf6251c2c1;
Git blob ce5beb5fd10d613d50dc7a87334b5a77d169d943.
Replay only own state, own visible scans and own fire confirmations into exact v3
source. No hidden enemy state/intent. Reproduce all 236 own shots in this game and
the round5 decision-turn144..3037 cold/fresh/affordable no-fire interval (2,894 turns).
Check expected shot count against recording rather than trusting this stated count.
The exact interval's count and metrics are assertions, not tuning inputs.

Servo-only fixture: replay v3 up to decision144, then hold its first plan; preserve
recorded legal geometry and change only gun heading using clamp(error,-20,20).
No simulated hits, damage, energy feedback, learning from hypothetical shots or
battle results. Before servo results, compute maximum consecutive line-of-sight
change D over the fixed interval and minimum original tolerance A. If D >=20 or
D>A, this simple bound is inapplicable and report failure rather than selecting a
new bound. Otherwise each capped step reduces error by at least20-D, and the first
uncapped step leaves next-turn error <=D<=A. Bound is
ceil(max(0,abs(initialError)-20)/(20-D))+1 servo steps, plus the initial decision.
Assert alignment by that bound, then stop. This validates liveness only.

## Proposed later bounded comparison (not authorized for this milestone)

After independent source/JAR/protocol review, freeze candidate hashes and exact
opponent/server pins from the existing v3 harness. Before games resolve issue
publication dependency (publish only with authorization or explicitly waive it).
No tuning from fresh outcomes. Proposed screen: 2 arms (exact v3 vs candidate),
3 opponents (pinned Diamond Port, Retcon, AISurvivor), 2 five-round games per arm
and opponent =60 rounds total, cold processes each game, fixed seeds 0 and1,
arm order baseline/candidate then candidate/baseline. Same opponent pins, arena,
server/launcher and timeouts; independent ports, never7654. Java17 builds; Java25
only for existing Retcon launcher. Archive every game and error, no replacements.
Stop on crash/skip/provenance mismatch and report the incomplete budget.
Report score share, wins, shots, damage dealt/taken, bullet expenditure and longest
eligible no-fire run by game. Review liveness benefit against accuracy/energy cost;
no automatic promotion or claim to beat Diamond from this screen. No confirmation
budget is implicitly authorized. Any later Actions jobs must be ubuntu-26.04.

Pre-test provenance correction: direct event counting finds244 own shots in this
specific recording (236 above referred to a different game in the supplied four-game
summary). Acceptance is exact agreement with all recorded events. Geometry-only
bound inputs, measured before servo execution: D=1.3525242047799964 degrees,
A=1.5255043413429084 degrees, 2,894 fresh legal scans. Thus the predeclared bound
is applicable; with an initial error between20 and38.647 degrees it permits2 steps.

## Local results and handoff (2026-11-04)

Candidate source frozen for independent review after24/24 Java17 Meridian tests
passed. The exact v3 archive passes10/10 original tests and matches all788 entries
of its unchanged committed JAR. Production source matches all822 entries of its
unchanged AISurvivor JAR. Candidate class entries match an independent direct
Java17 javac build (14 classes, class major version61).

Actual runtime: Homebrew OpenJDK17.0.21.0.3+0, Gradle8.14.3, macOS/minimind.
Candidate JAR (build output only):
`7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe`.
Unchanged v3 JAR: `5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9`.
Unchanged AIS JAR: `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800`.
Full source hashes are in `local-evidence/review-hashes.json`.

The exact baseline gun replay reproduced244/244 shots with no false/missing fire
decisions and matched power. Round5 decisions144..3037 reproduce2,894 eligible
no-fire turns,1,446 lateral sign flips, max absolute lateral speed1.474353609267,
and aim error7.918178484795..46.159732506279 degrees.
The committed plan's servo errors at144/145/146 are25.176071978936,
4.663895675504,-0.862297250866 degrees. It reaches the unchanged firing guard
in2 steps, meeting the predeclared2-step bound. No hypothetical hit, damage,
energy response, movement response or battle outcome is computed.

Broader unchanged production tests:89 total,83 passed,5 failed,1 skipped.
`MasterParityTest.masterMakesIdenticalDecisionsOnEveryGoldenTrace` and four
`TraceReplayTest` cases fail at the same exact floating-point golden comparison:
2.0869864576583836 expected vs2.086986457658327 actual. These production files
are untouched. This is consistent with runtime/platform math differences but its
cause was not proven; no tolerances were changed. The one skipped test has the
existing assumption "only on branches with the melee change". This is a remaining
validation limitation for independent review, not a passing full-repository suite.

First Gradle attempt failed before tests because sandbox local lock sockets were
blocked; the authorized offline retry succeeded. The first Meridian run had two
fixture mistakes (servo hydration inadvertently invoked a round reset; finishing
exception was checked after a second pending-suppressed request). Both fixes were
test-only. Logs/XML from that failure are retained. Before review, an additional
late-event guard prevents an old confirmation from consuming a newer pending
learning record; lifecycle tests cover that and the actual run-loop scan-loss path.
No tests were weakened after observed battle outcomes; no fresh battle was run.

Reproduction from repository root (use an installed actual Java17):

```sh
python3 tools/meridian_gun_plan_fixture.py
# Set JAVA_HOME to Java17; no auto-download or software installation.
./gradlew --offline --no-daemon -p experiments/Meridian test assemble
```

The fixture tool uses local Git objects only and fails if the pinned recording
commit is missing; the handoff also retains the byte-verified gzip locally.
The local host commands and full logs are in `local-evidence/run-gradle.sh`,
`candidate-build-test-review.log`, `v3-baseline-build-test.log`,
`production-build-test.log` and `test-summary.json`. The production parity build
redirects its JAR to `build/parity` with `production-parity.init.gradle`.
Committed candidate/production JARs were not replaced. Test-only baseline source
is exact v3 except class/constructor rename, verified by the fixture tool.

Decision: retain as an unpromoted local liveness candidate for independent review.
Competitive budget used:0 rounds. External issue/PR: none (publication deferred).
Independent review and a complete local runtime/opponent/game manifest remain
required before the proposed60-round comparison. The prior Ubuntu profile cannot
be assumed to match this Mac. The bounded comparison above is a proposal, not an
executable manifest or an authorization to launch before review. No strength claim.

## Independent review and pristine-baseline addendum

Independent review passes this isolated implementation/liveness milestone with no
source correctness blocker. A separate agent reran24/24 Java17 candidate tests,
verified exact244-shot replay,2,894-turn starvation and2-step convergence, source
hashes, baseline and production JAR parity, and candidate direct-compile parity.
See `local-evidence/independent-review.md` and `independent-retest.log`.

A separate untouched clone at exact809b880 reproduced the same five production
failures under the same Java17. The two targeted suites have8 total tests:2 pass,
5 fail,1 existing skip. Independent comparison confirms identical failure names,
exception types, full messages and skip reason, and43 relevant pristine tracked
files match HEAD. These are known baseline failures in this environment, not
evidence of a held-plan regression. The full suite remains non-green and the
platform/math root cause remains unproven; no code or tolerance was changed.
See `local-evidence/independent-pristine-comparison.json`,
`pristine-baseline-comparison.json` and `pristine-production-replay-test-tmp.log`.
The first pristine attempt stalled before tests reading a cached Gradle script;
a fresh temporary offline cache resolved that startup problem.

The source, tests, mechanism and predeclared fixture bound remain frozen. Only
these result/disposition paragraphs and the tracker row were updated after review.
Match readiness still requires a complete frozen executable manifest (engine starts
are uncontrolled by Meridian seeds), a verified Java25 Retcon runtime, and the
coordinator's resolution of the relevant-issue requirement. No fresh games,
publication, promotion or Diamond strength claim are included in this handoff.
