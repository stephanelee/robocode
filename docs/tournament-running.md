# Running the minimind invitational

The runner reproduces this event's frozen 14-bot roster. Run it locally; do not dispatch GitHub Actions. Each command executes one requested stage and stops. Practice is unranked. Swiss pairings award 3 points for a win, 1 each for an exact combined-score draw, and 0 for a loss. A pairing contains two cold 20-round games, with launch order reversed in game two.

This PR automates practice and Swiss rounds 1–4. The agreed top-eight double-elimination playoff format is documented in the protocol; its bracket execution is not yet automated by this runner. Do not use a Swiss stage as a substitute for a playoff. No bot is eliminated during the four Swiss rounds.

## Prepare an isolated source checkout

Keep the runner checkout separate from the frozen bot source. The roster is AISurvivor, Aegis, Breakwater, DiamondChallenger, Keel, KeelHelm, Meridian and Prism from `1f6e6a38582838fe891330a0b3e7a7d9f1668279`, plus pinned Retcon, Diamond Port, Adaptive Prime and the official Walls, SpinBot and RamFire samples.

```sh
CUP_SOURCE=/private/tmp/robocode-cup-source
CUP_OUTPUT="$PWD/benchmarks/my-minimind-cup"
CUP_PYTHON=/private/tmp/robocode-cup-python
CUP_SITE=/private/tmp/robocode-cup-website

git fetch origin 1f6e6a38582838fe891330a0b3e7a7d9f1668279
git worktree add --detach "$CUP_SOURCE" 1f6e6a38582838fe891330a0b3e7a7d9f1668279
python3.12 -m venv "$CUP_PYTHON"
"$CUP_PYTHON/bin/pip" install -r tools/tournament-macos-requirements.txt
cp -R website "$CUP_SITE"
```

A full source checkout includes the committed replay and characterization fixtures used by the Java tests. The Python requirements preserve the actual macOS/CPython 3.12 environment used here, including Bot API 1.3.1. They are not the Linux/CPython 3.13 wheel lock; report the actual runtime rather than claiming those environments are interchangeable.

Set paths to JDK 17 and JDK 25, then prepare the entries. The helper starts no games. It verifies the source commit, source pins and asset hashes, builds/tests with Java 17, checks every committed/rebuilt JAR, and only then selects Java 25 for Retcon's source launcher.

```sh
CUP_JAVA17=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
CUP_JAVA25=/path/to/jdk-25/Contents/Home/bin/java

python3 tools/prepare_tournament.py \
  --source "$CUP_SOURCE" --output "$CUP_OUTPUT" \
  --java17-home "$CUP_JAVA17" --java25 "$CUP_JAVA25" \
  --python "$CUP_PYTHON/bin/python"
```

Preparation requires an empty output directory and a fresh source checkout without `guest-source/` or `samples/`. It preserves build logs, test results, parity results and runtime/package provenance. It also writes prepared input snapshots. Every stage compares entry bytes, shared sample libraries, the battle harness, cached Retcon API and Java/Python runtime snapshots against preparation and the immutable first-stage inputs; a change stops before any game and never overwrites the original manifest. It fails on a bad pin or JAR mismatch. An existing Gradle executable can be supplied with `--gradle /absolute/path/to/gradle`.

On this Mac, five AISurvivor exact replay tests differed at tiny floating-point values; 83 tests passed and one was skipped. The first observed gun-turn value was `2.0869864576583836` versus `2.086986457658327`. The underlying platform/JDK cause has not been independently established. A normal preparation stops on that failure. `--allow-known-macos-replay-differences` acknowledges only the exact five recorded test cases/signature on macOS ARM64 and retains their failures. It never labels the tests as passed, permits unrelated failures, or relaxes JAR parity. All seven experimental bot build/test suites passed in the recorded run.

## Run a stage

Use the same source, runtime, output and site paths throughout an edition. Games run sequentially with private bot directories, absent model paths and normal within-game learning. Nothing loads a previous game's model. Supported bot seed overrides use `20261104`; arena starts and scheduling remain uncontrolled.

```sh
python3 tools/tournament.py \
  --source "$CUP_SOURCE" --output "$CUP_OUTPUT" \
  --java "$CUP_JAVA25" --python "$CUP_PYTHON/bin/python" \
  --site "$CUP_SITE" --stage practice

python3 tools/tournament.py \
  --source "$CUP_SOURCE" --output "$CUP_OUTPUT" \
  --java "$CUP_JAVA25" --python "$CUP_PYTHON/bin/python" \
  --site "$CUP_SITE" --stage swiss-1
```

Then explicitly run `swiss-2`, `swiss-3` and `swiss-4` as authorized. Each stage contains seven pairings, fourteen games and 280 battle rounds. Later rounds require every prior Swiss round to be complete. They use an exact rematch-free perfect matching, minimizing total point difference, then total rank-index difference, then the ordered rank-index pair tuple. The input standings and draw are written before the first game. Practice opponents do not count as Swiss rematches.

Standings use cumulative Swiss points, then current opponents' total Swiss points (Buchholz), then mean pairing score share. Each pairing's share uses its two-game combined score. Displayed standings remain provisional until all four Swiss rounds are complete. Exact qualification ties require the protocol's recorded draw of lots; no playoff qualification is performed automatically.

`--site` is optional. When supplied, it updates the copied static website's `data.json` after games and pairings. It reconstructs cumulative results from the canonical stage reports in the output directory, so an export added later or a restored website template cannot drop previous points. It exports profiles and official scores only. Models, telemetry, logs and invocation environments stay in the local evidence directory. The website preserves earlier rounds, highlights guests and completed pairing winners, and checks for new published snapshots every 15 seconds. A local export is not a hosted deployment: publish the static files through the Sites workflow, reusing the existing Site identity and audience. Hosting credentials are never stored in this repository.

## Failures, retention and continuation

A game has a 900-second runner deadline and a 960-second outer process deadline. Every official result must contain exactly two expected participants and twenty rounds. Incomplete games remain unscored; there are no automatic forfeits or favorable reruns. Skipped turns are not silently treated as zero, and telemetry schemas differ across bots.

Keep the complete output directory, including `*/pNN-gN/bots/`, models, JSONL telemetry, runner logs, invocation records, raw official results and initial failed attempts. Archive it outside itself and store a SHA256 checksum. Never overwrite an archive whose hash has already been recorded. The public repository/site contain score-only reports; full game artifacts stay local unless the user separately authorizes their publication.

A narrowly bounded `--resume-unplayed` option preserves the original draw and completed results. It refuses any attempted game whose status is incomplete, and any existing unplayed game directory. Before using it, record the interruption/ruling and retain an unlaunched staging directory under a different evidence path. It is not a retry mechanism for a failed battle. The recorded Swiss-round-1 continuation corrected a source-sample preflight check, consumed zero additional rounds and reran no completed game.

## Checks without battles

```sh
python3 -m unittest discover -s tools -p 'test_tournament.py' -v
python3 -m py_compile tools/tournament.py tools/prepare_tournament.py
python3 -m http.server 8766 --directory website
```

The twelve unit tests cover draw coverage, rematch exclusion, minimum point-gap matching, prerequisites, cumulative standings, idempotent score exports, canonical history reconstruction, immutable input guards, artifact exclusion recording integrity and the narrow replay-test exception. They do not establish bot strength or replace real game evidence. This event used Java 17 for builds and Temurin 25.0.5+8 for battles. No GitHub Actions jobs are needed to reproduce it.

## Replay recordings for future games

At the user’s request, future stages record by default through the new `tools/TournamentBattle.java` adapter. Its battle settings, participant launch behavior and completion barriers match the frozen original harness; the added observer recorder uses the official1.4.0API. A private Java system property carries the recording path to the runner, without feeding observer frames or the path to bot processes. Recording overhead is a runtime change and is prospectively documented separately from the unrecorded rounds.

Each completed game retains `pNN-gN/recordings/game-*.battle.gz`, with byte size and SHA256 in its private game record. The wrapper checks gzip integrity. In the Tank Royale desktop GUI, select **Battle → Replay from File** and open that file. These are replay data files, not videos. Full record-frame semantics and visual playback must be verified on an actual recorded game; no extra battle was run merely to test recording. The already completed practice and Swiss rounds1–3 did not record visual replays and are not rerun.

`--no-recordings` explicitly preserves the earlier behavior. A continuation must keep the same recording policy as its original stage. Recording harness source gets an immutable input snapshot; new preparations also pin it. Recordings remain local with the full game artifacts and are excluded from public score exports.
