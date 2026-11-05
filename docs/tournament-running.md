# Running the minimind invitational

The runner reproduces this event's frozen 14-bot roster. Run it locally; do not dispatch GitHub Actions. Each command executes one requested stage and stops. Practice is unranked. Swiss pairings award 3 points for a win, 1 each for an exact combined-score draw, and 0 for a loss. A pairing contains two cold 20-round games, with launch order reversed in game two.

The runner automates practice, Swiss rounds 1–4 and the fixed top-eight double-elimination bracket. Each invocation executes one explicit stage and stops. No bot is eliminated during Swiss; a second playoff match loss eliminates it. See the [prospective playoff registration](tournament-playoffs-2026-11-05.md) for the fixed feeds and tie rules. The current user request authorizes opening wave 1 only.

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
"$CUP_PYTHON/bin/pip" install -r tournament/scripts/tournament-macos-requirements.txt
cp -R tournament/website "$CUP_SITE"
```

A full source checkout includes the committed replay and characterization fixtures used by the Java tests. The Python requirements preserve the actual macOS/CPython 3.12 environment used here, including Bot API 1.3.1. They are not the Linux/CPython 3.13 wheel lock; report the actual runtime rather than claiming those environments are interchangeable.

Set paths to JDK 17 and JDK 25, then prepare the entries. The helper starts no games. It verifies the source commit, source pins and asset hashes, builds/tests with Java 17, checks every committed/rebuilt JAR, and only then selects Java 25 for Retcon's source launcher.

```sh
CUP_JAVA17=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
CUP_JAVA25=/path/to/jdk-25/Contents/Home/bin/java

python3 tournament/scripts/prepare_tournament.py \
  --source "$CUP_SOURCE" --output "$CUP_OUTPUT" \
  --java17-home "$CUP_JAVA17" --java25 "$CUP_JAVA25" \
  --python "$CUP_PYTHON/bin/python"
```

Preparation requires an empty output directory and a fresh source checkout without `guest-source/` or `samples/`. It preserves build logs, test results, parity results and runtime/package provenance. It also writes prepared input snapshots. Every stage compares entry bytes, shared sample libraries, the battle harness, cached Retcon API and Java/Python runtime snapshots against preparation and the immutable first-stage inputs; a change stops before any game and never overwrites the original manifest. It fails on a bad pin or JAR mismatch. An existing Gradle executable can be supplied with `--gradle /absolute/path/to/gradle`.

On this Mac, five AISurvivor exact replay tests differed at tiny floating-point values; 83 tests passed and one was skipped. The first observed gun-turn value was `2.0869864576583836` versus `2.086986457658327`. The underlying platform/JDK cause has not been independently established. A normal preparation stops on that failure. `--allow-known-macos-replay-differences` acknowledges only the exact five recorded test cases/signature on macOS ARM64 and retains their failures. It never labels the tests as passed, permits unrelated failures, or relaxes JAR parity. All seven experimental bot build/test suites passed in the recorded run.

## Run a stage

Use the same source, runtime, output and site paths throughout an edition. Games run sequentially with private bot directories, absent model paths and normal within-game learning. Nothing loads a previous game's model. Supported bot seed overrides use `20261104`; arena starts and scheduling remain uncontrolled.

```sh
python3 tournament/scripts/tournament.py \
  --source "$CUP_SOURCE" --output "$CUP_OUTPUT" \
  --java "$CUP_JAVA25" --python "$CUP_PYTHON/bin/python" \
  --site "$CUP_SITE" --stage practice

python3 tournament/scripts/tournament.py \
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
python3 -m unittest discover -s tournament/scripts -p 'test_*.py' -v
python3 -m py_compile tournament/scripts/tournament.py tournament/scripts/prepare_tournament.py
python3 -m http.server 8766 --directory tournament/website
```

The tournament and starter tests cover draw coverage, rematch exclusion, minimum point-gap matching, prerequisites, cumulative standings, idempotent score exports, canonical history reconstruction, immutable input guards, artifact exclusion recording integrity and the narrow replay-test exception. They do not establish bot strength or replace real game evidence. This event used Java 17 for builds and Temurin 25.0.5+8 for battles. No GitHub Actions jobs are needed to reproduce it.

## Replay recordings for future games

At the user’s request, future stages record by default through the new `tournament/scripts/TournamentBattle.java` adapter. Its battle settings, participant launch behavior and completion barriers match the frozen original harness; the added observer recorder uses the official 1.4.0 API. A private Java system property carries the recording path to the runner, without feeding observer frames or the path to bot processes. Recording overhead is a runtime change and is prospectively documented separately from the unrecorded rounds.

Each completed game retains `pNN-gN/recordings/game-*.battle.gz`, with byte size and SHA256 in its private game record. The wrapper checks gzip integrity. In the Tank Royale desktop GUI, select **Battle → Replay from File** and open that file. These are replay data files, not videos. Full record-frame semantics and visual playback must be verified on an actual recorded game; no extra battle was run merely to test recording. The already completed practice and Swiss rounds 1–3 did not record visual replays and are not rerun.

`--no-recordings` explicitly preserves the earlier behavior. A continuation must keep the same recording policy as its original stage. Recording harness source gets an immutable input snapshot; new preparations also pin it. Recordings remain local with the full game artifacts and are excluded from public score exports.

Older prepared editions lacking `prepared-recording-harness.json` must retain `--no-recordings` or use a fresh preparation to adopt recording. The current event predates those prepared files and uses its independently reviewed legacy snapshot path. Browser layout QA is unavailable in the current Work settings; the profile table was checked against the supplied mobile screenshot and JavaScript syntax.

## Render a replay as a shareable video

Keep video dependencies separate from the frozen bot Python environment. Install FFmpeg from its official distribution or Homebrew and use a separate Python venv:

```sh
python3.12 -m venv /private/tmp/robocode-video-python
/private/tmp/robocode-video-python/bin/pip install -r tournament/scripts/tournament-video-requirements.txt
/private/tmp/robocode-video-python/bin/python tournament/scripts/render_tournament_replay.py \
  --recording /absolute/path/to/game.battle.gz \
  --official /absolute/path/to/battle.json \
  --swiss-round 4 --game 1 \
  --output /absolute/path/to/new-replay.mp4
```

The renderer starts no battles. It produces a silent H.264/yuv420p MP4, a poster, a chapter VTT and a local provenance JSON. By default it samples recorded positions at 60 simulation turns/second, with 20 video frames/second. `--turns-per-second` accepts 20, 40, 60, 120 or 240. It preserves all 20 rounds and checks final scores against the official result when supplied. Recorded round-end arrays are already cumulative; it replaces the score totals rather than summing them. Energy bars follow each rendered frame. This is a data-rendered replay rather than a GUI screen capture. It refuses to overwrite a completed output.

## Round 4 playlist and slower playback

All 14 Round 4 games are available through the game selector. **Play all Round 4 games** starts at the first game and advances in draw order through all 14 games. Selecting another game stops automatic advancement. **Stop after this game** keeps the current game playing. Round-jump controls and MP4 downloads apply to the selected game.

The player defaults to **0.25×**, equivalent to 15 simulation turns/second for these 60-turn/second files. The 0.125× option slows it further to 7.5 turns/second. Other choices are 0.5×, 1×, 2× and 4×. The selected speed persists across game transitions. The original faster movies and raw recordings are retained. Downloaded MP4s play at their encoded speed unless adjusted in the external player.

Each movie loads on demand into a browser Blob so seeking does not depend on the host’s byte-range support. Only the selected game is fetched; changing games cancels an unfinished load and releases the previous Blob. Play-all attempts to start the next game automatically. Browsers that require a fresh gesture show **Tap to play replay** and retain the sequence. Initial checks used JavaScript syntax and player state transitions. Subsequent browser QA at 320/390/768/1280px verified the mobile layout and public playback, including a multipart large movie, chapter seeking and 0.125× speed.

To reproduce the playlist, iterate each pairing's `games` in `benchmarks/tournament-2026-11-04/swiss-4.json`. For each game, render its `recording.path` beneath that benchmark directory and pass its `directory/battle.json` as `--official`, `--game` as 1 or 2, and `--swiss-round 4`. Use a distinct output filename per game. The local JSON report supplies the source/video SHA256s, final scores, duration and round chapter timestamps. Public `data.json` lists these movies in `videos`; `video` retains the first movie for compatibility. Each entry supplies `title`, `label`, `mp4`, `poster`, `chaptersFile`, `chapters`, `sizeBytes` and `turnsPerSecond`.

Only user-approved rendered videos, posters and chapters are published. Raw `.battle.gz` files, models and private logs remain local. All 14 recordings passed gzip and game/round-boundary checks. Video reports check every final score against its original official result, and codec metadata is checked for each generated MP4. No games were rerun, and no competitive result changed for this presentation update.

## Run one authorized playoff wave

Use `--stage playoff-1` after all four canonical Swiss reports are complete. The runner freezes their hashes, final seeds and the full bracket in `playoff-registration.json`. It verifies the existing frozen bot/shared inputs and requires recordings. No bot rebuild or new preparation replaces this event’s frozen inputs.

```sh
python3 tournament/scripts/tournament.py \
  --source "$CUP_SOURCE" --output "$CUP_OUTPUT" \
  --java "$CUP_JAVA25" --python "$CUP_PYTHON/bin/python" \
  --site /absolute/path/to/tournament-site/dist \
  --stage playoff-1 --register-only
# Publish the registered draw, then execute its still-unplayed games:
python3 tournament/scripts/tournament.py \
  --source "$CUP_SOURCE" --output "$CUP_OUTPUT" \
  --java "$CUP_JAVA25" --python "$CUP_PYTHON/bin/python" \
  --site /absolute/path/to/tournament-site/dist \
  --stage playoff-1 --resume-unplayed
```

Opening wave is four pairings / eight games / 160 rounds, with at most one extra 20-round tie-break game per pairing. Two-game total scores decide first, round first places second. Only if both tie is game 3 played; its total score decides, or the predeclared SHA256 draw of lots if that game ties. Playoff wins/losses replace Swiss points for advancement. No lower-bracket game starts in this invocation.

Subsequent `playoff-2` through `playoff-7` stages require separate user authorization. Wave 7 is refused unless the lower finalist won G1. A failed/incomplete game stops the playoff wave and remains retained; no automatic rerun, forfeit or advancement. Public exports rebuild the cumulative bracket from canonical completed waves and expose only approved scores/resolutions, never raw replay paths or private telemetry.

## Offer every replay from Round 4 onward

After each subsequent recorded stage, run the replay catalog tool in the separate video venv, then publish the Site snapshot using the Sites workflow. This is part of the tournament publication workflow; rendering starts no games. Do not run a new battle to reconstruct an older unrecorded game.

```sh
/private/tmp/robocode-video-python/bin/python tournament/scripts/render_tournament_replays.py \
  --output "$CUP_OUTPUT" \
  --media-dir /absolute/path/to/private-video-provenance \
  --site /absolute/path/to/tournament-site/dist
```

Use the existing video/provenance directory for this edition. The tool scans the canonical Swiss 4 report and every existing playoff-wave report, including completed games in a partial stage. It requires a recording for every eligible completed game, reuses verified existing movies, renders only missing movies and rebuilds the public catalog in stage/draw/game order. Preserve this catalog step for every later wave before declaring its replay publication complete. The tool never launches or authorizes a tournament stage. Actual hosting publication still uses the approved Sites tools; the script makes no network calls.

Each catalog entry identifies its `stage`, `matchA`, `matchB` and `game`, so replay buttons appear beside the matching result. Previous/next and **Play all this round** stay inside the selected round. Every game from Swiss 4 onward has 20 battle-round chapters, live energy and official cumulative scores. The player retains the 0.25× default and 0.125× slower option. Raw recordings and private models/logs remain local; only approved rendered MP4/poster/chapter assets enter the public Site.

The renderer supports stage labels for playoffs, game 3 for an authorized tie-break, and draw endings. New outputs use immutable names. Existing public movies are never overwritten. A conflicting or partially rendered output requires an explicit new `--edition` rather than a silent retry/overwrite. The batch tool verifies raw/video hashes, official final scores, chapters 1–20, H.264/yuv420p metadata, frame counts, duration and a 24 MiB per-asset ceiling. It lowers the encoder bitrate for long movies. A changed Site data timestamp aborts the final catalog write; rerun to incorporate newer results without clobbering them. Verification is retained in the private `replay-catalog-verification.json`.


### Lossless replay transport and storage limits

Movies larger than 8,000,000 bytes are served as 4,000,000-byte binary pieces.
The catalog supplies `parts` (URL, byte count, SHA256), the original `fileName`
and full-movie `sha256`; `mp4` is null for those entries. The player downloads
pieces on demand, checks size and SHA256 when WebCrypto is available, and joins
them into the exact original MP4 Blob. The download link becomes available after
loading. This preserves quality, timing and chapters. The other movies retain a
direct MP4 URL. Duplicate full movies are omitted from public assets only; all
original rendered movies and raw recordings stay in private evidence.

The Site accepts at most 256 MiB per deployment archive. If later waves exceed
that budget, the user authorized archiving Round 4 public videos. Add
`--omit-swiss4` to the same catalog command for that and subsequent publications.
It removes only verified derived public copies and marks the stage archived;
the private movies, catalog and raw recordings remain retained. It does not
remove scores, change results or run games. Keep the flag on future catalog runs
to avoid republishing archived movies. Never delete recordings to make build room.

The user also authorized archiving older lower-bracket public videos for space.
Use `--omit-swiss4 --archive-lower-before-wave 4` for the current edition: it keeps
upper-bracket videos and the newest wave, while archiving only L-prefixed matches
from earlier waves. The catalog records exact `archivedGames`; result rows state
“Video archived · recording retained.” Original movies/recordings stay local.
Keep these flags on later catalog runs unless restoring archived live videos is
explicitly desired. The private archived catalog records the removed assets.

For recovery launches, check free space and execute the runner within the same
Python process; a failed shell preflight followed by another command can fall
through. Require at least 2 GiB, then `os.execv(sys.executable, [sys.executable,
"tournament/scripts/tournament.py", ...])` only after the assertion passes. The Wave 4 void
ruling documents the authorized single replacement; it is not a general rerun
policy. Preserve invalid engine games separately in physical-budget accounting.
