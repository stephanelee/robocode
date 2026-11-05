# Start another tournament

Use the current event as a reference, while retaining its evidence unchanged.
A new draft is a planning record; generating it does not start or publish an event.

## Create an isolated draft

Choose a new event ID and a full frozen source commit, then run:

```sh
python3 tournament/scripts/manage.py init next-event --source-commit <full-commit-sha>
```

An optional `--site-project-id` records an existing hosting project without storing
its URL. The helper creates `tournament/events/next-event/event.json` and a README,
with separate `benchmarks/next-event`, `local-evidence/next-event` and
`tournament-sites/next-event/dist` paths. It refuses an existing ID and leaves the
current-event pointer unchanged. Do not reuse the old event's output directory.

## Register before playing

1. Fetch the chosen main-repository revision, enumerate all candidates including
   experimental bots, and list proposed guests and rules for the user before play.
   Freeze names, versions, source/JAR hashes, guest licenses and dependency pins.
2. Select practice/Swiss/playoff budgets, fresh draw and bot seeds, a fresh lottery
   domain, qualification tie rules and the authorization limit. The template uses
   two practice games per bot, four Swiss rounds, top-eight double elimination and
   two twenty-round games per pairing. These are editable proposed defaults.
3. Build with Java 17 and verify source/committed-JAR parity. Pin Java 25 for Retcon
   and battles, and freeze the Python/shared runtimes. Preserve failed checks and
   runtime provenance. Play on minimind, not GitHub Actions.
4. Prepare a reviewed event-specific runner and prospective protocol. The existing
   `tournament/scripts/tournament.py` has the original roster, report commit and seeds; the
   bracket lottery domain is also fixed in `tournament/scripts/tournament_playoffs.py`. Merely
   passing a different `--output` or source directory does **not** parameterize all
   of those values. Do not run a new event with stale provenance or silently edit
   the ongoing event's frozen execution helpers.
5. Have a separate reviewer check the prepared manifest, bracket, budget and runner
   pins. Obtain the user's requested stage scope, then run only that scope. Do not
   tune frozen bots or carry learning models between games.

This starter deliberately leaves seeds, entrant pins and authorization unset.
Finish these steps before marking the draft ready. If the format changes, review
runner and website assumptions too; the template is not a general bracket engine.

## Prepare the website

Use the reusable [website source](website/), initialize the new event's static
output and select or create its hosting project through Sites. Use the returned
project ID in the event manifest. Supply the deployment's base URL through
`TOURNAMENT_SITE_URL`; do not put a production hostname into scripts or templates.
The link helper reads that variable and supports another event manifest:

```sh
python3 tournament/scripts/manage.py site-url \
  --event tournament/events/next-event/event.json --stage practice
```

Publish the site and profiles before the first games when requested. Keep the
chosen audience, blue design, guest highlights and winner indicators. For this
invitational's policy, request `noindex,nofollow`; these directives do not make a
public site private. Use relative links inside the website.

## Run, retain and publish

Require at least 2 GiB free before launch. Put the disk check and `os.execv` in a
single process so a failed assertion cannot fall through to another shell command.
Keep observer recordings, invocation/logs, official results and per-game models
from the first stage. Validate result identities and round coverage, gzip CRC,
recording boundaries and final scores. Stop on any failure; retain the exact
attempt and report. No automatic rerun, forfeit or advancement. A replacement
requires an explicit ruling and separate physical-round accounting.

Publish verified scores as available, then derived replays beside the game rows.
Render in a separate video environment after battles. Keep chapters, live scores,
slow playback and hash provenance; split large movies losslessly for transport.
Record any user-authorized public archival, with private original movies and raw
recordings preserved. Set archival cutoffs for the new event rather than blindly
copying this event's Round 4/lower-bracket cutoff.

Update the event record, score index, physical/competitive budgets, detailed
reports, [improvement log](../docs/improvements.md) and PR before marking the scope
complete. Do not infer strength improvements from tournament results. A separate
reviewer handles review/merge; an implementation agent must not merge its own PR.
Keep prior published evidence SHAs and author identity unchanged.

[Detailed runner/replay reference](../docs/tournament-running.md) ·
[Current event's measured record](events/2026-11-04/README.md)
