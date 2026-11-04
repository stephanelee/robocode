# KeelHelm harness interface

This implements issue107's registered protocol; it opens no additional game or
fixture-battle budget. All validation below is source-only/pure fixtures until
separate exact-head review and actual same-worker admission. The workflow's
source-clearance token is `KEEL-HELM-SOURCE-CLEARED <full head>`. The source freeze
is generated once with `python -I -B -S tools/keel_helm_experiment.py freeze` after
all reviewed source/JAR/tool files are final. It never dispatches or prepares.

The workflow uses exact Java17 builds/parity first, then switches to exact
Java25 for Retcon and battles. Both Keel arms use their committed unchanged JAR
or isolated sibling JAR. No development game or runtime preparation was run by
the implementation checks. The pure tests create synthetic gzip observer rows
and mock all process-start paths.

## Active reviewer dispatch record

Before dispatch, root must actually assign and start a separate review agent.
Issue107 must contain exactly one comment for the exact stage/head, with this
prefix followed by one JSON object (no Markdown fence or trailing text):

`KEEL_HELM_DISPATCH_ASSIGNMENT_V1`

Exact object keys are `schemaVersion`1, `repository`stephanelee/robocode,
`issue`107, `stage`screen or confirmation, full `headSHA`, `reviewerTask`,
`reviewerAccount`stephanelee, `implementationTasks`array, `assignmentToolRecord`
object, and `assignmentToolRecordSHA256`. The tool record contains the actual
separate-agent assignment/tool identity, the same `reviewerTask` and
`activeBeforeDispatch:true`; preserve actual tool-call/thread/time provenance.
Its digest is SHA256 of compact sorted-key JSON bytes (the helper's `canonical`).
The reviewer task cannot be an implementation task. The comment must predate the
actual workflow run creation. A shared GitHub account alone is no evidence of
independence: the actual separate-agent report/tool provenance is also retained.

The approval's `dispatchAssignmentSHA256` binds the **exact dispatch comment body
UTF-8 bytes**, including prefix/newline/JSON formatting. It is not the digest of
a shortened or normalized substitute.

## Finalized artifact and runtime review

The executing stage uploads `provenance/` with hidden source files included,
then records upload-artifact ID/digest and verifies the GitHub artifact metadata
for that exact run/head. Its provider `updated_at` starts the conservative
1800-second deadline. HTTP calls are individually capped20seconds and the
remaining deadline; comment scanning is bounded to2000comments. No gameplay
runs in another approval job. Missing/late approval aborts0games.

The actual runtime profile includes exact image, boot/workspace/run, Python
binary/base source/cache/native proof, owned venv package/source/wheel bytes,
JDK17 build and JDK25 file trees, tools, participants, opponents, source/JAR
freeze, commands/environment/staged input files and permissions. Collection is
once; rechecks only read/hash existing files and never prepare/repair packages.
The provenance artifact exports the reviewed source/JAR bytes, manifests,
original build logs and full fingerprints. Its original artifact ZIP/digest and
all files must be independently downloaded and reviewed before approval.

The strict approval comment prefix is `KEEL_HELM_RUNTIME_APPROVAL_V1` followed by
one JSON object with exactly the20keys fixed in the protocol (`APPROVAL_KEYS` in
the helper). Duplicates, unknown/missing keys, nonfinite JSON, bool-as-integer,
wrong author/head/run/attempt/boot/stage/artifact/population or stale comments
are rejected. `expected-approval.json` is created after upload; a reviewer can
reconstruct the exact values from the finalized artifact and GitHub API. No
extra approval keys may be added for explanatory prose.

Before that approval, the same assigned reviewer publishes a separate comment
with prefix `KEEL_HELM_RUNTIME_REVIEW_V1` and one JSON object containing:

- `approvalPayloadSHA256`: canonical compact sorted-key digest of the entire
  strict20-key approval payload;
- `reviewerTask`: the assigned separate task;
- `reviewReport`: nonempty reasoned review including checked source/JAR/runtime,
  package/native/artifact/population/command identities and limitations;
- `agentToolProvenance`: actual separate-agent tool/thread evidence, including
  matching `reviewerTask` and `reportGeneratedBeforeGames:true`.

The review comment must postdate artifact finalization and precede the approval.
The runner retains complete raw API comments and report/provenance, not just the
parsed20-key token. Root must preserve actual agent assignment evidence rather
than inventing a task identity or substituting a shared login.

## Live authority and immutable per-game records

The initial accepted API comments remain unchanged local originals. An
`accepted-authority.json` records their exact body/hash, provider IDs, author
identity, creation/update timestamps, original-copy hashes, original strict
approval payload and finalized artifact identity. That whole chain and its
canonical digest are embedded in the permanent annotated stage receipt.

Immediately before receipt consumption and before every new child, the runner
fetches the accepted approval, report and dispatch comments by their pinned API
IDs. Deletion, edited/reverted text, changed timestamps/author, missing or
ambiguous replacement comments, changed reviewer assignment/provenance, current
PR-head clearance or artifact identity all fail closed. It recomputes the
strict20-key payload from current full profile/participant/opponent/commands/
freeze/game anchors and compares it with the original accepted payload. After
receipt creation it also queries the permanent remote ref/tag and checks the
original receipt-bound authority chain. Original accepted copies are never
replaced by current responses; each check and any failure are retained under
`approval-revalidations/{before-receipt|before-ID}.json`.

An assigned reviewer can revoke authority with a new issue107 comment prefixed
`KEEL_HELM_RUNTIME_REVOCATION_V1` followed by one JSON object. Its exact keys are
`schemaVersion`1, `verdict`REVOKE, `repository`, `issue`, `stage`, `headSHA`,
`runId`, `runAttempt`, `bootId`, `artifactId`, `reviewerTask`,
`approvalCommentId` and nonempty `reason`. Identity fields must match the
original approved worker, and the comment must be from the assigned authorized
account after the accepted approval. A targeted malformed/inconsistent
revocation also fails closed. This transport only removes authority; it cannot
open a game budget. Any revocation prevents the next child without repair,
resumption or replacement.

Every command-manifest row now contains `id`, `preparedGameFile`,
`preparedGameSHA256` and the **complete** original `preparedGame` object. The
exact bytes include all executable command, environment, cwd, participant and
opponent identity, input/cold-state metadata. Before execution, those bytes and
the complete object must match the approved manifest; the manifest's own hash
must still match the original accepted API approval. Updating a mutable local
manifest alongside a prepared file cannot redefine the accepted command.

Owned teardown signals the original `start_new_session` process group even if
its Java root has exited, then performs a bounded10second TERM drain and KILL
if needed, with a separate bounded parent reap. It never signals another group.
The actual group/TERM/KILL record is retained as `games/ID/teardown.json`.

## Gameplay gates and outcome evidence

After approval, the runner rehashes the still-running worker without rebuilding,
checks current PR head/source token and35minutes remaining, atomically creates
the fixed annotated stage receipt, then starts games once in registered order.
Before each child it rehashes runtime/all registered inputs again and checks the
clock. Child environments have explicit allowlists; no checkout credentials or
GitHub/runtime tokens reach bots/server/launcher children. Private writable
homes and game outputs are fresh; Keel model hashes are N/A because both arms
learn only in memory. AIS hashes are recorded only for actual stored TRAIN
telemetry/model files.

Live complete marker lines are checked every250ms; a skip, malformed/duplicate
stats/maxMs or maxMs>=75 terminates only the owned process group. Every completed
round requires exactly one start/stats/end and integer maxMs. Candidate-only
exposure counters are recorded; baseline counters are N/A. Any invalid/partial
game, runtime change, timeout or admission failure stops the stage at that first
integrity failure. Scores never adaptively stop a stage. Retain attempted game,
observed/completed physical round IDs, partial data/traceback and unplayed IDs.

The observer reducer records official per-round/game scores/firstPlaces, turns,
uncapped event damage, shots/hits/rates, shot energy/power mix and contacts.
Missing/duplicate endpoints/ticks/participants or cold AIS logs invalidate the
stage. Exact fractional score ratios implement percentage-point gates without
binary floating boundary artifacts. Between-game variation remains descriptive;
adaptive rounds are not independent experimental units. An independent outcome
audit must read raw recordings without importing this reducer.

## Conditional confirmation transport

Confirmation also requires a newly reviewed exact head and actively reviewed
same-worker provenance. The read-only `benchmarks/keel-helm-v1/screen-evidence.json`
record has exactly: `schemaVersion`1, `sourceFreezeSHA256`, `screenRunId`,
`receiptTagSHA`, `decisionFile`, `decisionSHA256`, `retainedEvidenceFiles`map and
`independentOutcomeReviewCommentId`. It must retain and hash the full screen
raw decision and all fourteen official recordings. The unchanged candidate and
source freeze bind both stages; publishing evidence may change the Git head.

The actual permanent screen receipt/tag/run/head and completed attempt1 are
queried. The full retained decision is recomputed against all registered screen
gates and must open confirmation. The referenced independent issue107 outcome
comment uses prefix `KEEL_HELM_OUTCOME_REVIEW_V1` plus JSON binding
`verdict:SCREEN-PASS`, stage screen, exact screen head/run, source-freeze and raw
decision digests, a separate `reviewerTask` and actual `agentToolProvenance`.
This transport record supplies no clearance by itself; separate source/head and
actual runtime approval remain mandatory. No completed/partial battle or spent
receipt may be reset, replaced, resumed or rerun.
