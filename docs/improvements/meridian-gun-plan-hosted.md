# Meridian held gun plan: hosted execution of issue 90

This is a prospective execution addendum to [issue 90](https://github.com/stephanelee/robocode/issues/90)
and the [source/liveness protocol](meridian-gun-plan-local.md). No new candidate,
parameter search, game budget, score threshold or confirmation is introduced.
The current user request authorizes publication and reviewed bounded benchmarks.
Earlier local-only milestone restrictions describe that earlier milestone.

## Frozen comparison

Use baseline `809b880cdcee98d7946d05e6bcd30f623f604348`, JAR
`5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9`,
and candidate source `4a02dfde940356946f35c25a4f3b02640f7bdeb8`,
Meridian.java SHA256 `5268264c49dd1150eb862b4cd59814a9e589abb3e5f284c56bf07dd07989d0d1`,
JAR `7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe`.
Only the experimental committed JAR is updated to its verified build output.
Production AISurvivor source and committed JAR remain unchanged.

The mechanism holds the first fresh, cold gun's selected signed offset and power
until firing or invalidation. Keep movement, learning, radar and original firing
guards. Run exactly the twelve five-round games, seeds, order, ports and cold-state
policy in issue 90 and `benchmarks/meridian-gun-plan/games.json`. All twelve valid
games complete a descriptive comparison, with no automatic promotion or extra games.
Report arm/game variation, wins, score share, shots, hit and firing rates, damage,
energy, skips, collisions, legal-label activity and the common-opportunity no-fire
diagnostic. The diagnostic omits alignment and does not assert shot feasibility.

## Preflight and activation

The workflow is restricted to `ubuntu-26.04`, exact Temurin17.0.21+9 build/tests
and source/JAR parity for production, baseline and candidate, then Temurin25.0.5+8
for battles and Python3.13.18. Accept only the two previously registered v3 runtime
profiles, changing their candidate source/JAR assets only. No runtime substitution.
The preflight label starts zero games. It records installed package file hashes;
freeze those bytes in the repository before independent exact-execution clearance.

The run label requires a clearance token containing the exact reviewed head SHA.
Immediately before acquisition, create the fixed annotated Git tag ref
`refs/tags/experiment-receipts/meridian-gp60-issue90` atomically. Its object binds
the payload SHA and GitHub run ID. Existing ref creation failure aborts; never
delete/reset it or resume the attempt. `run_attempt == 1`, serialized workflow
concurrency, and refusal of an existing execution root supplement this durable gate.
An invalid attempt retains its receipt and all partial evidence. No repeated games.

Private skip markers are checked during a running game; observer-only skip events
are additionally validated from its official recording at the completed-game boundary.
Abort the game on a detected private skip and start no later game after any detected
invalidity. Preserve physical rounds even when validation fails. This states the
available detection boundary prospectively; it does not promise an observer event
callback which the current launcher does not expose.

Upload complete or partial raw recordings, endpoints, staged assets, logs, commands,
runtime/package fingerprints and the activation receipt. Preserve outcome and any
failure in the ledger and issue before completing the experiment. The implementer
will not merge this PR; independent review and required CI still apply.

## Preparation evidence

The coordinator and independent reviewer each ran the24-test candidate suite under
actual Java17, with zero failures or skips. Both builds reproduced the frozen
candidate JAR SHA above, the244-shot baseline replay,2,894-turn no-fire interval
and two-step geometry-only servo bound. These checks do not predict competitive
hits, damage or wins. Existing macOS production golden-comparison failures remain
documented in the local protocol; no test tolerance or production source changed.

During preparation the original shared Git pack became unreadable (reported size
74,787,469 bytes, zero allocated blocks, reads returned EOF). A fresh HTTPS clone
in `/private/tmp/robocode-gp60-recovery` preserves published history and permits
publication without repairing or rewriting the shared object store. No competitive
rounds were played during recovery.

The first publication used `[skip ci]`, so GitHub created no preflight job. This
spent zero games and consumed no durable receipt. The follow-up removes that
commit-message barrier, explicitly excludes this experiment branch from automatic
production battle/Retcon/engine-trace jobs, and keeps Java17 CI build/parity checks.
The normal Meridian CI build materializes the hash-pinned fixture from full Git
history. These workflow-only changes do not alter the registered candidate.

Before consuming the acquisition receipt, require at least346 minutes of the360-minute
job cap to remain, using GitHub's actual compare-job start timestamp. The twelve
1,680-second process caps total336 minutes; the remaining margin is for verification,
reduction and artifact preservation. Failed admission starts no game and consumes no
receipt. It does not justify a replacement of an already consumed acquisition.

The first actual zero-game preflight, [41765322623](https://github.com/stephanelee/robocode/actions/runs/41765322623),
passed all Java17 tests/parity builds but failed when the profiler read Java build
provenance deleted by the earlier root Gradle clean. It played0 rounds and created
no acquisition receipt. Preserve its original ZIP and compressed complete job log
in `benchmarks/meridian-gun-plan/preflight-41765322623/`. Capture provenance after
the root clean build in the corrected workflow; no runtime identity is waived.

The durable receipt also binds the measured Linux boot ID, resolved workspace,
run/attempt and runtime/package/tooling manifest hashes. Verify the same host document
and frozen assets before each game, refusing identity changes or cross-run reuse.

The corrected [preflight41765933504](https://github.com/stephanelee/robocode/actions/runs/41765933504)
passed all builds/parity, the exact allowed Ubuntu image20261026.180.1 profile,
package fingerprinting and Java25 launcher compilation, with0 games/no receipt.
Its originalZIP, complete compressed job log and extracted profile/host/build
provenance are retained under `preflight-41765933504/`. The twelve installed
distributions' exact measured file manifest is now frozen as
`benchmarks/meridian-gun-plan/package-contents.json`. This is the existing pinned
wheel/source environment, not a newly accepted runtime. The acquisition job must
match these package bytes and one of the two existing runtime profiles anew,
then bind its own fresh host document; never reuse the preflight host identity.

The sixty-round study has a descriptive completion rule, not a gameplay promotion
threshold: all12 valid games must complete, and otherwise no comparison decision
is made. Any claimed competitive improvement or claim to beat Diamond requires a
separately registered fresh confirmation. No additional game is implicit here.
