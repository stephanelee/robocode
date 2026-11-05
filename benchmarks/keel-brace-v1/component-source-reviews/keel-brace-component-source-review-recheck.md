# Independent KeelBrace component recheck — actor filter

**P2 actor-filter finding: RESOLVED in the reviewed revision. Overall operational/source-clearance status remains HOLD because P1 image provenance is unresolved.** No final source CLEAR, runtime approval, game/dispatch, merge or budget authority is issued.

Reviewer `/root/brace_independent_protocol_review` is separate from candidate/harness/CI authors. Parent explicitly assigned this read-only recheck. I edited neither implementation nor old evidence, ran no Java build, freeze, preparation, bot/server/Retcon/game, label, receipt/ref mutation or merge. Only pure/mock/synthetic regression tests and byte comparisons were executed.

## Exact revision and preservation

- Current controller SHA256 `9348f3055605f236543fbb29aa7ddf1a058166b57f6703c4797ae3b93a4b9071` (97150bytes).
- Current pure test SHA256 `8f9dece2e2660cf53bd98c97c52d4ac5d243463d194b3c00d77ea3be90940d25` (47806bytes).
- Original full component finding report remains byte-identical at `/private/tmp/keel-brace-component-source-review.md`, SHA256 `e50f35f0f94889a6e1c291caeeb5c262977cc25690b040c8c3de6314305c16ca`.
- Independently compared all17 files in the old review snapshot: ONLY controller and test file changed. Actual candidate source/JAR, wrapper, Python proof helper, harness docs, registered/bootstrap/CI/generic guard workflows remain identical to that snapshot. Candidate JAR remains SHA2568a1f1c3365846b7fb272a5a1885b65fad4f5f6ace29cc770fed04f34287d7eb9. Prior artifact790-entry/two30-test-execution observations are unchanged, not newly rebuilt here.
- Independently hashed all4 archived pre-repair files against `benchmarks/keel-brace-v1/harness-validation/actor-filter-revision/before-manifest.json`: old controller, old47-test source, status manifest and old test log match exactly. Author's before-fix51-test failure and after-fix51-test pass logs are preserved separately, not overwritten.

## Repair assessment

The controller diff adds ONLY one early author check to each of dispatch_assignment, validate_approval and review_report. Dispatch ignores accounts other than authorized stephanelee before prefixed JSON; approval/report ignore accounts other than the assigned account before prefixed JSON. No unassigned comment is parsed as authority. Relevant authorized malformed JSON still raises/fails closed. Existing targeted revocation author filtering remains unchanged. No game/score/exposure/timing/runtime/receipt or other criterion was changed.

Four meaningful pure regression fixtures cover malformed foreign approval, malformed foreign report preceding a valid report, foreign malformed dispatch, and authorized malformed JSON in all3 transports. I independently loaded the PRESERVED OLD controller in an isolated Python process and ran the NEW4 actor-filter fixtures: exactly1failure/2errors reproduced the three original vulnerable boundaries; the authorized malformed case already failed closed as intended. No source file was edited to run that contrast. The explicit expected-regression assertion passed.

I independently ran all51 current harness pure fixtures with `python3 -I -B -S -m unittest discover -s tools -p test_keel_brace_experiment.py -v`: PASS. Local interpreter is Python3.15.1; this does not claim the registered CPython3.13.17 runtime or required pinned CI has run.

Independent logs:

- `/private/tmp/keel-brace-component-review-recheck-old-source-regression.log` SHA256 `6676a35124c05b7674b13742e2b6793d92785283460dc6e1e1a3f31c2719e6ab` (expected old-source failures; NOT current failure or gameplay evidence).
- `/private/tmp/keel-brace-component-review-recheck-51-tests.log` SHA256 `eeaee019e410b0e04694b7e8d4f5fbe168f21e1454ff72a888db32360c597dc8` (current51PASS).

## Remaining hold and scope

The unconditional image_evidence() failure and current image allowlist/identity guard are unchanged by the six-line actor repair. I did not review incomplete new metadata-probe code or authorize its workflow. The provider-header20261025.179.1 versus ENV20261026.180.1 conflict remains unresolved; no favorable field selection, image waiver or retroactive old runtime admission is supplied. Preserve the first review and independent image audit.

Before registered preparation/games, separately resolve and prospectively review actual image authority/capture policy, review the completed collector/probe under its own scope, publish/freeze exact coherent code/JAR/protocol/inputs, pass required pinned Java17/parity/CI and obtain new exact-head/source clearance. Then an actually active separate reviewer must approve the new finalized original same-worker provenance before its different immutable receipt/game stage. This repair alone opens none of those gates.
