# Independent KeelBrace component source review

**Verdict: CHANGES REQUIRED / operational HOLD. No source-clearance or game authority.** This review covers the current candidate, six harness components and root CI/bootstrap/guards in `/private/tmp/robocode-keel-brace`; it is not an exact final published-head/freeze review. Candidate source producer is c5a4c3f97e11358656dcf6821529e9917c757bf5. Candidate and main helpers were not edited by this reviewer. New metadata-only image-probe code was expressly excluded from this review because it is still under implementation and has its own reviewer.

Reviewer: `/root/brace_independent_protocol_review`, separate from candidate author `/root/brace_protocol_recovery`, harness author `/root/brace_harness_impl` and root/CI author. I read AGENTS.md, actual published issue114 protocol and the existing independent image audit. I ran only stdlib/mock/synthetic pure checks and offline ZIP/source/JAR verification. No freeze, preparation, runtime collector, bot/server/Retcon, numeric fixture battle, labels, receipt/ref writes, source edits or merge were performed.

## Blocking findings

### P1 — actual image authority is unresolved (known open hold)

Raw original provider startup headers for Brace's zero-game run41934557748 and both prior Helm runs report image20261025.179.1; captured ENVImageVersion reports20261026.180.1. The former is outside the registered two revisions. I read the separately preserved image-audit report; no current-worker mapping/attestation resolving that conflict exists in this component set. No environment-only allowlist admission or old runtime-CLEAR claim can be inferred.

Current controller correctly FAILS CLOSED: identity() calls image_evidence(), which unconditionally raises UNRESOLVED IMAGE PROVENANCE before real preparation or gameplay. Keep that hold. A separately reviewed bounded zero-game capture and prospectively approved authority policy must resolve actual required file/setup/header/environment identities and contradiction handling; then final controller/code/protocol/pins, exact published head/source/JAR/CI and NEW actual same-worker provenance still require review before any game. Synthetic image_consensus agreement is not runtime evidence. This finding does not change raw historical score arithmetic and does not authorize old-game repetition.

### P2 — an unassigned public issue commenter can abort registered authority scans

`tools/keel_brace_experiment.py` dispatch_assignment line415, validate_approval line630 and review_report line666 call strict prefixed JSON parsing BEFORE checking the comment author's login. A comment by an unrelated/unassigned account with the recognized prefix and malformed body (for example `KEEL_BRACE_RUNTIME_APPROVAL_V1` newline `{`) raises JSONDecodeError. During wait_approval or per-child revalidate_authority this exception escapes and stops the one-attempt stage, including a partially consumed stage. The account cannot grant authority, but currently can cause loss of an otherwise authorized acquisition solely through non-authoritative text. This is separate from the intended fail-closed handling of an assigned authority's malformed/edited/revoked records.

Pure no-network/no-process reproduction is retained in `/private/tmp/keel-brace-component-review-unassigned-prefix-probe.log`: all three functions raise JSONDecodeError on the unrelated author before any identity or stage matching.

Required narrow correction before source freeze: ignore comments from accounts other than the assigned/authorized account BEFORE parsing these three transports. Keep strict parsing/fail-closed semantics for relevant authorized records and targeted malformed authorized revocations; revocation() already filters its account before parsing. Add pure fixtures proving malformed foreign prefixes are ignored in approval/dispatch/report scans while malformed authorized records still fail closed. Do not change any game, score, exposure, budget or runtime criterion. This reviewer did not edit that code.

## Candidate and pure semantic evidence

The actual current main diff is retained in `/private/tmp/keel-brace-component-review-actual-main.diff`. It changes original Keel only for isolated identity/paths, a separate constant-space RamLedger, same-turn ram snapshot application and exactly the five prospective diagnostics. Gun.java, Knn.java, Phys.java, Surfer.java and Track.java are byte-identical to original Keel, so rejected Helm STOP heading is not inherited. All13 immutable baseline files and four AIS production hashes matched; one absent sparse checkout file (`benchmarks/keel-v1/profile.json`) was checked from the actual HEAD Git object rather than falsely claimed materialized. Materialize it before any eventual local freeze; full Actions checkout includes it.

RamLedger follows the registered epoch: raw callbacks counted, stale/late corrections ignored, newer unconsumed turn replaces old bucket, first accepted scan consumes/discards the matching bucket even when consecutive history is absent, and bullet correction map/cleanup remains original. Eligible counters are called at the consecutive enemy-energy inference boundary before power/own-history gates. Physical correction does not branch on diagnostic count values; overflow emits one integrity marker and cannot silently authorize a successful stage. Scalar power/gate comparisons are O(1) diagnostics, not a legacy shadow search or policy. Protocol counter bounds/invariants match the parser.

The JUnit source exercises actual candidate callbacks and original frozen Keel.class callbacks through inert SDK objects, never start/run/go/network/event dispatch. It tests RAM alone/three powers/following shot/bullet epochs/missing/stale/duplicate/reset/no-ram unchanged features and checked diagnostic overflow, plus inherited contracts. This establishes narrow semantics, not strength.

I independently ran the offline artifact verifier (no JVM). The exact original bootstrap ZIP1463862B/SHA25671b036d63fe40bf0e83d10f907ca9f09720b3998f6aa3846c78d68d749bebeb6 has16 regular members with CRC/GitSHA/SHA256 matches; the two actual output JARs are byte-identical872696B/SHA2568a1f1c3365846b7fb272a5a1885b65fad4f5f6ace29cc770fed04f34287d7eb9,790 regular entries/826 including directories, Main-Class KeelBrace and no original Keel.class/JUnit/test classes. All27 producer source hashes match immutable producer Git bytes; current README is the sole documented later textual difference. Raw producer build log contains30 unique PASSED identities twice; retained second XML reports30/0errors/0failures/0skips. First XML was overwritten and is not claimed retained. Original committed candidate JAR was absent, so this is two-build parity, NOT an already performed committed-JAR parity check. The later exact-head committed JAR and Java17 AIS/Keel/Brace checks remain required.

The bootstrap's raw image contradiction is preserved and blocks approved-image/runtime claims. Its separately observed Java17 build facts are not a waiver of that policy. No Homebrew Java build was used by this reviewer.

## Harness and CI checks

I independently reran all47 pure harness fixtures with local Python3.15.1 using `python3 -I -B -S -m unittest discover -s tools -p test_keel_brace_experiment.py -v`: PASS. The log is `/private/tmp/keel-brace-component-review-pure-tests.log`, SHA256484f883a0f9be6fcde37aed7b1621cc500855071e4e2fe368f925d0507bedc69. This is NOT a CPython3.13.17 actual-runtime claim; pinned-runtime CI remains later. The offline artifact log is `/private/tmp/keel-brace-component-review-artifact.log`, SHA256a7a22e7143af0479a70ca507ffd3dad9cc4671ef05309a663c3913cba507f150.

Reviewed source covers exact14/120 and18/300 population hashes, participant identities/seeds/ports/launch order, rational equal-game gates, Retcon+1pp/Diamond−3/−2pp regression guards and zero-stage exposure INCONCLUSIVE precedence without filtering games. Markers reject skips in either arm, noninteger/negative/maxMs>=75, missing/duplicate rounds and overflow. Candidate counters are bounded/invariant-checked and cannot exceed recorded own contacts; their deliberate delivered-callback semantics do not establish a full private callback trace or causal benefit.

Reviewed authority chain binds exact20/10/4/13 schemas, separate active reviewer before dispatch, original finalized same-worker artifact and immutable API comment/body/author/timestamp/digest copies; strict live head/token/input/runtime/artifact/receipt/revocation checks precede activation/every child. Complete prepared command bytes are tied to the accepted payload. Fresh staged homes/no models, child environment allowlists and persist-credentials:false prevent repository tokens reaching bots/server/launchers. Source freeze/participant/runtime hash checks remain separate from root's operational permission to label a stage; the identified unassigned-prefix issue must still be corrected.

Custody admission checks8GiB actual named audit storage and timestamp,20GiB hosted preparation/8GiB before child,4GiB sticky owned-output cap,2GiB finalized original artifact cap. Virtual ZIP verification streams all range/member bytes/metadata, exact embedded freeze paths/source Git bindings and14 original recording paths. Screen gate requires passing fixed score/timing/exposure gates, exact historical receipt/run/head, finalized export validation and separate post-completion independent outcome review; no old approval supplies confirmation authority.

Child groups have one1800second outer budget with reserved teardown time, Java result wait1740s/callback flush15s, bounded TERM/KILL/reap including already-exited Java root cases; actual overrun/reap fault invalidates the stage. Score/contact/exposure never adaptively stop an acquisition. First infrastructure/integrity failure preserves attempted/observed/completed/unplayed budget; local later custody ENOSPC is distinct. Full final artifact validation is an external post-outcome companion and does not rewrite the original raw decision; incomplete export cannot open confirmation.

CI/bootstrap/generic changed jobs remainubuntu-26.04; root Java17 is exact17.0.21+9 and retains production committed-JAR parity. Bootstrap's missing-JAR condition is confined to initial zero-game build; the registered stage has no missing-JAR waiver. Generic numeric battle/Retcon/engine jobs are guarded out on Brace so they cannot spend unregistered games. Final committed source/JAR and full required CI still need later exact-head verification.

## Component snapshot and remaining gates

Snapshot `/private/tmp/keel-brace-component-review-snapshot.json` SHA2569dcbee403a5550dd2f5870f6571131358e8155c568a26170c6aa648194f8919e records17 exact reviewed component hashes:

| Component | SHA256 |
|---|---|
| `tools/keel_brace_experiment.py` | `cb4ba5f0d117d4e2773d84e895bc7ee325c5b221787653377e05cf4eb78d203e` |
| `tools/KeelBraceBattle.java` | `ffa0edb60703e057b2908c40aac8c0c0666d00ca48419c346a8ea73c27572a3f` |
| `tools/keel_brace_python_proof.py` | `19ce6c482f153df92f2ca284e58b9ba1c6bd043a6cc0325978c2c9a7c2d4841c` |
| `tools/test_keel_brace_experiment.py` | `57c4e7945c80e28f3cb4b22b6a9ce756f1db447bc6f5ea89449adc00d5d54604` |
| `tools/keel_brace_harness.md` | `cf739dceedf9f2bac50cffed707ac1382127e6e228aa4e6c7daf3ef3236e932e` |
| `.github/workflows/keel-brace-v1.yml` | `75ef80d4133d8e731462c65f8e561448920f1d3a489c8c612563a8f727d89b21` |
| `experiments/KeelBrace/src/KeelBrace.java` | `b3e1e86898a42247b6b7d919f87ab27b2c6e1695da2c58b1914dc962cb498a03` |
| `experiments/KeelBrace/src/RamLedger.java` | `602bbc6a98b434afbe593a2e2593b7ee6a6c72a3c11aad6d2f08051ac87c136b` |
| `experiments/KeelBrace/test/KeelBraceChronologyTest.java` | `d64726775fb2cc9138acfeed35be0a91573016e47910a551ea239901fda34f95` |
| `experiments/KeelBrace/test/KeelBraceInheritedContractTest.java` | `92732812b9b46c32350abd5782f4d7e4234d8b8f5147d8da71401b49e01a4bd5` |
| `experiments/KeelBrace/build.gradle` | `1e589a3d60049ecd4da8652766779d8d88c55fb3dc9ad124b3a9f678d1de044b` |
| `experiments/KeelBrace/KeelBrace.jar` | `8a1f1c3365846b7fb272a5a1885b65fad4f5f6ace29cc770fed04f34287d7eb9` |
| `.github/workflows/keel-brace-build.yml` | `bb1029c77b1a02a96afb54d42e1afb08c79580ea003d6b4b1cb0177f1a9e8e8d` |
| `.github/workflows/ci.yml` | `86d90b4e09dd17f350b84cffd92a31b337d48a96eb8ec079d908c309cce04366` |
| `.github/workflows/benchmark.yml` | `cef2254ba517f104ef993f94e94d4f66986fdda852b199cec24b1e84c871adb6` |
| `.github/workflows/retcon.yml` | `32f694c60a84ae7274d2a4c1d1b5067dce36b39806a466a5c149d5cbd85fad3e` |
| `.github/workflows/engine-trace.yml` | `82c25f6d28742145d5cd1cdc013fc5677a86c596a9c8047c80e23b0055827e20` |

The six harness implementation-status hashes matched at review. Any later image collector/controller/fixture/CI or candidate edit requires recheck of changed components; this report is not evergreen clearance. No final source-freeze manifest or final published source CLEAR token was issued. Resolve the P2 boundary, independently resolve the image-policy/capture hold prospectively, publish all real code/JAR bytes, rerun required pinned tests/parity, then request final exact-head/source review. Only subsequent fresh active runtime review may open a registered stage.
