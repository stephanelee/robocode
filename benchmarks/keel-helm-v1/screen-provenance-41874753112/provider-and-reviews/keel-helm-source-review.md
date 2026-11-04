# Independent KeelHelm source/preparation review

**Initial disposition: CHANGES REQUIRED in harness integrity/containment before freeze/publication/stage clearance.** Candidate stopped-heading treatment itself is consistent with the registered intervention. This is source/preparation review, not same-worker actualruntime approval or competitive evidence. No bot/server/game/dispatch/receipt/label/merge action was run.

Reviewed originalregisteredprotocol plusR1–R3, immutable Keel baseline24fb9a5f, KeelHelm source/JAR/tests/localbuild evidence, harness/reducer/JavaBattle/runtimeproof/tests/workflow. Candidate JAR SHA25660ef5b1d33c3deb7100d1e66135f311d4b7b6fd5651cd1146effb899b9ea08c2. All17baseline+productionmanifestfiles independentlyhashchecked; originalKeel JAR77b8a5c3… and productionAIS source/JAR unchanged. Gun/Phys/Track/Knn candidate source bytes equalcontrol. Fullcontrol-versuscandidateSurfer/mainfilediff has only registered first-legstoppedhelper + diagnosticcounters + isolatedname/descriptor/env keys. ExistingsecondWaveSTOP/search/power/gun/radar/learning/ramlogic unchanged. No telemetry influencespolicy.

## Required initial corrections

**B1: accepted approval/report/dispatch must remain current.** consume_receipt reads only locally saved approval/comments/report, and verify_worker re-fetchesPRhead/token/runtime but not current accepted APIcomments. An edit/deletion/revocation afterinitialapproval would not stop first or latergames. Re-fetch the pinned commentIDs and validateauthor,body/hash/timestamps,assignment/task/report/approvalbindings and current exacthead/artifact immediately before receipt and eachnewchild. Rejectmissing/edited/deleted/revoked/ambiguous records; preserveoriginalcopies and new revalidationevidence. A tightlybound explicit revocation record is acceptable failclosed transport; it must be documented/tested, not a new gameplayrule.

**B2: bind actual process metadata and every approval anchor per child.** Afterreceipt verify_worker does not recompute expected_approval==acceptedpayload, so current profile/commands/participant/opponent manifests are not rehashed against the immutable acceptedidentity. execute_game actually reads games/id/prepared-game.json, which is not bound in approvedcommands/profile; command/cwd/environment/identity could differ from the approvedmanifest. Before eachchild, verify all accepted metadata hashes/currentidentity and exact prepared-game bytes, preferably include pergamepreparationSHA in approvedcommands/provenance and compare actual runtimeprocessmetadata to the immutable approvedrecord. No local JSON can retrospectively redefine the acceptedprofile or actuallaunchcommand.

**B3: cleanup must cover owned descendants after root exit.** stop_child returns immediately when Java root poll()!=None, leaving possible living descendants in its start_new_session processgroup without the1800second/teardowncap. Always bound/terminate the ownedprocessgroup after launchercompletion/failure as needed, even when its root alreadyexited; tolerate absentgroup, never signalothergroups, and retain failures. Mockedpuretests should cover exitedroot/liveowneddescendants, timeoutTERM/KILL, noforeigngroup. Do not introduce a realgame/smoke battle.

Root and harnessimplementer were notified directly; reviewer makes no sourcechanges. These are finite harnesscorrections before any freeze/result, not gameplay retuning.

## Findings that pass

- Hypothesis is the real source seam: firstlegSTOP frozeheading while emittedSTOP requestedorbitalturn. NewstoppedStep reuses exactly orbit+unchangedPhys.step withtarget0, keepingpositionstationary and evolvingheading atnativecap beforesecondwaveescape. Secondwave terminalSTOP unchanged. Four newdiagnosticaggregates resetperround and feed no decision.
- Candidate retained15Java17tests (10old+5meaningfulnew) allpass in suppliedoriginalXML; newfixtures include bothdirections/signedspeeds/multiturnSTOP/wallcorner/nextmovingcontinuation and actualplanloop counters. Local logs/runtime clearlylabelHomebrew17/macOSpreparation, not hostedTemurin admission. I did not rerunJava tests/build in thisinitialreview; hosted exactJava17build/parity remains mandatory.
- I independentlyran12purePythonharnessfixtures with allprocess-startpaths mocked; allpass. They coverstrict20keyapproval,duplicates/nonfinite,assignedreport,markers/timingboundary,gatearithmetic,firstinvalidpartialbudget andsyntheticofficialrecords. Existingtests do not cover B1–B3, so their pass does not resolve those gaps.
- I ran the newreducer only on ONE already completed immutableKeelconfirmation recording c1 (no newgame). It reconciles exactly with archiveddecision:0.57981580967share,18wins,5480/498ownshots/hits,6013/613enemyshots/hits,367.3317653/941.5782372damage, shotenergy and recordingSHA951ef160…. It discovers77015turns,maxMs7 from retainedrawrows/markers. This is API/recordingcompatibility validation using olddata, not a newindependentoutcomeaudit or a KeelHelmstrengthresult. Output /private/tmp/keel-helm-existing-recording-c1-review.json retained.
- Screen14games/120r,confirmation18games/300r andfractionalpercentagegates matchregisteredfixedpopulations. Invalids stopremainingIDs,firstinvalidrecordsactualobserved/completedphysicalrounds/unplayedIDs;scoregates do not adaptivelytruncate. IntegermaxMs requiresdigits/nonnegative and<75,ownskip failslive. Candidateexposure requiredonlycandidate, controlN/A.
- Workflow usesonlyubuntu26.04,Java17build/test/AIS/control/candidateparity before exactJava25Retcon/battles,Python3.13.17,nonpersistedcheckout. Childenvironmentallowlist stripsGH/Java/Pythoninjections, freshhomes and separatemodelpaths, pinnedsource/wheels/samples/RetconAPI; noproductiontrainingstatechange. AISfinalcallback drain/modelheader/completeTRAIN evidence bounded.
- Samejob uploadfinalizedprovenance before30minuteapprovalpoll; strict20keys rejectunknown/missing/duplicateJSON/bool-as-int/stalerun/author/hash; reasonedseparatecomment/taskprovenance is mandatory. Current source assignment predatesruncreation and distinguishesreviewerTask fromimplementationTasks; actualrootassignment/provenance remains an independently reviewed proceduralfact beyond a sharedGitHublogin.
- Fixedcreate-onlyannotatedstage receipts,attempt1/serialnocancel,1800secondchild/340minjob/35minadmission,readonlyruntime/package/JDK/source/stagedinputrehash,noautomaticretry/resume. Confirmationgate bindspassingretainedscreen/receipt/run/head/outcomereview and unchangedsourcefreeze; no unconditionalconfirmation.

Initial reviewed file identities (before corrective edits):
{
  "tools/keel_helm_experiment.py": "861ba6bdcb4be6b08439699b622b4293bb2b2abc8bbc66a8f668827235a7fe36",
  "tools/KeelHelmBattle.java": "f745412be54dbe49c664f229166d2e4a319b2dde158c48dcb8f3128265aa95aa",
  "tools/keel_helm_python_proof.py": "19ce6c482f153df92f2ca284e58b9ba1c6bd043a6cc0325978c2c9a7c2d4841c",
  ".github/workflows/keel-helm-v1.yml": "24f209f6d241147ff5ac02e7745e46562d5392ca0c6aff7966d8710b807bbf7a",
  "experiments/KeelHelm/KeelHelm.jar": "60ef5b1d33c3deb7100d1e66135f311d4b7b6fd5651cd1146effb899b9ea08c2"
}

Actual same-worker finalizedartifact/profile/source/opponent/package/native proof and the final immutable publishedhead still require independent review before games. Initial source review supplies no runtimeapproval, stageactivation,competitiveclaim or mergeclearance.

## Independent copied-source Java17 rebuild

After initial source review, I copied only candidate source/test/bot/build/settings/committedJAR into /private/tmp/keel-helm-independent-java17-build and independently ran a clean offline Java17 build with the existing pinned dependency cache. All15pureunit tests pass (10KeelContract+5StoppedHeading),0failures/errors/skips. RebuiltJAR is byte-identical to committed60ef5b1d33c3deb7100d1e66135f311d4b7b6fd5651cd1146effb899b9ea08c2, all788regularentries match. Homebrew17.0.21.0.3/macOSarm64 is preparation only, not futurehostedTemurinruntimeadmission. Command/output evidence: /private/tmp/keel-helm-independent-java17-build.log and -review.json. No bot/server/game/main launched. B1–B3harnesscorrections remain pending.

## Narrow B1–B3 corrective recheck: PASS for freeze/publication/source preparation

All three required harness corrections are resolved in the exact bytes below. CandidateJAR/source/registeredgamepopulation remain unchanged; no gameplay treatment or outcome criterion was altered.

B1: accepted_authority pins complete originalapproval/report/dispatchAPI identity (IDs/nodeURLs/author/bodySHA/timestamps) and finalizedartifactidentity. revalidate_authority refetches eachpinnedID, strictapproval/currentreport/currentassignment/currentPRhead/token/artifact before receipt and everynewchild; edited/deleted/revoked/stale/ambiguous/changedrecords failclosed. A narrowlybound REVOKE transport can onlyremoveauthority. Complete old/new evidence isretained peruniquephase. Afterreceipt the actual remoteannotatedtag/ref and immutableacceptedauthority hashchain are checked again; localcopies cannot redefineremoteauthority.

B2: commandsmanifest now includes every fullprepared-game object, exactpath and SHA. approved_game checks allactualcommand/cwd/environment/participant/coldstate metadata against thatapprovedrecord and exactbytes. Each authorityrecheck recomputes every20keyapprovalanchor plusactualworkeridentity, and execute_game obtains its commandmetadata from the verifiedapprovedrecord. OriginalacceptedAPIapproval→commandsSHA→fullpreparedJSON chain and remote receiptanchor prevent mutablelocalmetadata from silently changing launches or profiles betweenchildren.

B3: ownedprocessgroup cleanup signals its exact start_new_sessionPID/pgid evenifJava rootalreadyexited, boundsTERMdrain10seconds, usesKILLifneeded and reapsparentwithin5seconds; absentgroups are handled. It never selectsforeigngroups. Teardownevidence isretained. Puremocktests cover exitedroot/liveowneddescendants and boundedKILLdrain, without startingbattles.

I independentlyreran all21purePythonharnessfixtures aftercorrections; PASS,0failures. Newtests include unchangedauthority/receiptPASS, APIcommentedit/delete/author/time, revocation/duplicates/assignment, artifact/currenthead, localapprovalanchor/commands/fullpreparedtamper and remoteReceiptmutation, plusownedcleanup. Earlier failingfixturelogs remainpreserved byimplementer; they were fixtureintegration/symlinkroot adjustments, no live failures. Independentlog /private/tmp/keel-helm-independent-harness21-review.log SHA256 79b9a1acc1126114fadd4cd24fd4f082a44cd94cdfce080d1eeb8561b5659236.

**PASS: clear generation of the write-once sourcefreeze and publication of this reviewed isolatedpayload/evidence.** No stage dispatch/gameplay is cleared yet: independently verify the generatedfreeze/exactpublishedhead/toolbytes and assign an activelyrunning runtime reviewer beforedispatch. The future same-running-worker finalizedartifact/fullactualruntime approval remains a separate mandatory gate before any fixedreceipt/game. ThissourcePASS does not substitute for runtimeapproval, outcomeaudit, conditionalconfirmation or separateprotectedmerge.

Exact final reviewed identities:
{
  "tools/keel_helm_experiment.py": "873501cf08e1afbaa8e5dfb326d52f097624a3c52ffdc6b632f2f724e1ee62dd",
  "tools/test_keel_helm_experiment.py": "0b90cf136916bf3c824e27aa4a90d3496448f977ba9712194a0b9debee3e3f3f",
  "tools/keel_helm_harness.md": "9c57e6f989413cc44375a6ccc0e9318cb2ed3b301973c37bc1e97aa16e004bc6",
  "tools/KeelHelmBattle.java": "f745412be54dbe49c664f229166d2e4a319b2dde158c48dcb8f3128265aa95aa",
  "tools/keel_helm_python_proof.py": "19ce6c482f153df92f2ca284e58b9ba1c6bd043a6cc0325978c2c9a7c2d4841c",
  ".github/workflows/keel-helm-v1.yml": "24f209f6d241147ff5ac02e7745e46562d5392ca0c6aff7966d8710b807bbf7a",
  "docs/improvements/keel-helm-v1.md": "d6f236669deaac74abe49699627e32af85f0c7fc604a516ba429e63c03ec913c",
  "benchmarks/keel-helm-v1/games.json": "bb10a78bd6344a85837d6d0e003d72617810542fe9930425462cad4996e6fcd5",
  "experiments/KeelHelm/KeelHelm.jar": "60ef5b1d33c3deb7100d1e66135f311d4b7b6fd5651cd1146effb899b9ea08c2"
}

## CI-only addition: PASS

Reviewed .github/workflows/ci.yml delta: existingJava17 selection is pinned toTemurin17.0.21+9; new isolatedKeelHelm cleanbuild/15puretests/JARparity and21pureharnessfixtures plus reportpath are added. ExistingAIS sourcebuild/committedJARparity remains unchanged, jobubuntu26.04/readpermissions unchanged; these commands launchnogames. CIfileSHA256 de94543582b73f93adc4fc578757c0f8d7a23bfc97814431b810566294d3553a. This metadata/test integration does not alter registeredcandidate/runtime/criteria.

## Generated sourcefreeze check: PASS

Read-only independent check of write-once freezeSHAef876a82069d5c73baf7171aa95cf4058e3bc4a196f46d51bdf5ff1cf0b81a9b: all90listed assethashes match actualreviewedbytes, explicitrequiredsource/JAR/protocol/registry/workflow/helper/opponent/control/productionassets present, exact3toolchainpins/twoimages/120+300budgets/20keyapproval/1800schild+review/340minjob/2100sadmission/75msguard matchregisteredvalues. Correctedmain873501cf… andcandidate60ef5b1d… unchanged. This JSONcheck executedno candidate/helperphysics/runtimecollection or games. Detailed /private/tmp/keel-helm-source-freeze-review.json. Exactpublishedhead/requiredCI and actualsame-workerapproval remainseparatependinggates.
