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
