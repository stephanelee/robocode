# Facet versus Diamond Port100 — invalid/stopped after20

2026-11-04. [PR103](https://github.com/stephanelee/robocode/pull/103), [issue101](https://github.com/stephanelee/robocode/issues/101), [prospective protocol](facet-diamond-100-protocol.md), [frozen registration](../benchmarks/facet-diamond-100/attempt-1/registration.json), [all raw evidence](../benchmarks/facet-diamond-100/attempt-1/), [independent review](../benchmarks/facet-diamond-100/attempt-1/independent-review.md).

**INVALID screen; reject promotion.20/100officialphysicalrounds;80unrun.** No retries, replacement, automaticcontinuation, retuning, confirmation or merge. Separate/unpooled from the original invalid140rounds. Both attempts together have160officialphysicalrounds; no later100round screen is authorized automatically.

| Official outcome, cold game1/5 | Facet | Diamond Port1.8.28 |
| --- | ---: | ---: |
| Total score |262|2952|
| Official first places |0/20|20/20|
| Official capped bullet damage |262|1460|

Facet share262/(262+2952)=**8.1518357187%**. Only one cold20-round cluster was acquired; rounds share onlinelearning and are dependent. This is a poor observed result, retained despite acquisitioninvalidity; it is not a complete five-game gate assessment or generalstrength estimate. No equality/rating/improvement claim.

All20ownschema3completionrows1..20 exist exactlyonce and report successful terminalobservation/hash/snapshot. The required finalFACET_model.dat is absent, so acquisition stopped at stagefacet-model-export withFileNotFoundError. No finalmodel or partialmodel file is present in the retainedartifact. Successful roundrows do not satisfy the model-export gate. Exactcause is unproved.

| Reported own telemetry | Value |
| --- | ---: |
| Requested / accepted / confirmed physical shots |436/436/436|
| Hit callbacks / incoming hits / bullet contacts |31/317/14|
| Raw event bullet damage given / taken |262.370641/1516.514522|
| Wall / bot collisions; reported skipped events |0/0;0|
| Decisions / observed turn gaps / unmatched incoming |17699/3/3|
| Mean achieved legal-scan range; scan age mean/max |446.529095;0.050790/23|
| Decision p99 bucket interval |[4.194304,8.388608)ms|
| GC count / milliseconds |678/813|
| Max observed predictor steps / neighbor payload bytes |481/5160240|
| Movement emergencies |51|

Raw eventdamage can exceed officialcapped damage; collision/skipcounts are reported owncallbacks, not independent engineproof. Three turngaps are retained and not silently equated to zero lostticks. Cumulativefallback gauges are not naively summed as events. Primarytrace reached100000rows byround2/turn4250; boundedcritical evidence retained20roundhashes/20completionrecords/3lateTickrecords. No ownedcriticalfailure record is present, but that does not prove GameEnded/export execution or exactcause. ModelSnapshot's success counter occurs after finalroundsnapshot and is not a dedicated game-exportevent.

Frozen measured botJAR817a9d53ce944d27e3f536a92869eeca2e0aec20ef5c52b2dab882e931205792; canonicalsource/inputdc5cc37c364b7ff76b122116cdc6a3d65a85abd102202a2f75f8add65b68ec5d; config9ee52f73dc46e8503fe784d116e247fefc566dcb4e183215385840869d412a25. RegistrationSHA1db6ea8f4757bf9e1e4c77c6fbc900dd7ad2deac0d2749ccc2d2b96db220815b; acquisitioncommit89525947cce0fa94a784e79b92e9ef1ee3fc1d1f, reviewedcandidate57ec8d521ccb4547240e36964e9a0c7bcf24ad66. Exactopponent/runtime/engine/infrastructurepins in registration; Ubuntu26.04x86_64, pinnedCPython3.13.9sourcebootstrap succeeded; Java17build/parity completed before preservedTemurin25.0.1build8runtime. ColdFACETseed49667,Facet-first launch; engine seed/seat/positions uncontrolled.

Run41884501684/job126765233621. [Original full artifact](https://github.com/stephanelee/robocode/actions/runs/41884501684/artifacts/13350324848),1968309bytes ZIP SHA2569b8dbe7b83fd2c48062b94cefc1a2b55315835b958f6e6089f323c9247134a15; durableverifiedZIPparts and selective own/public/runtime files committed in rawdirectory. Broadfailedlogretrieval was rejected by automatic approval review because it could expose opponenttracebacks/source. It was not bypassed: only ownstructured/public data and officialcontracts were examined; opaque runner.log stays unopened in originalZIP.

## Offline runner repair and remaining authority

Officialfrozen engine broadcastsGameEnded without an alive/dead filter; officialAPI publishes usercallback before internalstop/flush. Death alone does not justify skipping export. RepositoryBenchmarkBattle can closehandle/runner immediately after observerresults; its existing explicitcompletion waits do not coverFACEToutputs. Python's later5s readiness wait cannot retain a bot already stopped. This is a concrete lifecycle hazard, not proof of the exact observed cause.

[Prospective runner repair](facet-runner-completion-repair.md) adds a bounded own-output barrier inside runner lifetime after persisting officialresult and before closinghandle. It changes no gameplay, botJAR/profile, engine or opponent. OfflineJava17fixtures validate delayedwriter/finalrename, partial/missing/duplicate/failedmetrics, path/symlink/sizebounds andtimeout; no newgames. Independentreview/CI and a concrete newregistered userbudget/exception are required before furthermatches. Old20remaininvalid and their unfavorable result remains evidence. Allunrun80rounds stayunrun unless explicitly reauthorized under a newprospectiveprotocol; no gatewaiver or automatic retry.
