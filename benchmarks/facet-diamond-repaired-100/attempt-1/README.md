# Facet versus Diamond Port: repaired100-round screen

[PR103](https://github.com/stephanelee/robocode/pull/103), [issue101](https://github.com/stephanelee/robocode/issues/101), [prospective protocol](../../../docs/facet-diamond-repaired-100-protocol.md), [full result](../../../docs/facet-diamond-repaired-100-results.md).

Acquisition complete: five cold20-round games,100officialrounds,100successfulschema3completionrows andfive finalmodelexports. Facet1/100firstplaces and5.7386344431%equal-game score share; registeredengineeringgatespass,40%scoregateFAIL, rejectprofile/promotion. No extra games, tuning or confirmation. Priorinvalid140and20 remainseparate/unpooled.

Original [artifact](https://github.com/stephanelee/robocode/actions/runs/41889566437/artifacts/13353138114),20752953bytes ZIP SHA256957e78cc5574c6c8619809a06ede7c5d08bc91925281b85aba822fa28546dbb1. ImmutableZIP split into20verified1MiBparts (lastpartshorter). Concatenate raw-artifact.zip.part* in numericnameorder andverify size/hash fromartifact-provenance.json before ZIPuse. Models andfull owntraces are inside theoriginalZIP; modelbytes are archived/hashed only, neverloaded/trained. Opponent runner.log members remainopaque/unopened; extract only ownFACET/publicbattle files for analysis.

Readableofficialresults,ownmetrics/critical,generatedreport/runtimeprovenance and independentreview/analysis accompany theparts. Missingorfailed data would remain evidence, not zero-filled or dropped. These arefiveindependentgameclusters, not100independent observations. NoDiamondbehavior/equality/rating claim.

Portable calculation: after reconstructing theoriginalZIP and selectively extracting own/public members, run `python3 independent-analysis.py --root /path/to/extracted/build/facet-diamond-repaired-100 --output /tmp/facet-diamond-analysis.json`. The script checks own model hashes only and never imports a model or reads opponent logs. See independent-review.md and processing-notes.md for scope/limits/local extraction failure.
