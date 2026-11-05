# Independent audit: unintended inherited CI acquisition before KeelWake implementation

Actual reviewer: `/root/wake_independent_review`, independently assigned by root for prospective issue118/PR119 review. No candidate, harness, decoder or guard code was authored by this reviewer. This review launched no games, changed no Git index/source/ref, and repeated no acquisition.

## Finding and disposition

Creating PR119 at registered proposal head `48bd8e343b9ef017b35aa9b8103adc2825f87c02` unexpectedly ran inherited engine CI because three existing branch guards omitted `codex/keel-wake-v1`. These were actual engine executions, not zero-game checks. Exactly **3 separate games / 3 physical rounds** completed: two unchanged production-AIS smoke games and one two-probe timing game. No KeelWake binary/source existed, no original-Keel comparator participated, no Diamond participated, and no Wake screen/confirmation receipt exists. The planned **14/120 screen and conditional18/300 confirmation remain unused**, but global physical acquisition is not zero. Preserve this preparation-policy failure separately; do not pool, replace, rerun, claim strength, or retrospectively call it authorized Wake development.

| Run / job | Actual participants | Official physical rounds | Public outcome / own telemetry |
|---|---|---:|---|
|41962290338 /127031217035|AISurvivor1.2.1 vs Target1.0|1|AIS178/178 total share100%,1 first place;15 shots/11 hits (73.333333%),105.825571 own damage/0 taken,251 turns,4 walls/0 contacts/0 skips; own maxTurnMicros2929|
|41962290605 /127031218100|AISurvivor1.2.1 vs Retcon0.2.0|1|AIS18/177 total share10.1694915254%,0 first places;17 shots/3 hits (17.647059%),16.866533 own damage/82.4 taken,270 turns,0 walls/1 contact/0 skips; own maxTurnMicros2270|
|41962290505 /127031217276|Alpha1.0 vs Beta1.0 stationary probes|1|Official scores181/2, Alpha1 first place; trace status verified; instrumentation/clock probe only|

Each official battle.json declares numberOfRounds1. Both AIS metrics streams have one round1 row, eval mode, checkpointLoaded=false, totalRounds0, seed20261102; original result summaries declare trainingRounds0/modelSHA256 null. Separate per-game sample SD is undefined (one game in each category). No candidate firing/effect/cost inference is possible. These are not admitted Wake evidence and did not undergo Wake's registered pregame gate. Observed complete official rows and zero AIS own skips do not retroactively supply that gate. Retcon-private stats/log payloads were retained as opaque original bytes and hashes, not interpreted.

## Actual commands, source and runtime

Original job-step APIs and complete provider job logs show actual headless smoke command `python tools/benchmark.py --smoke --timeout 90 --require-metrics`, Retcon smoke `python tools/benchmark.py --suite retcon --smoke --java "$JAVA_HOME/bin/java" --timeout 180 --require-metrics --require-retcon-stats --output build/retcon`, and `python tools/engine_energy_trace.py --java "$JAVA_HOME/bin/java"`. No FULL branch or extra training scenario was selected. Engine steps completed03:34:27–03:34:33Z,03:35:00–03:35:16Z and03:33:44–03:33:55Z respectively; all jobs are completed SUCCESS.

Default checkout in the two production smoke jobs measured synthetic merge `818ca64a609463b7a2aa5241cce8f9e0bd68b5bf`, whose independently fetched actual parents are master `3b733e93b1da5ff2d999fb6b3870273d411b2814` and proposal `48bd8e...`. Trace explicitly checked out48bd8e. Both production result artifacts report unchanged AISurvivor JAR SHA256 `b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800`. Actual source builds/parity under Java17 preceded Retcon Java25 installation, per original job steps/logs. Runtime result strings report Temurin17.0.21+9 for samples and25.0.5+8-LTS for Retcon; trace reports17.0.21+9. Generic workflows selected Java17/25 and Python3.12 (actual trace library path3.12.16), not Wake's fully frozen runtime; no Wake runtime clearance is granted.

All three jobs used ubuntu-26.04/x86_64. Exact actual startup `Runner Image` groups (not the distinct `Runner Image Provisioner` group) give provider version20261025.179.1 for the two production jobs and20261101.186.1 for trace. These are provider observations only: native-before-tools and separate ENV-case evidence were not captured as Wake would require. Do not claim qualified/canonical Wake image admission or validate historical workers from these runs. Full original logs and run/jobs/source APIs are retained, not reconstructed excerpts.

## Original artifact custody

| Artifact | Original ZIP bytes | Whole SHA256 |
|---|---:|---|
|13386320688 battle-results|4127|48d18f5f02c3ca7ef53f076af0c7c51a33881c0290c9ee541e0d75bf6e8e2c91|
|13385374817 retcon-results|7097|2afec5c7aeac09b6a9601dcaed5b443c7c1b2aa47a87ddf6ea803211e545406d|
|13385772686 pinned-engine-energy-trace|7509|6d470f79708a631cf5ac06a635426e7cb7990d6725e7f6a73ee8fb0807f6c958|

All original ZIP hashes equal the actual artifact API digest; all19 regular members were completely read for CRC/whole-byte hashing and preserved losslessly, including opaque opponent entries. Structured analysis used only official results, own AIS telemetry and purpose-built Alpha/Beta trace. Whole job logs are140621/152791/29085bytes with hashes recorded in original-custody.json. ZIP/member inventories and complete current APIs are in this directory. A first ZIP-fetch command failed on gh binary terminal-escape protection before writing bytes; the explicit binary-to-captured-memory/file correction is recorded and did not launch/retry gameplay. An initial human inspection matched the distinct provisioner header; the retained final exact-group parser requires a line ending precisely `Runner Image` and makes no provisioner-as-image claim.

## Prospective narrow repair method review

I independently fetched unedited issue118 comment6649074233 by stephanelee, created03:42:42Z. Exact comment body SHA256 `2ede376b3f61f79af9a04acd4dfbb02983d2c92da1dfb32209ef3eb94c6787da`; embedded draft bytes SHA256 `11c1d0a316078c89013b2bb26745d7fa285f66d45e5c63c6b8d49b9fdbfb4ed3` match the separately preserved draft. **METHOD-PASS only for implementing the specified future literal Wake PR exclusions and publishing one combined guard/amendment/evidence update.** No numerical criterion, population, pin or receipt changes; no automatic/manual replay; no invented successful statuses; preserve actual SKIPPED job conclusions and active genuine zero-game CI. Existing global master-push validation is disclosed separately and cannot provide candidate/control evidence. Any new accidental acquisition stops experimental admission for reconciliation.

Prospective fixture-scope clarification, requested by root before implementation and before any Wake outcome: the original no-extra-"numeric fixtures/battles" wording is read with its explicit required pure contracts and full-reader fixture requirements. ONLY required pure, no-engine semantic/arithmetic/evidence-shaped fixtures are allowed. No engine-backed gameplay fixture, smoke/development match, parameter search or physical acquisition beyond the registered fixed games is allowed. This is an explicit future-only clarification of mandatory zero-game validation, not retrospective permission for the three CI rounds. It must be included with the published amendment so the allowed test scope is reviewable.

This is not yet final repair acceptance, candidate implementation approval, source/JAR review or dispatch authority. The exact next published head must preserve this complete evidence, ledger/checkpoint failure record and original protocol hashes; change ONLY the three future branch guards as specified, then actual new-head jobs must be observed SKIPPED with no engine steps while legitimate required checks pass. Independent exact protocol approval remains held until that publication/guard verification. Baseline/AIS17-file Gitblob identity and32 unique registered IDs/seeds/ports have independently passed; all22 published registration files matched actual remote bytes, but that coherent prospective text does not erase this preparation failure.

Hourly automation remains PAUSED. No review here grants any competitive claim.
