# Independent KeelHelm registered screen outcome audit

**PASS: the registered screen opens confirmation eligibility only.** All14cold games/120physical rounds are valid; every prospectively registered screen gate passes. No confirmation game, stage activation, merge, promotion or broad-strength claim is cleared by this outcome report. Root coordination is required before posting the machine-readable eligibility record.

## Independence and preserved originals

I did not implement or execute KeelHelm or its harness. I used the standalone standard-library `benchmarks/keel-helm-v1/independent-analysis/audit.py`, importing no candidate, benchmark harness or harness reducer. Its source SHA256 is `1a1c8e3de7a6f05e8b2cac987665d3b0ce7d3c9772e502b427df2dd615f1984a`. The parser was written before these outcomes and separately verified against existing historical Keel recordings; those historical games are not pooled here. I read only the finished registered screen, official observer schema, private markers and preserved metadata. No game, motion helper, source change, retry, label, receipt reset or stage action was performed by this audit.

- Sole screen run:41874753112, attempt1, exact execution head`368b49a9bae42a0cefd4fd9aa033a6bd576e566d`. Provider completed/success; exactly one registered-stage job. The workflow history currently contains exactly one KeelHelm run.
- Original finalized evidence artifact:13349356945, 117,899,908bytes, SHA256`8c49b360f4a442b3f241734ce0e4b662faabb8e44bd986da2a080cc7cbbbc14b`. Preserved unchanged at`/private/tmp/keel-helm-screen-41874753112.zip`. All1,349regular ZIP members were safe-extracted and independently compared byte-for-byte.
- Complete decoded job log:`/private/tmp/keel-helm-screen-41874753112.log`, 128035bytes, SHA256`10abd3c23bd74a1c6cb3debbeca923e1c892121674bfc78ba70f22568ac88f52`. It includes provider-masked token notation, not exposed credentials.
- Raw screen decision:`/private/tmp/keel-helm-screen-41874753112/robocode/robocode/build/keel-helm/screen/decision.json`, SHA256`bb4a2e40d438c25ff6f82083de26a6c90954325f2789d4c63ce4ee457ce8bb0c`. It remains byte-identical; this independent report does not replace it.
- Full recording manifest:`/private/tmp/keel-helm-screen-41874753112-recordings.json`; all14original complete gzip files,95,814,904bytes total. Each raw official file hash matches the unchanged original decision.

## Raw validity, physical budget and decision reconciliation

Independently replayed all official GameStarted/Tick/RoundEnded/GameEnded records: exact1v1/800x600/100000microsecond timeout, registered participants/versions, unique IDs, every tick contiguous from1to the endpoint in every round, exact unique round endpoints, and final/round cumulative scores and official firstPlaces. All120rounds reconcile. Every completed round has exactly one ordered start/stats/end; both arms have zero private skip markers, valid nonnegative integer maxMs<75, and required candidate-only exposure counters. Incoming shots are nonzero. AIS games contain actual complete cold TRAIN mode/seed/checkpoint telemetry and stored model round-count evidence. Keel persistent-model hashes are correctly N/A.

Actual game population/order/seeds/ports exactly match the frozen14-game registry. All14attempts exist, all120physical rounds were observed and completed, no invalid/partial game, no missing or unplayed ID and no hidden failure.json. Every child has an unchanged prepared command/cwd/environment, fresh process/writable-directory/no imported-model declaration, and normal teardown with neither TERM nor KILL sent. No observed outcome was replaced.

Compared2,630shared per-game/per-round raw-decision fields with the independent replay: **zero mismatches** (official scores/wins/share, shots/hits, uncapped damage/energy, contacts, turns/firing rate, power mixes, skips/timing/exposure and recording hashes). Means and percentage-point gates were independently computed with exact Fraction arithmetic. The original decision is`progress; confirmation open`; all its checks and the independent checks pass.

## Per-game measurements

| ID | Arm/opponent | Share | FirstPlaces | Own hits/shots | Enemy hits/shots | Damage dealt/taken | Shot energy own/enemy | Walls/rams | MaxMs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|s01|baseline/diamond|43.173432%|7/10|256/2679|293/2951|145.600000/547.398215|487.588296/1170.226809|3/0|8|
|s02|candidate/diamond|48.826979%|8/10|300/3203|386/3549|163.200000/558.002579|495.143104/1200.983458|4/0|11|
|s03|candidate/diamond|57.598784%|9/10|278/2983|370/3213|189.911441/483.985304|555.670388/1125.046298|5/0|13|
|s04|baseline/diamond|56.157270%|8/10|267/2848|315/3068|183.600000/516.416539|542.992338/1130.001802|1/2|10|
|s05|baseline/diamond|46.993318%|7/10|269/2997|361/3417|184.400000/492.213614|570.437554/1136.199247|9/0|12|
|s06|candidate/diamond|69.261637%|10/10|239/2667|264/2891|222.801675/383.309336|622.706060/1033.038320|3/0|9|
|s07|candidate/retcon|96.468401%|10/10|99/651|8/616|373.780000/38.080000|584.158333/647.720000|11/0|7|
|s08|baseline/retcon|95.383195%|10/10|97/629|10/604|368.160000/50.320000|577.320000/672.400000|0/0|10|
|s09|baseline/retcon|98.127341%|10/10|106/594|4/583|385.880000/20.240000|533.970000/636.840000|3/0|10|
|s10|candidate/retcon|95.676692%|10/10|94/642|10/594|357.200000/45.920000|572.193333/681.560000|6/0|4|
|s11|baseline/walls|94.659091%|5/5|63/95|4/18|441.100000/40.000000|108.116667/36.000000|3/9|8|
|s12|candidate/walls|93.799324%|5/5|78/124|5/24|438.500000/50.000000|120.400000/48.000000|2/7|11|
|s13|candidate/ais|89.984825%|5/5|92/290|12/240|248.900000/65.600000|199.708333/300.200000|1/0|11|
|s14|baseline/ais|92.735703%|5/5|84/275|9/237|254.500000/46.800000|199.808333/282.300000|3/0|3|

All per-round values, firing rates, hit rates, energy and contacts are retained in`/private/tmp/keel-helm-screen-41874753112-games.csv` and`/private/tmp/keel-helm-screen-41874753112-rounds.csv` (120rows). Exact raw JSON outputs remain distinct. Bot contacts count official events with botId equal to own ID; bullet collisions count official observer collision events.

## Registered screen gates

| Check | Independently measured result |
|---|---|
|diamond candidate/control equal-game mean|58.56246697% / 48.77467342%; delta+9.78779355pp|
|retcon candidate/control equal-game mean|96.07254661% / 96.75526783%; delta-0.68272122pp|
|walls candidate/control equal-game mean|93.79932356% / 94.65909091%; delta-0.85976735pp|
|ais candidate/control equal-game mean|89.98482549% / 92.73570325%; delta-2.75087775pp|
|Diamond absolute/delta/win/game guards|58.56246697%>50%; +9.78779355pp>=3pp;27/30>=16;2/3games>50%|
|Retcon absolute/delta/win guards|96.07254661%>=90%; -0.68272122pp>=-3pp;20/20>=16|
|Walls guard|93.79932356% vs94.65909091%, -0.85976735pp>=-5pp;5/5>=4|
|AIS guard|89.98482549% vs92.73570325%, -2.75087775pp>=-5pp;5/5>=4|
|Both-arm timing/integrity|120/120valid rounds;0own skips; maximum reported integer maxMs13candidate/12baseline, all<75|

Candidate Diamond game shares range48.82697947–69.26163724%, sampleSD10.25135712pp; control43.17343173–56.15727003%, sampleSD6.67270060pp. One candidate game is below50%; that does not fail the registered2of3gate. Retcon shares have candidate/control sampleSD0.55982334/1.94040424pp. These are cold-game descriptive variations, not independent-round inference or an unregistered significance test.

## Mechanism diagnostics and limits

Candidate Diamond exposure counters total9,952,343stopped rollout steps,1,946,628heading-changed steps,1,627,219second-wave evaluations and11,786executed stop-now choices. Baseline exposure is N/A by registration; no exposure minimum was used as a strength gate. These counters establish activation of the changed forecast, not a directly observed causal contribution or count of changed final route choices.

Across Diamond games, candidate dealt575.913117/took1425.297220uncapped bullet damage, vs control513.600000/1556.028368. Candidate pooled own hit rate9.228510% vs control9.291412%; incoming hit rate10.566663% vs10.269182%. Candidate therefore did not demonstrate a lower pooled incoming hit frequency, despite lower incoming damage. Both arms still lost the bullet-damage exchange. Token0.1shots were8109/8853candidate and7843/8524control; score gains remain compatible with energy attrition/survival scoring, not proof of aiming superiority. These pooled descriptive rates do not replace equal-game score means.

Diamond ram-event counts were2control versus0candidate (and Walls9control/7candidate). The baseline's known frozen ram-energy bookkeeping limitation therefore cannot be assumed inactive in this screen. It was not changed or used to discard any game. These contacts do not prove their effect on the score difference.

Retcon candidate damage730.98/84.00 vs control754.04/70.56, and candidate20/20wins does not erase its small score-share decline. General transfer is not established by two ten-round games. The registered fresh confirmation remains necessary: six20-round Diamond games per arm plus three10-round Retcon games per arm,300new rounds total, and unchanged candidate/source freeze/gates. Do not pool the screen with confirmation or historical Keel evidence.

## Runtime, approval, source and receipt provenance

Actual runtime: Ubuntu26.04/x86_64 image`20261026.180.1`, CPython3.13.17, Temurin17.0.21+9build/tests/JAR parity first, then Temurin25.0.5+8-LTSbattles/Retcon. The original job log and API step chronology agree. Frozen runner1.4.0 SHA25602f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf, pinned DiamondPort1.8.28/Retcon0.2.0 and approved sample/AIS identities reconcile.

- Source freeze SHA256`ef876a82069d5c73baf7171aa95cf4058e3bc4a196f46d51bdf5ff1cf0b81a9b`; actual full same-worker profile SHA256`63f9ff12175b39216e160aaefb6668a60c1781933d46b1904c356fea2e778095`. All exported frozen source/JAR bytes and full profile/manifests reconcile. Original Keel JAR77b8a5c3… and candidate60ef5b1d… match frozen participants; immutable baseline and production AIS hashes independently reconcile across16retained files.
- Actual boot`7bd7cace-1b34-4348-ad11-a8cfb7c1f6fb`, workspace`/home/runner/work/robocode/robocode`, exact run/attempt/head agree. Pre-game provenance artifact13345154116 SHA2569aa2f9ed04dabec39d4737d823b7cc66ce5097961641a1bbd220cc1aa102cd04 is distinct from the final evidence ZIP. Its original download/review belongs to the separately assigned runtime reviewer; I independently checked its unchanged provider metadata and archived bindings, not a duplicate download.
- Actual dispatch comment6636942994predates run creation. Separate runtime report6637228566and exact20-key approval6637230264postdate finalized artifact and precede gameplay, within30minutes. The assigned reviewer task/root tool provenance is separate from implementation and this outcome audit; a shared GitHub account alone is not the independence evidence. Current provider API identities/body/timestamps/author agree with original accepted copies.
- Fifteen archived live authority revalidations (before receipt plus before every game) retain valid unchanged dispatch/report/approval, artifact, source-head clearance and exact permanent receipt.
- Fixed remote receipt`refs/tags/experiment-receipts/keel-helm-v1-screen` is the original annotated tag object`5add0bbade3239772c0350a8b0bf40a6b06e0e41`, pointing at exact screen head. Its remote JSON message equals archived receipt and binds approval/run/boot/profile/freeze/population; it has not been reset.
- Independent provider/authority/receipt/command/budget audit passed1341checks, including1074retained actual staged input-file hashes. Every actual child start followed runtime approval/authority/receipt, matched fixed order and retained at least17902.083seconds (>35minutes) of job budget. Overall job stayed below340minutes; all owned processes had normal teardown.

Limits: an offline audit cannot rehash unexported live native/JDK/Python filesystem bytes or recover RAM. The reviewed same-worker gate performed those rehashes; this audit independently verifies exported source/inputs, complete archived manifests/authority snapshots, original provider metadata and logs. Diamond private skipped turns remain unobservable. Uncontrolled starts/opponent randomness and correlated within-game learning preclude a paired or general-strength claim. The13/12ms values are the frozen logged integer maxMs measurement, not a comprehensive latency guarantee.

## Disposition

**All registered screen conditions pass; conditional confirmation eligibility is supported.** This is not confirmation dispatch or actual worker approval. Root must first retain full original raw files/decision and source freeze, coordinate the eligibility comment, publish unchanged evidence, obtain new exact-head review and actively assign a separate runtime reviewer before any confirmation dispatch. No future game, retry, source/runtime change or merge is authorized by this report alone.
