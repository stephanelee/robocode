# Independent actual zero-game probe review

Reviewer `/root/runner_image_provenance_audit`, separate from implementer/root. **PASS for diagnostic custody and registered question only. No canonical-image, runtime, source-freeze or game admission.** No source edits, dispatch, rerun, label, receipt, ref, commit or merge performed by this reviewer.

Run [41940635533](https://github.com/stephanelee/robocode/actions/runs/41940635533), attempt1, pull_request, head1fd0e78bf36e8021657ca32240d66137731bfc77, created2026-11-05T00:20:47Z, completed success at00:20:57Z. Sole job126960225659. Independently fetched official run/jobs/artifact JSON, full unmodified provider job log and original artifact ZIP into `/private/tmp/keel-brace-image-probe-41940635533/`. Artifact13376875968, name keel-brace-image-probe-41940635533: **33858B SHA256 d7becc8e2819bda17b535f483e244828c463fc453116bfd91f992a95b32e84fc**, identical to artifact API digest/size. Full job.log: **55945B SHA256 cc9a989d1d4d98914dc331354e7aae3f6cfaea3ef51ec537b4ddf9cbf463f1ce**.

The original ZIP has26 members,115944 uncompressed bytes, safely extracted into original-members. Every indexed payload hash/size in both capture statuses independently verified (12 per phase), including both exact collector-source copies SHA520a6b5aace934939464338b45ad07f01c95d81648e29bd2e2393a5897c7ca94. Both reports complete, diagnosticErrorCount0, no parse/read/discovery errors, no prefix/truncation. The actual PR merge workflow605385e89b3a5e6e63c560a04b213c3ce07b1502 independently fetched from GitHub has SHA0680c78000a8a3425384764fc8d5e5da412ce2a2f1b4a54154fb594412bf720e, identical to reviewed published source. PR head is separately recorded as1fd0e78..., not confused with merge workflow SHA. Prior prospective source review/registration remain the scope authority.

## Registered question answered

**Disagreement already exists before tool setup, and all captured metadata remains unchanged across the setup interval.** This rules out those later setup-java/setup-python steps as the origin of the already-existing conflict in this worker. It does not identify the original cause or authenticate a canonical VM build revision.

Both snapshots have the SAME boot ID5cad7169-95ec-4b8a-9e88-7b4a1a0493ce, same discovered runner root `/home/runner/actions-runner/cached/2.338.2`, architecturex86_64 and kernel7.0.0-1017-azure. Before snapshot00:20:52.437762–00:20:52.442373Z; after00:20:52.905740–00:20:52.910071Z. Both use system /usr/bin/python3; after reports both setup outcomes success. Every one of seven captured raw-file byte strings is identical across phases, independently checked.

| Named source | Exact observation, both phases |
|---|---|
| Process ImageOS / ImageVersion | present ubuntu26 / present20261026.180.1 |
| /etc/environment ImageOS / ImageVersion | present ubuntu26 / present20261026.180.1 |
| /imagegeneration/imagedata.json Runner Image | image ubuntu-26.04; version20261025.179.1 |
| Discovered runner .setup_info Runner Image | image ubuntu-26.04; version20261025.179.1 |
| Provider Set up job Runner Image | image ubuntu-26.04; version20261025.179.1 at00:20:51.2289721Z |

Both file claims and provider header use Included Software `https://github.com/actions/runner-images/blob/ubuntu26/20261025.179/images/ubuntu/Ubuntu2604-Readme.md` and Image Release `https://github.com/actions/runner-images/releases/tag/ubuntu26%2F20261025.179`. This is corroboration of the **provider-reported claim**, not three independent attestations of VM contents; runner output originates in setup-info. OS metadata reports Ubuntu26.04.1 LTS.

Raw SHA256/bytes, identical before/after:

- imagedata:6153275ed304a83a64222eee33acd6512a0493912bb780eea54ff6a2004d9a97 /398B;
- .setup_info:c7495c43bd92de56f1c17d7c7e00d9f9d114c181ecfbea56172abf1dd44ba921 /751B;
- /etc/environment:ce4fb62ae47ff50c623bddda77c371111e43e0e7424a723cf0d88b1907817eb9 /2243B;
- /etc/os-release and /usr/lib/os-release:cf72627ff81aef345d0a9f3807eb80f1035e0f80fe2932d12d94efc02fd68104 /410B each;
- /etc/lsb-release:77b958705ba7401d2ad5706c7bb0c4e9d999975ee8e5d667507529b199b5f2c3 /107B;
- boot_id:3dde0a782ba27bbe51d6694918a815af32297190f88295eadd2440bee1ca7e6d /37B.

Provider resolved actions: setup-java@v4 **cf277c60eb25467037889841efdb72551f06f6c3**; setup-python@v5 **a26af69be951a213d495a4c3e4e4022e16d87065**; upload-artifact@v4 **ea165f8d65b6e75b540449e92b4886f43607fa02**. Action repositories are downloaded by runner preparation before first actual step; the diagnostic claim is before *tool setup execution*, not before provider action preparation. Setup-java log states17.0.21+9. This job performs no Gradle/JAR execution, source parity build, Retcon source launch or game; it does not establish the future execution graph.

## Custody limitations

The first log-fetch command failed because gh refused terminal escape sequences, before saving output; a repeated failure/stderr is retained as job-log-attempt2. `gh api --allow-escape-sequences` then retained the exact completed log. These are CLI output refusals, **not HTTP404**. I first inspected the run after it had already completed; this probe establishes no evidence of in-progress full-log accessibility. Per-call UTC fetch times were not separately recorded; retained local file mtimes and subsequent manifest-generation timestamp bound local custody timing, not provider events. Original provider event times come from API/logs. No artifacts were redownloaded for retries; ZIP downloaded once and kept unchanged.

The earlier official-source audit could not resolve the exact Ubuntu tag/release URLs (404) or establish an authoritative Ubuntu mapping. This probe supplies previously missing raw same-worker metadata and excludes later setup steps as origin, but does not supply that mapping or provider VM attestation. The Windows maintainer explanation of re-specialization remains a plausible analogous class, not proven cause here.

## Defensible prospective policy options

**A qualified exact observed-runtime policy is methodologically defensible for FUTURE Brace, subject to separately published source/protocol review before games.** It can explicitly pin the precise named-source tuple above, ubuntu-26.04/x86_64, OS/kernel observations, exact Java17 build/tests/committed-JAR parity first, exact Java25 Retcon only after Gradle, Python/opponent/wheel/native execution graph, commands and candidate/control bytes. Freeze full actual same-worker source/runtime/raw-file evidence and independent approval before outcomes; rehash without repair before each child, reject every unexpected tuple, missing/ambiguous field or mutation. All arms share that approved worker. Keep all numerical budgets/rules and source restrictions unchanged.

That establishes an operational runtime with **qualified provider-revision identity**, not a content-addressed whole VM or canonical image-build revision. AGENTS requires ubuntu-26.04, not the old protocol's chosen revision-authority rule; therefore a clearly stated future-only amendment is possible without changing history. It must explicitly replace the NEW Brace revision interpretation prospectively and name the admitted tuple, not conceal a broadened allowlist or substitute ENV for metadata. Old Helm remains UNPROVEN under its unchanged literal criterion. A future stage's whole .setup_info includes worker-specific material, so do not blindly require equality to this probe's whole-file hash across different workers; pin claims first and freeze each actual stage's complete raw-file hashes before games. No score-dependent tuple selection or retrospective profile acceptance.

For the strongest proposed version, **independent actual same-job external provider-header review before games** must really occur within the admission deadline. Do not treat the completed diagnostic header as another worker's header. If stage log retrieval is404/unavailable before completion, this version must stop with zero games. Concrete missing resource is provider access to that active job's startup log/header during the pregame window; a completed zero-game probe or separately hosted job cannot provide it for a future worker.

A different prospectively reviewed policy could make the captured actual runner .setup_info the pregame authority (source-backed as the information runner prints), bind its original bytes to the pre-outcome immutable upload/reviewer decision, and require final external provider-log reconciliation before classifying outcomes valid. That is **not** external provider corroboration before games; it accepts a weaker custody boundary and must be named explicitly rather than used as a silent fallback. I am not approving that alternative by this report.

If the desired claim is instead “canonical provider VM build revision verified,” additional authoritative GitHub hosted-runner/provider evidence is essential: an explanation/attestation binding this exact Ubuntu tuple to the provisioned revision, or consistent trustworthy provider metadata under a new prospective protocol. The observed tuple and executable hashes cannot make that stronger claim. No provider attestation is inherently required merely to report a precisely qualified operational-runtime comparison, provided the chosen future methodology explicitly permits that scope and all before-outcome controls actually pass.

This report endorses no actual future stage/profile. Root's protocol owner must publish the exact amendment, the independent reviewer must inspect source/enforcement and actual stage evidence, and runtime/game gates remain closed until then. Physical game budget consumed by this diagnostic:0. Historical results, criteria, raw records, published SHAs and consumed receipts remain unchanged.
