# Frozen Keel-Prism descriptive60-round comparison

[Protocol and administrative history](../../docs/improvements/keel-prism-v1.md), [issue109](https://github.com/stephanelee/robocode/issues/109), [PR110](https://github.com/stephanelee/robocode/pull/110).

Keel official score4221 vs Prism3394; official first places47 vs13 (ties are possible). Equal-game Keel mean score share56.2926886419%;4/6 game shares above50%, range37.1681-74.3007%. This is one descriptive six-game cold-process head-to-head on local Windows/Java17, not a tier gate, general superiority claim, confirmation or pooled Ubuntu result. Keel shots3642/hits504 (13.8386%); Prism4394/408 (9.2854%). Official bullet damage978 vs2698; shot energy2018.1967 vs6479.4818. Event hit damage and official credited bullet damage differ because official credit caps damage; retain both. These bundled results do not isolate a causal feature. Both private skip totals are zero, sourced from Keel markers/Prism cumulative telemetry, not observer events.

## Immutable phases

* `invalid-startup/`: source1368692, spent originalreceipt,0 games/0 rounds. Windows1252 author identity mismatch; all attempted startup/raw/profile/registration retained.
* `partial-utf8/`: sourcefa14ee2, separately reviewed UTF8 registration. Game kp-utf8-01 physically completed10rounds, then serializer collided rounds fields. Original report remains INVALID. This complete recording is parsed offline, never replayed. Exact partial archive hashes in SHA256.json.
* `completed-remaining/raw/`: sourceab418c6, separately reviewed exact FIVE previously unplayed IDs02..06/original seeds/order,50rounds. New permanent remaining50receipt. No replacements or extra games. Source17 executable/modules/runner/settings unchanged; only field name serialization fix and phase roster/receipt.

`combined-report.json` combines those actual10+50 physical records transparently, never rewrites either invalid report. Each game includes official cumulative endpoint differences for per-round score/first places/bullet damage, event shots/hits/shot-energy, last-observed energy and complete turn counts. Dead participants can be absent from later ticks: lastObservedEnergy is not certified death/final energy. All six games have ten private round snapshots plus final Prism game counters, and ten Keel start/end markers. In-memory learning resets with every new bot JVM; neither bot persists checkpoints, so there is no saved-model endpoint claim.

Registered profile sourceCommit fields identify generation commits (invalid-startup590aa2; UTF81368692; remainingfa14), NOT execution heads. Actual OUT/runtime, attempts, receipt and report identify execution136/fa14/ab418 as applicable. `runtime.json` freezes actual local JDK files, options, harness/plan hashes and Windows/Python environment; no silent runtime switch. `SHA256.json` hashes every archived file except itself. Bot JARs remain original77b8/8c045; Java17 clean builds/tests passed10/39 and entry parity788/855 before execution. Accepted source is masterd76 and no controller/descriptor/JAR changed.

## Offline reproduction (zero games)

From repository root: `python -m unittest discover -s tools -p test_keel_prism.py -v`. It verifies archived byte hashes and recomputes every completed raw recording and combined totals. The acquisition entrypoint is spent: both local-output exclusivity and permanent remote receipt prevent replay; never remove/reset them. No further game, optimization, tuning, replacement or confirmation is authorized by this report.

Exact executed harness bytes are archived perphase under executed-tools with provenance.json: original136 uses Git bytes; laterfa14/ab418 Python script Windows CRLF expansion matches runtime.assets exactly. This is newline normalization only, not a logic change. All source files can be reconstructed from the named Git objects and this explicit transformation.
