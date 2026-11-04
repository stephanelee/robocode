# Independent static implementation review: STOP24

**CHANGES REQUIRED before the payload/profile freeze and preparation clearance.** No SDK/predictor imports, Math calls, bridge main, numerical cases, tests or games were invoked. Reviewed `tools/diamond_stop24/{prepare.py,evaluate.py,loader.py,DiamondStopBridge.java,source_pins.json,README.md}`, `tools/diamond_stop24_host.py` and `.github/workflows/diamond-stop24.yml` against registered issue99 protocol/cards.

## Required correction R1: prefreeze Python dependency closure

`current_profile()` binds the base Python executable, owned venv inventory/package RECORD bytes and tested SDK/port files, but does not bind the base standard library, native extension modules or dynamically linked Python runtime before acquisition. `ClosureGuard` permits any import resolved beneath `guard.stdlib`; `inventory()` records hashes (and cached-bytecode hashes) only after imports. Thus a changed stdlib source/pyc/lib-dynload module can execute during evaluator startup/tested module import before its bytes have been compared with an independently frozen dependency proof. A binary-only interpreter hash and same image label do not bind those files.

Add a zero-case metadata-only proof of the actual base standard-library/native-runtime closure, then include it in the portable frozen profile and host/receipt binding. Verify those exact bytes/readable paths again before the numerical child starts; require actually loaded stdlib/extension paths and source/cache hashes to belong to that frozen proof. Account explicitly for interpreter-used frozen/builtin code (binary-bound), permitted source/bytecode cache policy, extension modules and libpython/shared runtime. This can use file inventories/sysconfig metadata without importing SDK/port or calling any physics. Do not make the first acquisition import inventory retrospectively define what dependencies were allowed. Preserve source/dependency failures as zero-case preparation evidence.

## Required correction R2: no persisted write credential in evaluated checkout

The acquire job has contents:write, while checkout retains its default persisted credentials. `clean_env()` correctly removes GH_TOKEN/GITHUB_TOKEN from numeric children, but the write credential would still be readable from `.git/config` in their shared checkout. This also contradicts the source comment that the API credential stays only in the wrapper. Set `persist-credentials: false` for the checkout (both jobs consistently is simplest); wrapper `gh api` already receives its explicit step token. No stage/evidence action needs a persisted Git credential. This is the same workflow-only defect corrected in external PR98 before its screen; it should not be reintroduced for STOP24.

## Static findings that pass

- Independent ZIP/source hash reads verify original six SDK module pins against wheel074f28…, both port modules, five retained engine-reference sources, nested server16c277… and MathKt.class3ed7d2…. No SDK module was imported for this check.
- `prepare.py` only AST-parses tested originals, reads JAR/class metadata, invokes javap and javac, and produces deterministic ZIP timestamps/order/permissions for bridge.jar. It does not run Java main/Math or tested Python modules. `core.json` avoids machine-specific paths; compile provenance is separately retained. Source/card/engine byte checks are explicit.
- The bridge invokes actual MathKt speed and turn helpers in the correct argument order. Old-heading translation, post-new-speed turning, exact first wall constraint speed reset and second positional-only constraint match the registered normal-turn source composition. Each constraint branch retains original candidate x/y. Java CodeSource, unique MathKt resource, actual class/JAR hashes, explicit classpath and public/static/double signatures are checked before its numeric requests.
- The evaluator invokes the unchanged original predictor once per card, then actual pinned SDK controller and clamps; Java receives hex inputs and one request per ordered card. It retains raw numbers/hex/bits, SDK intermediates, actual class/method binding provenance and whole-path intermediate positions. It executes no bot/server main. The adapter normalization is a recorded source-derived equivalence, while the explicit frozen command pair drives the reference as registered.
- Namespace shells avoid SDK/port package bootstrap; validated original source bytes are compiled directly rather than loading stale tested-module pyc. Reviewed Color exports/declarative constructors do not invoke motion controllers. Tested-module method origins are checked. The stdlib prefreeze gap above remains distinct from these correct SDK/port protections.
- Error comparison uses fixed1e-9position/speed/degree-heading and exacttime14; mismatch is a characterization result. Exactly24rows and ordered IDs are required, no hidden smoke physics calls. Exceptions retain phase/attempted ID/partial row/import failure provenance, and no partial-root resume is available.
- Hosted jobs pin Ubuntu26.04/Temurin17.0.21+9/Python3.13.18. Runtime membership is checked against the two registered existing profiles; added JDK17/tool hashes and deterministic core bytes enter the new profile. Fresh environment collection is once; repeat verification is read-only. No Java25, Retcon or gameplay launch occurs.
- Acquisition verifies frozen profile, exact PR head/token/branch, run_attempt1, boot/workspace/runtime/tool/package/core binding, absent output root and actual job-start admission. Atomic create-only receipt uses the fixed registered ref; no update/delete/reset path. After activation it re-verifies and rechecks at least10minutes remain. Timeout bounds the whole evaluator/bridge process group to300seconds (TERM then5s KILL), not per card, with60minute outer job and always-upload evidence. A consumed receipt stays consumed after any failure.

## Preparation/profile points for subsequent exact review

The profile binds all core files including compile provenance; independently check that its fresh second hosted build reproduces the exact portable/core and expected path-dependent metadata under the fixed runner layout. The planned actual profile must be reviewed before acquisition, not accepted on the fly. The added stdlib/runtime proof must remain metadata-only during preparation. No successful numerical feasibility or importer execution is claimed by static review.

Only the registered24cards may exercise original/controller/reference physics. Any discovered import/class-binding failure in the sole acquisition consumes the attempt; fixes cannot rehabilitate or replace its rows. Any pre-acquisition zero-case build/profile failure remains evidence and requires exact changed-payload review before another preparation job.

## External coordination: PR98 read-only

Read the latest returned PR98 conversation comments. The last public pre-execution comment clears only head843768f's registered screen after `persist-credentials:false`, concurrency and confirmation-byte-check documentation were added. It expressly leaves confirmation/promotion/merge uncleared. The earlier review notes procedural repeated-label exposure, hand-committed decision verification, frozen ram correction and interpretation limits. No new public outcome/confirmation clearance appeared in the returned comment list. My prior raw80round status audit remains only a screen decision/budget check; I did not alter/clear/trigger any Keel stage. STOP24 must not duplicate Keel's owned gameplay work.

Disposition: address R1/R2, then request a narrow static recheck for zero-case preparation/publication clearance. Numerical acquisition remains uncleared until separately frozen artifacts/runtime and exact published head receive independent review.

## Narrow R1/R2 recheck: PASS for zero-case preparation/publication only

The corrected source now resolves both required corrections. R1's metadata-only `diamond_stop24_python_proof.py` inventories base stdlib source, every existing pyc, lib-dynload, all links, executable/libpython and ldd-resolved native dependencies, loader cache/config and startup directory/absent paths. It records explicit builtin/frozen binary binding and rejects native preloads. No tested import or numeric helper is involved in its collection/verification path. The portable proof is included in profile.pythonRuntime, hashed into host identity/receipt and reverified by the host before child startup. The child rehashes the frozen proof before tested imports; the loader validates preloaded module paths and then source/cache/native module origins before future import execution. Prefix-only acceptance is removed. The frozen verifier source hash is also bound in core metadata. R2 adds `persist-credentials:false` to both checkout steps.

Exact source hashes reviewed in this recheck:

- host wrapper25c001cbd456e27c2b1ac16c1872e6bcab9fc4b778aa6e9f47c6d9363deaaed3
- Python proof58174fd8a7a8d5a228b1612c91be59c3219b835a2fcbfa4ee12d1811a3b93aa4
- prepare4d86868cfc7fe305e9b440ef0b89184ab42931c0a3b652dfee93ab352bc62204
- evaluate101d85fb94ec1296f49b697118d04545cf589e13647b30ae4ac9b0a97d19ebfb
- loader66674fc9d2f856cda29fd02fce71af1ceebda142f6de8cfe52e70197737c2284
- workflow3b416cf5652664ee0de1e446580827f8902cf5ae87647d67280e1a640c8cfd2a

Registered protocol f70fd73d… and input cards7e0182ab… remain byte-identical. Original review/findings are preserved above rather than replaced.

**Clear publication and one fresh hosted zero-case preparation/profile only** for these corrected bytes. Metadata collection, javac/javap and two read-only prepared verifications may run; no SDK/predictor/Math/bridge numeric evaluation is cleared. Actual hosted proof/compile/runtime artifacts still require independent audit and an immutable exact published-head freeze before any receipt/case clearance. Static review cannot establish actual loader feasibility, deterministic hosted rebuild equality or numerical agreement. No proof collection/verification, loader/tested imports, numerical tests/cases, games or workflow actions were executed by this reviewer.
