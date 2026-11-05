# Independent review of explicit metadata-reader recovery

Reviewer `/root/runner_image_provenance_audit`, separate from outcome reviewer/adapter author `/root/brace_independent_protocol_review`. **PASS for the actual read-only metadata/custody recovery described here.** This does not claim the original frozen full decoder passed, change a numeric gate/decision, authorize confirmation, or classify gameplay success.

Exact inspected adapter `provenance_path_selection.py`:2463B/SHA45039bb4c91b9ee8b0948b5906cec65f331419f335330c5a9543767f14069c9f. Exact separate fact script `independent_final_metadata.py`:8176B/SHA99dca114aa52333f03fb7a632c4d7e39b6f49f42b33095b8a938c4deb4c2d0c2. Both assert actual frozen decoder source8cef410dc20c0af7b317cf0de62c8cf65d65d10b559ea44145e00c69710e2cd5. I read them completely and independently reproduced the relevant frozen-function checks without running their output-overwriting entry points or modifying originals.

## Actual capsule path selection

The actual final ZIP contains BOTH stage/profile.json and stage/provenance/profile.json with identical2722975 bytes/SHA b670725af5deb0e711dfa51f4ead24c24066ff65e29d5a1675977966438119f8. Actual freeze, effective-freeze-origin, image-identity and source files are under provenance/. Stage/freeze.json does not exist. The frozen full reader picks the duplicate stage profile first and incorrectly infers the capsule parent; this is a consumer path integration limitation, not evidence of missing original capsule bytes.

The inspected adapter maps only the profile.json lookup to the actual original provenance/profile.json. It checks byte identity, exact frozen source and ZIP hashes, delegates all other lookups to original stage files, does not mask existence and neither edits nor synthesizes original members. I independently applied that same explicit lookup and called frozen verify_exported_provenance: PASS, with the actual source/freeze/receipt anchors. My separate completed image audit already compared ALL260 original pregame capsule members and actual cached metadata byte-for-byte in the final ZIP.

The original first frozen machine report remains1723482B/SHA cac27489c73908949c5eebe1c4be84cf431720ed8b2e4ded5fbdaca4ec16f7ab. The path-selection-only supplement remains38406B/SHA9eec5bc3b1647d558403f3e817e650777f868a9880a0dbfc7e464c69562d8c20 and still reports runnerImage PROVENANCE-INCOMPLETE. Neither is relabeled a full PASS.

## Actual admission chronology, without altered timestamps

The frozen producer records checkedAtUTC at ENTRY to revalidate_authority, then performs image capture/full graph/authority checks. The frozen consumer at audit.py:1284 instead requires capture.utcEnd<=authority.checkedAtUTC, treating that entry timestamp as an already-completed admission deadline. Its description does not change the producer's actual field semantics. These two interpretations are incompatible for actual sequential collection.

I independently compared all15 ORIGINAL capture reports, authority records and original admission records: before-receipt uses the immutable consumed tagger.date02:08:01Z; each of14 child phases uses that child's original attempt.json startUTC. Every capture ends BEFORE its actual receipt/child admission, with margins **7.068823–8.644107seconds**. Capture end is **0.078816–0.084616seconds after** validation START, exactly consistent with source control flow. No clock-skew allowance, derived replacement timestamp, overwritten checkedAtUTC, copied fake record or altered input was used. The receipt comparison even passes against the exact recorded second itself, without needing a second-resolution extension.

I independently called the frozen audit_image_snapshot function on all15 originals, compared stable metadata to original prepared, and cross-checked the outcome reviewer's separate fact rows to my original-field/member/hash proof. Full raw/stat/parser/cap/policy/source/boot/case invariants remain enforced. The source method requires capture before actual receipt/child admission; the original evidence satisfies that requirement. The separate proof corrects interpretation of an integration field, not a numeric or physical-round criterion.

Own exact per-phase proof `original-before-admission-temporal-proof.json`:21427B/SHA bca0f002b5120f7c7a95d190b0b23760c8c205d58c9a3c07eca5fe39c330ec36. Own independent recovery result `independent-readonly-metadata-recovery-review.json`:1357B/SHA9b23d4414b6b87a50bf28bbe981854fa52535edc740e764ee64bf237d12d4598. Outcome reviewer's actual separate fact output8906B/SHA d84d538cdbf88cd63cae342a011cc0ff65c8a8acf547f4ec601982f1749bad79 was independently checked; its correct attribution must remain that reviewer's own work, distinct from my final provider-image record06af54b2….

## Scope and preservation

I verified original failed machine reports, adapter/script and supplied fact output stayed unchanged during independent replication. The shared original ZIP was only opened read-only; no battle/decision/source bytes or timestamps were rewritten. Final image/provider reconciliation is separately PASS from original complete APIs/header and all15 image/authority phases. These supplements must remain explicit alongside both original machine failures; no statement that the whole original frozen parser succeeded is supported.

This review supplies metadata/custody facts only. The outcome reviewer owns raw numerical/manual checks and final disposition, with unchanged registered gates and no confirmation/promotion permission from this document. Future consumer repairs should use these authentic integration cases and preserve the published old source/results, rather than hiding failures or rerunning physical battles.
