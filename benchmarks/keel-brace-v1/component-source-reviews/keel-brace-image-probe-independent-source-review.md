# Independent probe-source review — 2026-11-05 UTC

Reviewer: `/root/runner_image_provenance_audit`, independent of probe author `/root/brace_protocol_recovery` and root. Scope: frozen local diagnostic source, workflow, prospective contract and pure fake-file fixtures. **PASS for probe-source scope only**, conditional on the required exact published-head recheck before root's one diagnostic label. This is not runtime admission, image compliance, benchmark-source CLEAR, permission to consume a game receipt, or permission to change historical criteria.

Reviewed exact bytes:

| File | Bytes | SHA256 |
|---|---:|---|
| tools/keel_brace_image_probe.py | 23237 | 520a6b5aace934939464338b45ad07f01c95d81648e29bd2e2393a5897c7ca94 |
| .github/workflows/keel-brace-image-probe.yml | 31148 | 0680c78000a8a3425384764fc8d5e5da412ce2a2f1b4a54154fb594412bf720e |
| docs/improvements/keel-brace-image-identity-probe.md | 4111 | 72aa345cfb83ba1db133cb341fa482be1f151fd4c0f71f489307a23496ef25d9 |
| fixture test_image_probe.py | 16618 | 47f7a26c7d37afcfee22f086132cf73b3a6eea2db97fcce8c3d5798ea1a3a978 |
| generator make_workflow.py | 4493 | dc2815f5b7d010e0924b3b42daf4b90b85a75da4c4fc2250627f5cb3766fd718 |
| validation manifest.json | 2184 | 81b80056f0d12eee8ee6d83c3706d49b20c2f777abb13aea69c63c8bce97c88d |

Prospective [issue114 registration 6645660227](https://github.com/stephanelee/robocode/issues/114#issuecomment-6645660227) independently read through GitHub API; created and last updated 2026-11-05T00:01:43Z. It registers one attempt1, ubuntu-26.04, two same-worker snapshots, zero games, bounded collection and separate exact-head review before dispatch. No probe outcome has been supplied to this review.

## Evidence and assessment

I read the complete collector, relevant workflow/generator and all 21 tests, hashed actual files and independently ran:

`python3 -I -B -S benchmarks/keel-brace-v1/image-probe-validation/test_image_probe.py -v`

**21 pure tests passed**, 0 failures/errors, 0.024 seconds, on the local preparation interpreter. The fixtures substitute temporary metadata/process trees and mock collection; they neither collect actual worker identity nor imply hosted correctness. No Gradle, JVM, bot, server, Retcon, engine, benchmark, actual probe collector or process spawned by the tested collector was launched. No implementation edits, labels, refs, receipts, commits, admissions or merges were performed by me.

The workflow pins its sole job to ubuntu-26.04, timeout10minutes, attempt1, the exact PR branch/repository, new diagnostic label and exact-head probe-only token. It has read-only contents permission. Its first actual step writes the reviewed embedded collector and calls /usr/bin/python3 before any tool-setup step; no checkout exists anywhere. The same temporary collector pathname and system interpreter are reused after setup-java17.0.21+9 and setup-python3.13.17. Pure byte-equivalence checks establish that the YAML embedded collector equals the standalone source. Setup outcomes are recorded; always/upload and final outcome checking preserve errors while preventing failed collection from being represented as clean job success. Actions remain the existing major tags; retain resolved action SHAs from the eventual provider log, since a tag alone does not prove identical implementation to an older run.

The collector reads a finite metadata allowlist. It does not source/eval /etc/environment, inspect arbitrary process argv/environ, read credentials, invoke subprocesses, install tools or perform network calls. Runner-root inference requires observed Runner.Worker/Runner.Listener comm plus a matching executable in a bin directory; only its derived .setup_info is added. It preserves discovery evidence and errors rather than guessing a home path. Failure to discover a root is explicitly unresolved, not global proof of absent setup-info. Raw .setup_info is an authorized metadata source; this review does not assert it is provider-signed attestation.

Raw file bytes precede parsing and retain path/symlink resolution, stat identity, bounded read times and SHA256 scope. Path escape, nonregular file, symlink loop, ambiguous JSON keys/case aliases/groups, duplicate/malformed environment assignments and capture caps have explicit errors. Only complete input bytes may produce parsed claims. Per-file, per-phase and total payload budgets reserve report/status and ZIP overhead. Oversized material remains labelled a prefix with incomplete source scope; report truncation/fatal status failure yields diagnostic failure and retains existing originals. The 120second collection deadline is a collection-loop limit, not a hard timeout around each syscall; finite regular-file reads and the outer 10minute job cap bound ordinary hosted operation. This is sufficient for this diagnostic and not an unqualified guarantee against arbitrary filesystem hangs.

The output keeps process ENV, /etc/environment, imagedata and discovered setup-info as independent named observations. It intentionally permits successful collection of contradictory claims; success means observation completeness, never agreement or admission. No canonical-version selection, offset mapping, majority vote, fallback or runtime gate exists. A before-tools conflict excludes these later setup actions as its origin. A conflict arising only after setup implicates that interval; it does not isolate one action. Agreement on one worker cannot validate a different worker or either historical worker.

## Required follow-through, not new approval

Before root dispatches once, publish these bytes and the contract, then independently recheck exact published head, token and registration. This local PASS cannot authorize a different source revision. After completion retain original ZIP/digest/bytes/API receipts, full provider log and actual source/workflow refs, then compare source snapshots, both phases, boot/run identity and all separate claims independently. Missing or capped data stay unresolved. No automatic rerun or favorable-worker search.

I have not assessed or approved a future metadata-tuple policy. A fully declared observation tuple plus native runtime fingerprint might support a carefully qualified prospective operational-runtime claim; it cannot establish a canonical provider image revision or retroactively satisfy the original allowlist. Whether a defensible future policy can meet the experiment's substantive controls needs the actual probe evidence and a separate published independent review. Current Brace runtime/game admission remains fail-closed. Historical Helm criteria, raw results and consumed receipts remain unchanged.
