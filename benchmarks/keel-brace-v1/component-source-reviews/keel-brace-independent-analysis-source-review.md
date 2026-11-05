# KeelBrace independent analysis component source review

Verdict: PASS, scoped to the exact local independent-analysis component source below. This is not whole-source clearance, runtime approval, a source freeze, dispatch authority, a merge approval, or a competitive result. The runner-image provenance hold remains unresolved by this review. No games or rounds were run.

Reviewer: /root/brace_independent_protocol_review; independent of candidate, harness, CI, and analysis implementation authors. Reviewed AGENTS.md, the prospectively published issue114 protocol/registry and custody/authority contracts, actual local analysis source and fixtures, and the producer's declared data interface. No analysis/harness/bot implementation was edited. Prior prospective and component reviews remain separate immutable records.

## Exact reviewed bytes

- audit.py: ab4b8957df2f1cada0094332fac89faca80fd3fa53e98461b945ee6e127c6c6a
- test_audit.py: 67d2e9cb2ce0d5c650a970ca4f749bf38f06158a9c6a4464df41c86662094ec9
- manifest.json: 8580812ec2cfbacec7869a805d02903c283a7d98df4a4f228584492019ee5c33
- Registered protocol: cd1ec9f5bba78b47f487dc829b59d99ed52953266f15c1c3b5407186b2ad20a8
- Registered games: 9526d1a75d7ee99ac3d52e5affbf2487823b696c3d9b1e0f0497b1d42d65b3a3

All 40 files in the final manifest independently rehashed with byte lengths; PASS. Original 48-test and intermediate 55-test sources, fixtures and manifests remain preserved in verification/pre-normalized-pin-48 and verification/pre-canonical-reconcile-55. This review applies to these exact bytes, not an unverified later published head.

## Findings resolved before this PASS

1. Original accepted-comment comparison incorrectly expected full raw API objects instead of the retained normalized pin. Real producer evidence would fail despite valid authority. The repair derives the exact stable normalized pin from each original full API object, retaining whole-original digests and comparing current provider stable identity. Realistic API extras, pin mutations and current-identity mutations are tested.
2. Revocation scanning could parse foreign comments or revoke an unrelated accepted approval. Foreign accounts are now ignored before JSON parsing; different/missing approvalCommentId is ignored; malformed assigned-account input fails closed; exact-target revocation must satisfy the strict13 schema, types, accepted worker/reviewer identity, reason and later timestamp. A valid targeted revocation removes authority.
3. The raw reducer originally could report PASS after a final child produced a complete recording and then exited unsuccessfully: it did not consume the original invalid decision/failure evidence. Current reconciliation is mandatory for both audit_stage and upstream original-screen replay. It requires the exact original stage/head/ordered population, full registered physical budget, valid acquired rows, no unplayed IDs, no retained per-game or direct phase failures and no integrity/infrastructure failure. It independently compares game and per-round score/combat/timing/exposure metrics, whole recording digests, firing rates and power mixes, all exact registered gate names/booleans, equal-game means/deltas in share units, four variation groups in share units, stage-local exposure and literal canonical decision. It never rewrites the original decision. An independent raw PASS remains visible as rawDisposition but cannot produce final PASS when canonical acquisition is invalid.
4. Marker validation now requires the prior round's end before the next round's start as well as each round's unique start/stats/end order. Reversed and interleaved sequences are rejected.

The reconciler's names, ratio units, perRound fields and 14/18 ordered-population contracts were checked against the actual producer data interface. Its canonical gate mapping matches the protocol and current producer keys; no new score criterion or extra budget was introduced. Upstream original ZIP replay calls this same reconciliation after replaying all 14 screen recordings, before allowing confirmation prerequisites.

## Other reviewed boundaries

Stdlib-only offline analysis, with no candidate/harness imports or process/game launcher. Full original ZIP/member hashes and metadata, safe member names/types, CRC-streaming and split original-byte custody are checked. Registered cold-game participants/populations, official cumulative score endpoints and round deltas, firing/hit/damage/contact counts, private integer timing/counter invariants and callback-versus-contact upper bounds are checked without contact-based game selection. Exact fraction gates preserve zero-exposure INCONCLUSIVE precedence; acquisition/integrity failures prevent acceptance. Provider run/job/artifact identities, accepted authority/receipts, storage/teardown evidence and original raw runner-header handling remain mandatory offline evidence checks. Missing/conflicting authoritative image identity remains a hold; later environment variables cannot rescue a conflicting header.

## Independent validation and limits

Command: python3 -B -m unittest discover -s benchmarks/keel-brace-v1/independent-analysis -p test_audit.py -v

Result: all 60 unique pure tests PASS. The full-stage mock first demonstrates final PASS with valid canonical acquisition, then demonstrates rawDisposition PASS with final INVALID/INCOMPLETE after a last-child nonzero-exit-style invalid/failure record. Population, budget, gate, score, per-round, mean, exposure and variation mutations fail. These are pure fixtures, not Robocode battles.

Log: /private/tmp/keel-brace-independent-analysis-review-60-tests.log
Log SHA256: b3ae2267d730805ffd437661b159bac1ab972ecb4613874d780fba7e2484df5e
Verification: /private/tmp/keel-brace-independent-analysis-review-verification.json
Local runtime: Python3.15.1, macOS27.2 arm64. This is not the pinned hosted Python3.13.17/runtime validation. The author's retained historical replay reports 28 parser fields with zero mismatches; this review verified retained manifest bytes and did not rerun historical gameplay.

Before any game authority: independently review the prospective image amendment and its actual probe evidence, verify the final published coherent source/JAR hashes and pinned CI results, then complete the same-worker source/runtime/authority/custody gates. Offline source checks do not independently attest live tool provenance, API freshness or actual worker admission. No operational clearance is inferred from this scoped PASS.
