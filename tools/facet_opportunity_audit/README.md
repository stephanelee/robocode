# Facet recorded firing-opportunity audit

Read-only analysis of the five capped own FACET traces from the fixed completed
100-round prepared-cohort screen. No battle launcher, gameplay change, parameter
search, model import or training. Opaque opponent logs and own model exports are
outside the exact archive allowlist. Verified archive parts are held in memory;
traces are streamed, with finite file/line/record and replay quotas.

Run under Java 17:

```sh
FACET_AUDIT_JAVA_HOME="$JAVA_HOME" PYTHONPATH=tools/facet_opportunity_audit \
  python3 -m unittest discover -s tools/facet_opportunity_audit -p 'test_*.py'
python3 tools/facet_opportunity_audit/audit.py \
  --parts-root benchmarks/facet-prepared-cohort-100/attempt-1 \
  --jar experiments/Facet/Facet.jar --java-home "$JAVA_HOME" \
  --output /tmp/FACET_opportunity-audit
```

The JAR must hash to `9b489b8429b01ff96caba378fbb481ddac6811a9a48b522649b3e4ea0eccb174`.
If the checkout later changes its bot, supply that historical own JAR explicitly
and set `FACET_AUDIT_JAR` for tests. CI fetches only that own JAR from candidate
commit `61677622675ca7ac536f3ef5b89b5cce82176f4e`.

`audit.py` reconstructs the active plan from full serialized Plan records,
accepted fire requests, confirmations, round/gap/death/skipped-turn resets and
legally accepted scans. `GateReplay.java` calls the actual frozen own
`FireControl.valid` on mechanical fields only, and evaluates each predicate to
show overlap. Empty Enemy stores are constructed but never trained or queried.
The result is compared with acquired prepared intents. A cap can cut between a
prepared and submitted command; it is reported separately. No per-frame replay
dataset is emitted; only aggregate counts and at most a few examples per game.

Overlapping predicates are not logged first rejection reasons. Held frames are
dependent, often consecutive, and cannot be summed into additional shots: firing
would change subsequent heat, energy, scans and outcomes. Covered outcomes are
selected prefixes, not competitive gun comparisons. Unobserved outcomes remain
censored, and simultaneous contact/hit records retain both forms of evidence.
This audit does not explain every miss or establish improvement.

See the [prospective protocol](../../docs/facet-firing-opportunity-audit-protocol.md)
and [results](../../docs/facet-firing-opportunity-audit-results.md).

The fixed archive has no terminal callback preceding its confirmation, as checked independently. The join relies on this property; buffer earlier terminal records before generalizing to other evidence. The whole-archive hash prevents silently substituting another population.
