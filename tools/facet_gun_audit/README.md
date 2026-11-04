# Historical Facet gun/wave audit

Read-only offline reproduction of the rejected Diamond100 candidate. No battles, model import/training, parameter search or gameplay edits. Use only own FACET traces/metrics plus the supplied specification and frozen public contracts. Opponent logs/models/source are never opened. This historical reproducer intentionally demonstrates a defect in pinned JAR817a; it does not test a future correction or competitive improvement.

From repository root with Java17:

```sh
python3 tools/facet_gun_audit/reproduce.py \
  --parts-root benchmarks/facet-diamond-repaired-100/attempt-1 \
  --jar experiments/Facet/Facet.jar \
  --java-home /path/to/jdk17 \
  --output /tmp/FACET_gun-audit
```

The JAR must hash to817a9d53ce944d27e3f536a92869eeca2e0aec20ef5c52b2dab882e931205792. CI fetches that exact own JAR from commit93863d2 even if a future current bot changes. Archive parts are verified against their manifest and originalZIP SHA957e78cc5574c6c8619809a06ede7c5d08bc91925281b85aba822fa28546dbb1. Extraction uses an exact allowlist: registration, five own traces and five own metric files, with32MiB per-member bounds. No own model or opponent log is read. Alternatively use `--root /path/to/already/selectively/extracted/build/facet-diamond-repaired-100`.

`analyze.py` reads at most five100000-line traces, reconciles observed own request/confirmation heading/origin/clocks and reports capped coverage, gun attribution, common rating snapshots and mechanically aligned holds. Its output is aggregate diagnostics with at most three hold examples/game, not a training dataset. Missing post-cap outcomes are not misses. Gun-specific counts cannot rank gun strength. Actual bullet power, per-gun prediction angles and label/wave records were not separately serialized, so their frequency/chronology cannot be reconstructed independently from these traces.

`PreparedPredictionMismatch.java` calls the real pinned Core/FireControl/Learner. Sixteen typed toy observations per eligible view make the synthetic starting state coherent; they are not acquired-data fitting or a shipped model. A legally changed scan leaves the prior analytic plan valid but the rating cohort recomputes a different analytic angle. It proves a semantic possibility and construction mismatch, not occurrence frequency or cause of the Diamond losses. Initial empty-view fixture and output are preserved separately in audit evidence; the strengthened fixture is authoritative.

See docs/facet-gun-wave-audit-protocol.md and docs/facet-gun-wave-audit-results.md; no new gameplay correction or measurement is authorized by these tools.
