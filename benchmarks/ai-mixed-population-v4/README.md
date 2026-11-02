# Mixed-opponent training population: development rejected

[PR #33](https://github.com/stephanelee/robocode/pull/33) / [prospective protocol and measured outcome](../../docs/ai-mixed-population-v4.md).

One unchanged-v3 fit reclassified the three spent original sample games as TRAINING alongside six Retcon training games; three Retcon development games stayed DEV. The original three spent Retcon withheld games remained excluded. No new collection was needed. Original raw corpus/provenance remains in [ai-motion-v1](../ai-motion-v1/README.md), uncompressed SHA d2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2.

**Reject; no fresh18 or live model.** The single [run41540789004](https://github.com/stephanelee/robocode/actions/runs/41540789004) produced development containment6.551200%→7.794587% (+1.243387pp<2 required). All three games improved; all other guards passed. Mean error148.020468→125.686838px is diagnostic. This is neither a fresh sample result nor a competitive gain.

- `model.json`: frozen SHA10599435a4fa41480efa011b6cc4a0ff63ccc61a8f3c280f31cedff0c7ec6f4b, published before the single DEV screen. Not autoloaded.
- `fit-report.json` / `development-report.json`: exact original artifacts13186966199/13187006874; no refit or rescreen.
- `manifest.json` / `registered-plan.md`: immutable044e training roles/protocol before outcomes.
- `execution-workflow.yml`: original exact-head Ubuntu26.04 fit→publishedSHA→single-screen workflow; temporary active copy removed before final push.
- `unrun-fresh-games.json`: prospectively registered mf4 layout; none of its18 rounds ran.
- `preservation.json`: source/run/artifact identities and exact per-file hashes.

All26features, context, candidates, original source hashes, equal eligible-row weighting, alpha10 and gates stayed unchanged. See [v3 source evidence](../ai-own-wave-v3/README.md) for the reused feature/model implementation. Production source/test/JAR are unchanged from accepted50b2. No raw corpus or runnable bundle is duplicated here.
