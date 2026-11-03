# Fixed nonlinear gain selector: rejected at development

[PR #36](https://github.com/stephanelee/robocode/pull/36) / [prospective plan and measured result](../../docs/ai-nonlinear-gain-v5.md).

One deterministic depth3/minleaf200 gain-tree fit, followed by one development screen, ran in [41548985063](https://github.com/stephanelee/robocode/actions/runs/41548985063). Same mixed9TRAIN/3RetconDEV population and26legal features as v4; only the declared combined nonlinear/gain-target/abstention family changed. No search, new collection or live model.

**Reject solely primaryGain:** containment6.551200%→7.515421% (+0.964221pp<2 required). All3games improved and remaining gates passed. Mean error148.020468→131.102996px is diagnostic. The conditional18-round tg5 layout was not run; no refit or waiver follows.

- `model.json`: exact frozen54e5da030e334a3760a18d38310e7357ffb0a90693fef6d42fe07a6325383d16, published before screening. Research-only; not autoloaded.
- `fit-report.json` / `development-report.json`: original artifacts13189563390/13189757796.
- `registered-plan.md` / `manifest.json` / `unrun-fresh-games.json`: exact pre-outcome02050591 roles/parameters/new-ID layout.
- `containment_tree_v5.py`: exact measured Git source, model tool hash4878561eb34eaa04a3d42537fe9a9a7b10678c00ff139b5adf5745eef1ec4cb5. Original input helpers remain unchanged and hash-pinned in the model.
- `execution-workflow.yml`: exact Ubuntu26.04 fit→publishedSHA→singleDEV workflow; active one-off removed before final push.
- `preservation.json`: original source/run/artifact identities and per-file hashes.

The original raw corpus is referenced from [ai-motion-v1](../ai-motion-v1/README.md), uncompressed d2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2. No data/bundle/JAR is duplicated here. Production source/JAR remain accepted241156; no competitive improvement claim.
