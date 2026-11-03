# Bounded challenger shot/defense diagnostic

[PR63](https://github.com/stephanelee/robocode/pull/63) / [fixed schema/protocol and findings](../../docs/improvements/challenger-diagnostic-v5.md).

One5-round observation-only game completed, no strength gate or repeat. Full12891-record/3876266-byte trace,0drops/skips/failures. Descriptive0/5wins/4.961832%share. No production AIS change or causal strength claim.

- `experiment.json`, `registered-plan.md`: original canonical capture and immutable3572c0b plan.
- `raw/diamond-diagnostic-v5-g01/`: original5roundmetrics, official identities/scores/runner output, own private cold counter5 endpoint/231gun rows/4x47danger bins; diagnostic status and lossless `diagnostic.jsonl.gz` (mtime0). Uncompressed SHA256 is2c42ba7b3ed7a761565ed65016354918749fccda7de0884ac746926c5008979d.
- `analysis.json`: reproducible descriptive shot/passages/coverage/movement joins. It is not a model fit or a competitive rerun.
- `source/`: measured observer/module/test/build/driver/lock, raw reviewed/executed workflow bytes; `capture-tooling/` preserves the original unused-in-game analyzer, while `analysis-tooling/` preserves corrected coverage/removal-aware analyzer plus movement join. Changes occurred offline after capture and never affected policy.
- `execution-provenance.json`, `files.sha256.json`: exact run/artifact/source/JAR and retained byte hashes. Reviewed/executed workflow equality holds after explicit CRLF-to-LF normalization. Instrumented JAR is tracked at source80e039bb, not duplicated here. Licenses preserve upstream notices.

Mean forecast error13.386degrees versus2.306degree body half-width dwarfs.884degree actual aiming tolerance.237/237 firing phase and231/231 origin shifts are zero;3624unmasked next-state residuals zero,5terminal energy masks. These measured limits identify a prediction/usefulness issue without proving a new selector would help. Missing/censored/unjoined data stay unknown; passage and virtual-hit proxies are not actual hit labels. No repeats or automatic policy tuning.
