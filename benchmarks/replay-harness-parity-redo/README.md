# Fresh replay-harness parity: noninferiority passed

[PR #48](https://github.com/stephanelee/robocode/pull/48) · [protocol and results](../../docs/improvements/replay-harness-parity-redo.md) · [single fresh run](https://github.com/stephanelee/robocode/actions/runs/41649569184).

Original artifact `13239206319` is preserved without bundle duplication: all 1,260 metric rows, 66 official games, 60 native Retcon exports, runner logs and **66 actual numeric cold-model endpoints**. `raw/experiment.json` is the original canonical report; `raw/tooling/` contains executed workflow/source and execution SHA. `raw/preregistered-plan.md` is the exact immutable `2edd072` plan. The integrity manifest covers all original artifact bytes plus execution provenance; models are evidence, never default-path bot inputs.

Candidate source `f224bc7c` / JAR `b41b5de4…` and accepted PR47 baseline `cfdb508b` / JAR `21c5c3d2…` remained frozen. All fresh registered guards pass: mean Retcon difference −0.00090508 pp; one-sided game-block Student-t lower bound −2.30542198 pp above the −5 pp margin; 561 baseline versus 559 candidate first places / 600; zero AIS skips; samples pass. This is **noninferiority within the margin, not a strength gain or universal equivalence proof**.

The [original 380-round failed attempt](../replay-harness-parity/README.md) remains unchanged and is not pooled. That historical archive contains no checkpoint bytes; this is a provenance limitation, not a reason to rewrite its result or repeat games. A separate independent architecture decision is still required before merging PR48.
