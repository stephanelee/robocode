# Foundation scan-lag experiment evidence

**Rejected after all 120 fixed-policy rounds.** The mandatory online stage was skipped because primary failed. [Report and immutable protocol](../../docs/improvements/foundation-scan-lag.md). Underlying [issue #18](https://github.com/stephanelee/robocode/issues/18) remains open; gameplay/JAR were restored to accepted `a1417774d4f31c43dca0cb83b13c9b216989fbb0`.

Measured [run 41511115116 / job 125502888515](https://github.com/stephanelee/robocode/actions/runs/41511115116/job/125502888515), exact candidate head `78315b77eb4e763bb438cf0cb7df5e7535aa5638`; [complete original artifact](https://github.com/stephanelee/robocode/actions/runs/41511115116/artifacts/13173640726). The artifact contains both complete immutable bot bundles. No candidate JAR is duplicated in Git.

Preserved here: canonical comparison JSON; original registered plan and candidate addendum exact bytes; all six official 20-round results, 120 AIS telemetry rows, runner logs and complete native Retcon exports; candidate source/test patch; initial Java17 actual-engine trace and original source blobs; and the same-job JDK25 actual-engine precondition trace. No model was trained or loaded in these fixed games. There were no failed or repeated performance attempts.

`tools/test_foundation_evidence.py` independently revalidates every raw game, native source identity/hash, telemetry, budget, registered plan hash, and guard decision. The engine trace demonstrates bullet visibility, while the fixed-game non-regression guards failed. These are distinct findings.
