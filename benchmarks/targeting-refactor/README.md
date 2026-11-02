# Targeting geometry refactor evidence

[PR #25](https://github.com/stephanelee/robocode/pull/25) / [prospective plan and report](../../docs/targeting-refactor.md).

**Behavior replay and bounded compatibility passed; no strength gain established.** Before editing production code, commit `74bcb6f8e79a97513c93a77425d8e631ba34cc64` froze 288 actual accepted `aimGF` command/state cases, including 256 wide inputs. `accepted-AISurvivor.java.snapshot`, `accepted-goldens.json`, and `capture-provenance.json` preserve the source and output hashes. The refactor reproduces all recorded IEEE-754 gun/fire/pending-wave outputs and preserves stat arrays, bin identity and bot RNG. Pure result fields also match accepted API arithmetic bit for bit.

[Actual sanity run 41515841364 / job 125519071370](https://github.com/stephanelee/robocode/actions/runs/41515841364/job/125519071370) used exact candidate `438e16de5cef95e10fb8a8d2d6b8254a0e47a68f` and immutable plan `c7f64d3be25220d4574f1b5b31bfe9d8bc6c6db7`. All five fixed minimum-risk rounds validated, with zero AIS skipped turns and no model writes: Retcon 0/2 wins; Target, SpinBot, Walls each 1/1. These are descriptive integration outcomes, not an A/B strength comparison.

`sanity.json`, `registered-plan.md`, and all four games' official results, five raw AIS rows and runner logs are retained. Retcon's complete native export and hashes are retained too. `tools/test_targeting_sanity_evidence.py` revalidates budgets, policy, counters, native hashes, results and metrics. The [original complete artifact](https://github.com/stephanelee/robocode/actions/runs/41515841364/artifacts/13173474829) also contains runnable bundles and the pinned official sample sources/libraries. No duplicate JAR is committed in this evidence directory.
