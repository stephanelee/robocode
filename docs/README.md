# Project documentation

For performance work, start with **Retcon 1v1**: the [Retcon benchmark workflow](https://github.com/stephanelee/robocode/actions/workflows/retcon.yml) or explicit `--suite retcon` commands in the methodology. Build with Java 17 and run Retcon matches with JDK 25. The default easy sample roster is retained for regression/integration checks.

- [Benchmark methodology and runtime setup](benchmarks.md)
- [Source-grounded plan to beat Retcon](retcon-strategy.md)
- [Measured improvement log and three completed experiments](improvements.md)
- [Cycle 3: rejected saved-targeting experiment](improvements/cycle-3.md)
- [Cycle 2 wall-smoothing report and preregistration](improvements/cycle-2.md)
- [Immutable version-comparison tooling](version-comparisons.md)
- [Cycle 1: rejected back-as-front experiment](improvements/cycle-1.md)
- [Retcon 0.2.0 1v1 baseline](benchmarks/retcon-2026-11-02.md)
- [Original sample reference experiment](benchmarks/samples-2026-11-02.md)

Actual experiment JSON/CSV/dashboard artifacts remain under [benchmarks/](../benchmarks/). Human-readable methods and reports live here; compatibility pointers remain at older documentation paths.
