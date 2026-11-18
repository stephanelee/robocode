# Meridian and Diamond STOP24 experiment archive

The [archive tag](https://github.com/stephanelee/robocode/tree/archive/meridian-stop24-2026-11) anchors commit `b1a84a566a75370c9465f6bd7e20383ec4213cfa` and the full draft history. The archive retains the original sources, executable artifacts, protocols, run records, and independent reviews.

## Meridian held gun

12 games and 60 valid rounds, with zero own skips. Reject the held-gun policy. Diamond Port score share fell from 20.77% to 12.21%, with first places falling from 1/10 to 0/10. The original setup failure consumed zero games.

[Original PR #92](https://github.com/stephanelee/robocode/pull/92) · [Detailed report](https://github.com/stephanelee/robocode/blob/archive/meridian-stop24-2026-11/docs/improvements/meridian-gun-plan-results.md) · [Archived evidence](https://github.com/stephanelee/robocode/tree/archive/meridian-stop24-2026-11/benchmarks/meridian-gun-plan)

## Meridian wave support

2,941 states, 5,882 trajectories and 294,100 forecast rows across a single diagnostic with zero games. Keep the geometry finding and reject live advancement. Omitted first contacts fell from five to zero, while route exposures and wall constraints exceeded the preregistered limits.

[Original PR #94](https://github.com/stephanelee/robocode/pull/94) · [Detailed report](https://github.com/stephanelee/robocode/blob/archive/meridian-stop24-2026-11/docs/improvements/meridian-wave-support-results.md) · [Archived evidence](https://github.com/stephanelee/robocode/tree/archive/meridian-stop24-2026-11/benchmarks/meridian-wave-support)

## MeridianPeak

Five games and 25 valid rounds, with zero own skips. Reject advancement. Against Diamond Port the candidate scored 25.51% and won one of five rounds, below the greater-than-35% share requirement. Incoming damage per 100 opponent shots also increased. Confirmation did not open.

[Original PR #96](https://github.com/stephanelee/robocode/pull/96) · [Detailed report](https://github.com/stephanelee/robocode/blob/archive/meridian-stop24-2026-11/docs/improvements/meridian-peak-v1-results.md) · [Archived evidence](https://github.com/stephanelee/robocode/tree/archive/meridian-stop24-2026-11/benchmarks/meridian-peak-v1)

## Diamond STOP24

The hosted runtime differed from the frozen profile and the acquisition stopped before activation. Zero of 24 numeric cases and zero games ran. The attempt establishes an infrastructure rejection only. The runtime amendment is parked.

[Original PR #100](https://github.com/stephanelee/robocode/pull/100) · [Detailed report](https://github.com/stephanelee/robocode/blob/archive/meridian-stop24-2026-11/docs/improvements/diamond-stop-contract24-results.md) · [Archived evidence](https://github.com/stephanelee/robocode/tree/archive/meridian-stop24-2026-11/benchmarks/diamond-stop-contract24)

[PR #91](https://github.com/stephanelee/robocode/pull/91) supplied the held-gun candidate evaluated in #92. Its original head is an ancestor of the archive commit. KeelWake remains tracked separately in [draft #119](https://github.com/stephanelee/robocode/pull/119).
