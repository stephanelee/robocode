# Meridian future-wave support: completed zero-game diagnostic

**Geometry evidence retained; reject live advancement.** The one registered acquisition completed2941 poses,5882 contexts/trajectories and294100 forecast rows in14.72seconds, with0 game rounds. Corrected support omitted0 oracle first contacts versus5 for the old window, but selected eligible exposures1208→1209 and wall constraints16326→16358 failed the predeclared advancement guards. No rerun, live game or promotion follows. [Result report](../../docs/improvements/meridian-wave-support-results.md) · [PR94](https://github.com/stephanelee/robocode/pull/94) · [issue93](https://github.com/stephanelee/robocode/issues/93).

`summary.json` is the original machine summary. `raw/manifest.json` preserves size/SHA256 and ordered byte parts for all11 original files; large gzip files are split into at-most2MiB parts without altering original bytes. Restore and independently audit them:

```sh
python benchmarks/meridian-wave-support/reassemble.py --output /tmp/meridian-wave-support-raw
python benchmarks/meridian-wave-support/analysis/independent_audit.py --raw /tmp/meridian-wave-support-raw
```

Both commands acquire no new cases. [Independent audit documentation/provenance](analysis/README.md) and [public outcome verification](https://github.com/stephanelee/robocode/pull/94#issuecomment-6622703802) describe full recomputation. `acquisition/` preserves wrapper/timing logs and byte-reassembly verification. All original files also remain locally under the fixed attempt root; it must never be deleted to authorize a repeat.

The mask-only comparison kept unchanged one-pass forecasts, cold prior, two orbits, scalar costs and retirement36. Motion reference checks found0 sampled mismatches, but none of196 raw double-axis translations produced the known bad first-pass interpolation. That source defect is not repaired or disproved. All contact/exposure results are conditional synthetic geometry, not hit rates or competitive strength.

[Protocol and pre-output clarifications](../../docs/improvements/meridian-wave-support.md), `freeze.json`, engine-source/lock and `artifact/wave-support-diagnostic.jar` are unchanged. Acquisition head e57d71c86789af43e02a4d107fae79054b57cfee; freeze093a4ad1e203b16d73461a751e6d7c761bebc316cc41d6894bf2b47d71cd1403; diagnosticJARaa3889096a5a9b60b62e1f94a8b4fc06446884834df0298bb116d0f699b748ef. Main Meridian and production AISurvivor source/JARs are unchanged. preparation/ retains compile/metadata errors, unused pre-review freezes and independent compilation evidence; those steps ran0 cases.
