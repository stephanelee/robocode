# Invalid first acquisition — all140 physical rounds retained

Run41858182879, source commit8a129ab3f2e437585dd5d894d71ba824a00e9873. Original results.json remains statusinvalid and Retcong02 remains recorded as running at the exception point; battle.json proves its20official completed rounds. Do not convert it into a passing acquisition or exclude its favorable score. Retcong02 metrics have rounds1–11,13–20; missing12. No remaining160rounds ran and no automatic retry/resume is authorized. See docs/facet-results.md and independent-review.md.

Readable battle/Facet metrics are beside each game. All original per-game files (including complete bounded own legal trace/model and opaque runner output) are preserved in normalized tar.gz archives split into at-most1MiBparts. Reassemble into an empty directory:

```
python3 tools/facet_reassemble_evidence.py --root benchmarks/facet-registered-300/attempt-1 --output /tmp/FACET_raw_archives
```

raw-manifest.json checks every part, complete archive and uncompressed file. Reassembly was verified against all seven original archive SHA256 values. No model is loaded or trained by this utility. Opaque runner logs must not expose restricted opponent traceback/source to the implementation side. Original Actions ZIP digest and link are in artifact-provenance.json; Git parts provide durable evidence beyond artifact expiration.

runtime/ contains actual pregame source/compiler/flags/package/license/interpreter provenance; original results retain actual JVM properties, executable hashes, OS/image and every launch argv/state environment. learning-by-round-interval.json preserves cumulative deltas: the missing round12 produces a combined12–13 interval, never invented separate observations. Dependent rounds are not independent game clusters.
