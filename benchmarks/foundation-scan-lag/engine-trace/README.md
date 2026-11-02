# Recorded official-engine precondition

Actual event recording: [run 41506306635, job 125486396339](https://github.com/stephanelee/robocode/actions/runs/41506306635/job/125486396339), source `825bf4ef6c547a8e2cc49bf8b287598f770d398a`. Ubuntu 26.04.1 LTS, Temurin 17.0.21, SHA-pinned official Tank Royale Runner/API 1.4.0. No AISurvivor or Retcon gameplay loaded.

`Alpha.jsonl` and `Beta.jsonl` are unmodified raw probe callback records. `trace.json` derives four scan windows from their actual power-0.5 hits at turns 57 and 207: both victim damage −2 and shooter bonus +1.5 first appear on the next scan. The report contains raw/source/release asset hashes. Python tests recompute the raw hashes and all derived assertions; Java tests replay those recorded windows against the candidate tracker for both callback delivery orders. Callback delivery order does not itself establish engine processing order.

An earlier successful recording is available as [run 41506076794](https://github.com/stephanelee/robocode/actions/runs/41506076794), before explicit OS-release reporting was added. Both runs verified the same contract; neither was a performance match. See [method](../../../docs/engine-energy-trace.md) and [prospective Retcon protocol](../../../docs/retcon-strategy.md).
