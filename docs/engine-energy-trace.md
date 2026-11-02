# Recording bullet-energy visibility

`python tools/engine_energy_trace.py` launches two stationary diagnostic bots through the SHA-256-pinned official Tank Royale 1.4.0 Runner and Bot API. It allocates an embedded server and does not connect to the interactive game. Neither AISurvivor nor Retcon participates.

Alpha and Beta each fire one power-0.5 shot, separated by 150 turns. Both keep their gun and radar pointed at the other stationary bot; after turn 400 Alpha fires finishing shots so the single diagnostic round ends. Each probe flushes raw JSONL for actual scan, fired, hit, and hit-by-bullet callbacks. Callback delivery order is retained in each raw file but is not confused with the engine's internal processing order.

The validator finds each actual power-0.5 hit in both bots' observations, requires damage 2, and requires consecutive enemy-energy scans at `k-1`, `k`, and `k+1`. The scan at `k` must still equal the preceding scan; the next scan must reflect victim damage −2 and shooter bonus +1.5. The trace report records the runtime and hashes of the official release assets, API, probe, recorder, adapter, and raw files. Missing or inconsistent evidence fails the job; synthetic unit fixtures test the validator and are never presented as recorded engine events.

The separate `Pinned engine energy trace` workflow uses `ubuntu-26.04` and captures the actual event files. Its result is a source-contract precondition, not a Retcon strength experiment. The registered performance plan is [Retcon strategy](retcon-strategy.md); its 120 fixed-policy and 120 default-online rounds remain gated on independent review and the reviewed observability integration.

The [preserved actual trace](../benchmarks/foundation-scan-lag/engine-trace/README.md) verifies this contract on Ubuntu 26.04.1 and Temurin 17.0.21. Both isolated hits produced the expected next-scan changes. The same-job JDK25 precondition independently confirmed it before the [120-round Retcon experiment](improvements/foundation-scan-lag.md). That candidate failed its non-regression guards and was reverted. The observation contract remains proven; the correction is not retained.
