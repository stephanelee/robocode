# MeridianPeak 0.1.0

Isolated Java 17 Tank Royale movement experiment forked from Meridian v3 source `809b880cdcee98d7946d05e6bcd30f623f604348`. The prospective hypothesis, opponent pins, budgets, review gates and acceptance criteria are in [the registered protocol](../../docs/improvements/meridian-peak-v1.md). No competitive result is established by the unit tests.

From the repository root, with `JAVA_HOME` pointing to Java 17:

```sh
./gradlew -p experiments/MeridianPeak clean build
python3 tools/verify_jar.py experiments/MeridianPeak/MeridianPeak.jar experiments/MeridianPeak/build/bot/MeridianPeak/MeridianPeak.jar
```

Load `experiments/MeridianPeak/build/bot/MeridianPeak` in the pinned official booter. It contains the descriptor, launchers and fat JAR. To use the committed binary, copy the three files in `bot/` beside `MeridianPeak.jar` in a directory named `MeridianPeak`.

The defense forecasts the original two orbital commands for 50 ticks. For each inferred wave it obtains a read-only copy of the unchanged incoming learner's weighted rays. Each ray contributes its largest smooth proximity risk over the route, measured from the predicted bot centre to the finite one-tick bullet segment; those weighted peaks are summed. A ray disappears once its near point reaches an arena boundary. Wall and switching costs, wave retirement and the kernel scale remain those specified by the protocol. This route alone uses the pinned engine's two-pass wall constraints in `PeakMotion`; the original `Flight.step` remains unchanged for the gun.

The original gun, power, radar, energy correction and learning policies are preserved. Incoming waves remain inferred from ordinary scans and energy changes, rather than observed enemy bullet launches. Gun support is capped at 192 examples, while separately recorded cumulative labels are uncapped. No opponent code, observer recording or telemetry enters policy decisions.

`MERIDIAN_PEAK_SEED` determines initial orbit direction. `MERIDIAN_PEAK_METRICS` names an optional write-only marker file. `MERIDIAN_PEAK_STATE_DIR` names an optional **absent** output directory: startup refuses an existing directory and never loads a checkpoint. Explicit cold, per-round and endpoint DTOs record models, own callbacks and planner counters. Round-end marks the adapter closing and stops decisions while accepting the SDK's terminal callback drain. The next round start, or final game-end, seals the preceding round exactly once before reset. Callbacks and calculations share one monitor; `go()` waits outside it, and registered old run epochs cannot mutate a reopened round. Timing measurements are diagnostics only.

[Preparation evidence](../../benchmarks/meridian-peak-v1/preparation/) includes build logs, source-origin proof and artifact provenance. Fixtures establish numerical and reporting contracts; fresh independently reviewed games are required to assess strength.

Lifecycle ownership follows the [reviewed Challenger boundary](../../docs/improvements/challenger-round-lifecycle.md). The pinned SDK checks its run-thread identity after waiting for the first tick and immediately before calling `run()`. The adapter rejects a previously registered stale epoch, including attempts to register again. Run registration checks the thread interruption flag under the adapter monitor before claiming an epoch: pinned SDK stopThread interrupts old threads, and no operation clears that flag between its final ownership check and run entry. This closes the supported delayed-entry gap. The boundary assumes SDK ordered dispatch/invalidation and preserved interruption; it does not recognize arbitrary injected callbacks or externally cleared interrupts. No extra event dispatch is inserted at run entry.
