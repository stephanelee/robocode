# Separate incomplete probe startup from timing evidence

[Issue102](https://github.com/stephanelee/robocode/issues/102) retains the original
failed CI attempt: its official result contained Alpha alone, both private traces
were empty, and the recorder reported a missing power-0.5 hit. The cause of the
missing Beta is still unknown. A later passing job did not resolve that cause.

The recorder now requires one official round with exactly Alpha1.0 and Beta1.0,
distinct positive integer IDs, before opening traces or evaluating hit/scan timing.
It records the original battle-result SHA256 and identifies failed execution,
roster, trace-loading or timing stages in `trace.json`. Original result and trace
files remain untouched. A missing participant is a startup/result failure, not
evidence that the engine's bullet/scan contract failed. There is no automatic retry.

This is fail-closed **result validation**, not a startup-admission repair: it does
not prevent a solo engine round from occurring or establish why a process failed
to join. Issue102 remains open for that investigation. Probe behavior, runner
command/setup, timeouts, all hit/scan assertions, bot source/JARs, historical raw
evidence and workflows are unchanged.

Offline validation: eleven tests pass, including the real recorder entrypoint
with mocked downloads/process execution. A solo result is rejected before absent
traces are opened; a complete roster still must pass every existing timing check.
Both original engine-trace fixtures continue to validate unchanged. These tests
launch no engine or bot; ordinary required CI retains its existing diagnostic.
