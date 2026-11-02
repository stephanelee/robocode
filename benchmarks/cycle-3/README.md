# Cycle 3 evidence

The [human-readable report and decision](../../docs/improvements/cycle-3.md) live under docs/.

This directory retains canonical JSON, original registration and prospective addendum, candidate source patch, and all nine games' raw official results, telemetry, and logs: 60 incumbent training rounds and 120 frozen evaluation rounds.

The three original numeric checkpoints are retained under checkpoints/ with byte-preserving Git attributes. They are research evidence outside the bot's working directory, not automatically loaded models. Their SHA256, persisted 20-round counters, and saved targeting mass/nonzero counts are verified by tests. The gameplay candidate was rejected and reverted.
