# Aegis v1 registered pilot evidence

The registered plan and outcome are in [docs/improvements/aegis-shield-v1.md](../../../docs/improvements/aegis-shield-v1.md). The sole execution is [run 41721083718](https://github.com/stephanelee/robocode/actions/runs/41721083718) (job 126213698415, `ubuntu-26.04`, execution source `28cd9333d325f44ca407990fbdf5d00fc2d493ee`).

- `pilot-log-output.jsonl` holds the two canonical JSON lines that `tools/aegis_experiment.py` printed to the job log: the pilot game summary and the `reject` decision. They are copied verbatim.
- The full raw artifact `aegis-shield-v1-28cd9333d325f44ca407990fbdf5d00fc2d493ee` (artifact ID 13273276099, 182 files, 2,633,726 bytes, zip sha256 `fd8052073a95e2353544f42ad2c26ef1ae9ebf5cb41a34504f61fd29406e784c`, expires 2026-12-03) contains:
  - `experiment.json` with runtime and opponent provenance;
  - `registered-plan.md`;
  - all 20 metrics rows, `battle.json` and `runner.log`;
  - the staged opponent source hashes and the executed tooling and workflow bytes.

  **It has not yet been imported into this repository** because the implementing session's network policy blocked the artifact storage host. Import it here, unmodified, before it expires, and verify the zip digest.
- No other attempt exists. The pilot failed, so no confirmation data was produced.
