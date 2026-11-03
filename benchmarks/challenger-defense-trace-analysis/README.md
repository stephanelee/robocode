# Existing Challenger defense trace: bounded exploratory analysis

No new games, capture, fitting, controller changes or strength decision. This reads the already spent five-round PR63 trace; all original evidence stays in [challenger-diagnostic-v5](../challenger-diagnostic-v5/README.md).

The [definition and findings](../../docs/improvements/challenger-defense-trace-analysis.md) were frozen before calculation in Git5f9977b and23f1fdd. `analysis-plan.md` preserves the peer's original declaration; the Git declaration additionally records reviewer cautions about unavailable waves and the uniform prior. `analyze_defense.py` reads only the fixed trace/status and local declaration. `defense-analysis.json` retains all52 hit-level joins, recorded scores/timings and aggregate summaries. No unchosen trajectories or per-decision profile bins were recorded or invented.

Independent reproduction produced **byte-identical**101195-byte JSON, SHA256 `e32f21ccc4d9990c4b085556f560e995039d7720054258761e8f06a8a3babb91`. Reproduce from repository root:

```sh
python benchmarks/challenger-defense-trace-analysis/analyze_defense.py benchmarks/challenger-diagnostic-v5/raw/diamond-diagnostic-v5-g01/diagnostic.jsonl.gz /tmp/defense-reproduced.json
```

Input gzip SHA256 `7d80449fb0b8a15aff0272c70d79a4359bcc68cd00caf09b6c2d940a56428e2c`; decompressed SHA256 `2c42ba7b3ed7a761565ed65016354918749fccda7de0884ac746926c5008979d`. No duplicate trace, model, JAR or opponent bundle is added here. File hashes are in `files.sha256.json`.

Run the four boundary fixtures with `python -m unittest discover -s benchmarks/challenger-defense-trace-analysis -p test_analyze_defense.py`. They preserve unknown/ambiguous joins and reject same-turn or post-hit planner leakage. Peer original findings are retained in `findings.md`; the concise report qualifies the prior-dominance hypothesis.
