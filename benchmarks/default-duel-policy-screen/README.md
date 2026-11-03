# Default TRAIN duel policy screen (PR #41)

[Actual run 41561225665](https://github.com/stephanelee/robocode/actions/runs/41561225665), frozen candidate `7894af4f8c07db7274538bf8c2c82c338c44cbb8`, accepted baseline `7104b65d03298caceb010a5bfd085bda65092d17`; [immutable plan](registered-plan.md). All 48 rounds completed once; all prospective screen gates passed.

Retcon score share 24.3845165% to 48.8262561% (+24.4417395 pp), block deltas +18.3301173/+28.9414137/+26.0536876 pp. Wins 2 to 7 out of 15 per arm; walls/1000 turns 20.182389 to 0.233245; zero skipped turns. All three sample guards passed. Hit rate fell 18.10% to 10.17% and damage dealt fell 540.73 to 517.78; this is not an aiming or learning improvement.

Decision: **screen passed; no standalone gameplay promotion**. The user then selected a coherent integrated-bot approach. The separate confirmation was never launched. This screen does not validate the later radar/gun/energy composition. Original narrow source patch remains here; outcomes are not pooled with a new candidate.

`screen.json` retains exact unrounded decisions, source/bundle/runtime pins and mechanism evidence. `raw/` contains all 48 metric rows, 12 official result arrays/logs, six Retcon native exports/capture provenance, and all 12 independently cold model endpoints (losslessly gzip-compressed). Original endpoint SHA256 values remain in the report; gzip timestamps are zero. `files.sha256.json` hashes durable files. The original Actions artifact also retains bot/sample bundles; their identities are pinned in the report/source history. No checkpoint was shared across arms.
