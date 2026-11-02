# Containment scorer development screen — rejected

[PR #29](https://github.com/stephanelee/robocode/pull/29) / [registered protocol and full result](../../docs/ai-containment-proposal.md).

One fixed train-only fit and one frozen development-only screen reused the original six training and three development games. **No new games were run.** The old withheld/transfer data was not scored or reused as confirmation. Development radius-18 containment improved6.5512%→8.1738%, but the +1.622601-point gain missed the fixed2-point gate, so the candidate was **rejected** before the planned18 fresh confirmation rounds. No live model or gameplay change is included.

Actual [run41525051072](https://github.com/stephanelee/robocode/actions/runs/41525051072), exact source `1516813a02ede80d673d9562fd853d4f866d01ff`; model `eacae24357e378e857ef2a37ad7e7533dac847773921556494e3c36a079e48dc` was published in [the frozen artifact](https://github.com/stephanelee/robocode/actions/runs/41525051072/artifacts/13178642759) before [the single development screen](https://github.com/stephanelee/robocode/actions/runs/41525051072/artifacts/13179511790).

The exact model, fit/screen reports, original protocol, reserved-but-unused fresh-game layout, and original model/workflow source are preserved here. Original input observations, manifest, schema and dependency code remain in [the first pilot archive](../ai-motion-v1/README.md), avoiding a duplicate corpus. Input SHA-256 is `d2cb8e055659485d89648eec6cd3aebe8cbfce8dfb4dd19f83e7ba3efa1d11e2`. `tools/test_containment_v2_evidence.py` verifies hashes, development-only identities and exact rejection without fitting another model.

All three development games improved; coverage, worst-game and short-horizon guards passed. Lower radial error148.020→115.778pixels is descriptive. Development progress is not fresh confirmation or competitive strength; no criterion waiver, retune or additional fit/screen occurred. Research models remain outside the bot directory and are never automatically loaded.
