# Independent retained-row audit

`independent_audit.py` starts no JVM, matrix, test case, bot or server. It imports no repository diagnostic reducer. It streams the four retained raw gzip files, independently reconstructs their frozen state population, histories, eligibility, commands, same-input motion steps, masks, cold-prior risks, route scores, ray geometry and choices, then reconciles the decision guards. Recorded Java source/helper score bits are compared exactly. General Python numeric comparisons allow absolute 1e-8/relative 1e-12 for cross-language math; independently recomputed motion mismatch decisions use the registered absolute 1e-9 threshold. Segment classifications use the retained endpoint doubles after checking their reconstruction, separating classifier fidelity from Java/Python trigonometric rounding.

Restore the original raw files from the committed byte parts to a new directory, then audit:

```sh
python benchmarks/meridian-wave-support/reassemble.py --output /tmp/meridian-wave-support-raw
python benchmarks/meridian-wave-support/analysis/independent_audit.py \
  --raw /tmp/meridian-wave-support-raw --output /tmp/meridian-wave-support-audit.json
```

`--repository` defaults to the repository inferred from the script path. `--raw` defaults to the original local `build/meridian-wave-support-attempt` directory. `--output` is optional. The default audit validates recorded actual-runtime/freeze equality, published source and engine hashes, archived diagnostic classes/JAR, and raw-file hashes. `--verify-local-runtime` additionally rehashes the original absolute classpath, build-output, Homebrew JDK and Python files; it requires the original host and caches. These current-byte checks supplement recorded runtime evidence and do not reconstruct an unavailable historical machine.

`original-audit.json` preserves the initial independent result. `independent-audit.json` includes the final portable audit with original-host checks and independent motion-threshold decisions. `provenance.json` records script/output hashes. Repeated audits read the same preserved rows and acquire no new data.

The sampled paths contain 196 raw translations across both wall axes, but no first-pass result outside the legal center bounds. Zero motion mismatches therefore do not exercise the source-proven bad corner interpolation case or establish general engine conformance. Do not reinterpret this fixed-matrix scope as a repair, held-out validation or competitive evidence.
