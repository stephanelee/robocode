# Project working agreements

- Pin every GitHub Actions job, including temporary experiment jobs, to `ubuntu-26.04`. Do not use `ubuntu-latest` or silently fall back to another image.
- Build and test AISurvivor with Java 17 and preserve the source/committed-JAR parity check. Use JDK 25 for Retcon's source launcher after the Gradle build has finished.
- Respect the user's existing author identity, Stéphane Lee. Do not add Codex/AI `Co-authored-by` or `Generated-by` trailers to future commits or merges. Preserve published history and evidence SHAs unless the user explicitly requests a rewrite.
- For each gameplay change, document the hypothesis and comparison protocol prospectively in `docs/` and the relevant issue, then benchmark it. Record outcomes, failures, rejected candidates, source/JAR hashes, opponent pins, and runtime provenance; do not invent improvement claims.
- Implementation agents create PRs. A separate agent independently reviews and merges after required CI checks pass. An implementation agent must not merge its own PR.
