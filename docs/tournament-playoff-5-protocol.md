# Playoff wave 5 — prospective registration

The user authorized the next round. Execute **wave 5 only** on minimind, not
GitHub Actions. The fixed lower final L6 is **Keel vs Diamond Port**, each with one
prior playoff loss. The loser receives its second loss and is eliminated; the
winner advances to meet unbeaten KeelHelm. No grand-final game is authorized.

Budget: two scheduled games / forty rounds, twenty rounds per cold game with
reversed launch order. Registered combined-score and round-first-place tie rules
apply, then at most one extra twenty-round game and the registered SHA256 draw.
Maximum three games / sixty rounds. No tuning, source/JAR change, training carryover,
forfeit, automatic rerun or bracket advancement on a validation failure.

Before play, require at least 2 GiB free and use a single-process assertion followed
by os.execv; failure must not fall through to another shell command. The user
explicitly authorized clearing only the Homebrew download cache. Installed
software, game artifacts and original private videos remain retained.

## Relocated orchestration registration

Scripts now live under tournament/scripts/ and the reusable website template
under tournament/website/. The separately reviewed relocation changes only source
lookup/import priority and current-path instructions; preserve all frozen bot,
shared-runtime, validator and Java-adapter bytes. The current PR is 120 at commit
8a76974d18f13eb567aececa49110089b67bba79; it need not be merged to execute the
reviewed local code. Historical reports and pre-relocation scripts remain intact.

| Input | SHA256 |
| --- | --- |
| tournament/scripts/tournament.py | 7dbe043e148dff321d9ca7a57861adefaf0440aad0f28af9afb2e0331de91839 |
| tournament/scripts/tournament_playoffs.py | 7b1b58726d713840b822c2727f17385ea3a24eaaae6befe849072e3fdf2740b5 |
| tournament/scripts/TournamentBattle.java | 13336dc71373893a8342279ce5db65972cfac9fffd20f7bb619aabe9cf8e5c2e |

Register the exact draw and verify the existing entrant/shared/recording manifests
before launching games. All 29 orchestration/starter tests pass. Stop on any
failure and preserve the attempted game and exact stage report before a separate
ruling. The prior wave-4 void/replacement applies only to that original game.

Publish validated scores as available, then all completed wave-5 replays beside
their results. Retain every raw recording, log, invocation, model and original
private video. Keep the existing public archival cutoff (Swiss 4 and lower matches
before wave 4); this does not authorize removing newer replays. Do not rerun older
games to create missing recordings. Stop after this wave.
