# Lower final — measured results

Diamond Port wins L6 **2,642–2,621**, a twenty-one-point margin. The two valid
twenty-round games split one each; the registered pairing rule uses combined
official score rather than game wins or round first places.

| Game | Diamond Port | Keel | Game winner |
| --- | --- | --- | --- |
| 1 | 1,078 | 1,458 | Keel |
| 2 | 1,564 | 1,163 | Diamond Port |
| Combined | 2,642 | 2,621 | Diamond Port |

Keel receives its second playoff loss and is eliminated. Diamond Port retains
one loss and advances to meet unbeaten KeelHelm. After the lower final, the user
explicitly authorized the grand final and its registered conditional reset;
see the separate prospective grand-final protocol.

Measured budget: two original games / forty competitive rounds, no extra tie
game, failures, reruns or new voids. All frozen bot/source/JAR/runtime inputs,
cold starts, seeds, validators and observer-adapter bytes remain unchanged.
This is descriptive event evidence, without a gameplay change or strength claim.

At completion of this wave: **96 valid games / 1,920 competitive rounds**, plus
twenty earlier excluded solo rounds and twenty user-voided wave-4 rounds:
**1,960 known physical rounds**. The score index preserves prior official hashes
and excludes the earlier voided result. Both new recordings pass independent
gzip CRC, twenty-round coverage, game-boundary and final-score checks.

## Runtime and storage

The reviewed orchestration moved to tournament/scripts/ before this wave.
Its runner SHA256 is
7dbe043e148dff321d9ca7a57861adefaf0440aad0f28af9afb2e0331de91839.
Bracket and Java-adapter pins remain unchanged. The exact new draw and hashes
were registered prospectively, and all 29 checks passed before play. Execution
remained on minimind, not GitHub Actions.

The first launch guard refused insufficient disk space and started zero games.
The user authorized Homebrew download-cache removal, then older inactive
temporary Git stores after cache cleanup did not provide lasting free capacity.
Only matching temporary Git layouts with no open files were removed, keeping the
newest thirty minutes. This may discard temporary undo history, as disclosed.
Repositories, published history and original game artifacts stayed intact.
The successful single-process guard measured 13,527,375,872 bytes free before
os.execv launched the unchanged battle rules.

## Evidence and publication

Canonical stage/draw: benchmarks/tournament-2026-11-04/playoff-5.json and
playoff-5-prospective-draw.json. Full private directory: playoff-5/.
Independent proof: local-evidence/tournament-2026-11-04/playoff-5-verification.json.
Execution logs and approved cleanup manifests remain retained with the evidence.

Full archive: local-evidence/tournament-2026-11-04/playoff-5-full-artifacts.tar.gz,
55,659,771 bytes, SHA256
3bfc24fe4d2cb3b6d0a3cb2fbe36ed417f29f505f9de4c6d96413730530c49f0.

Site version 39 published the first validated score. Version 40 published the
complete pairing and both verified replay movies, bringing the live catalog to
eighteen games / 360 battle rounds and the private valid-movie pool to forty.
Existing user-approved archival remained unchanged. No game was rerun for video;
raw recordings and original movies remain private. The public site retains
twenty chapters, live energy/official scores and 0.25x / 0.125x controls.

[Protocol](tournament-playoff-5-protocol.md),
[grand-final protocol](tournament-grand-final-protocol.md),
[current event hub](../tournament/events/2026-11-04/README.md).
The website base URL is supplied externally through TOURNAMENT_SITE_URL.
