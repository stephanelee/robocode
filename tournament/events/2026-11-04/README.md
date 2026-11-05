# Robocode Elimination Cup — completed event

**KeelHelm is champion**, finishing the double-elimination playoffs with four
pairing wins and zero losses. Diamond Port is runner-up after its second loss.
The registered reset was not required and no wave-7 game started.

| Final match | Combined valid score | Outcome |
| --- | --- | --- |
| Grand final: KeelHelm vs Diamond Port | 3,175–1,861 | KeelHelm is champion |
| Lower final: Diamond Port vs Keel | 2,642–2,621 | Diamond Port advanced; Keel was eliminated |

Final totals: **98 validated games / 1,960 valid rounds**, including unranked
practice. Preserve twenty earlier excluded solo startup rounds and twenty
user-voided wave-4 rounds separately: **2,000 known physical rounds**.
Only the valid replacement of the earlier corrupt-recording game counts.
The tournament is complete; another event requires a separate draft.

## Rules and entrants

Practice: two games per bot, unranked. Four Swiss rounds qualify the top eight
for double-elimination playoffs. Each pairing uses two twenty-round games with
reversed launch order and cold models. Swiss pairing points: three for a win,
one each for a draw, zero for a loss. Playoffs award advancement, not Swiss points.
Two playoff match losses eliminate a bot; a first loss sends it to the lower
bracket. A lower-finalist grand-final win triggers the registered bracket reset.
Playoff ties use combined score, round first places, one extra twenty-round game,
then the pre-registered SHA256 draw. See [full registration](../../../docs/tournament-playoffs-2026-11-05.md).

Home/experimental entries: AISurvivor, Aegis, Breakwater, DiamondChallenger, Keel,
KeelHelm, Meridian and Prism. Guests: **Retcon, Diamond Port and Adaptive Prime**.
Official samples: Walls, SpinBot and RamFire. All fourteen entries came from the
frozen source/guest/sample manifests; no mid-event tuning or bot replacements.

Frozen main source: `1f6e6a38582838fe891330a0b3e7a7d9f1668279`.
[Event configuration](event.json), [all entrant file hashes](entrant-pins.json)
and [shared runtime/helper pins](shared-inputs.json) preserve the inputs. Build
with Java 17, use the pinned Java 25 battle runtime and preserve source/JAR parity.
All games ran on minimind, not GitHub Actions.

## Results and evidence

- [Official score index](../../../benchmarks/tournament-2026-11-04/official-results.json).
- [Practice and Swiss report](../../../docs/tournament-2026-11-04-results.md).
- Playoff reports: [opening](../../../docs/tournament-playoff-1-results.md),
  [wave 2](../../../docs/tournament-playoff-2-results.md),
  [wave 3](../../../docs/tournament-playoff-3-results.md),
  [wave 4](../../../docs/tournament-playoff-4-results.md),
  [lower final](../../../docs/tournament-playoff-5-results.md),
  [grand final](../../../docs/tournament-grand-final-results.md).
- Exceptional rulings: [wave-2 telemetry](../../../docs/tournament-playoff-2-ruling.md),
  [wave-3 zero-game startup](../../../docs/tournament-playoff-3-startup-recovery.md),
  [wave-4 explicit void/replacement](../../../docs/tournament-playoff-4-void-ruling.md).

Raw recordings, logs, metrics, models, bot copies and full archives remain in
`local-evidence/tournament-2026-11-04/` and the local benchmark directories. They
are not committed or published as website assets. Archive hashes and exact local
paths are recorded in the measured reports. Keep the corrupt voided attempt too.

## Website and replay status

The site has separate Home, Matches, Bots and Rules pages, a blue palette, guest
highlights, pairing winners and aligned bot-profile results. It is public with
`noindex,nofollow` requested. The production hostname is resolved from hosting,
then supplied through `TOURNAMENT_SITE_URL`; it is not hard-coded in this folder.

```sh
# Set TOURNAMENT_SITE_URL from the Sites deployment response.
python3 tournament/scripts/manage.py site-url --stage playoff-6
```

There are **42 verified private movies** for valid recorded games. The live site
has **20 replays**, including both grand-final and both lower-final games. The user authorized
archiving the fourteen Swiss-4 and eight older lower-bracket public videos for
space; their original movies, recordings and scores remain retained. The voided
original has a corrupt replay; its replacement movie uses the new valid recording.
The player retains live energy/official scores, twenty chapters and 0.25×/0.125×
controls. Large MP4s use hash-verified pieces with no quality change.

For this event, keep `--omit-swiss4 --archive-lower-before-wave 4` when rebuilding
the public catalog. For another event, review archival policy and choose its own
output/media directories. [Create a separate draft](../../new-event.md).
