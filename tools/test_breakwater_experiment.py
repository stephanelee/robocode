import json
from pathlib import Path
import tempfile
import unittest

import breakwater_experiment as runner


class BreakwaterTests(unittest.TestCase):
    def games(self, stage, shares, wins, skips=0):
        return [dict(gameId=gid, seed=seed, wins=w, scoreShare=s, metrics={'skippedTurns': skips})
                for (gid, seed), w, s in zip(runner.layout(stage), wins, shares)]

    def test_pilot_and_confirmation_boundaries(self):
        for share, wins, skips, expected in ((.20001, 1, 0, 'advance'), (.20, 1, 0, 'reject'),
                                           (.23, 0, 0, 'reject'), (.23, 1, 1, 'reject')):
            games = self.games('pilot', [share], [wins], skips)
            self.assertEqual(expected, runner.decision(games, 'pilot')['status'])
        for shares, wins, expected in (([.5, .51, .51], [10, 10, 11], 'accept'),
                                       ([.5, .51, .49], [11, 11, 11], 'reject'),
                                       ([.8, .49, .49], [11, 11, 11], 'reject'),
                                       ([.6] * 3, [10] * 3, 'reject')):
            self.assertEqual(expected, runner.decision(self.games('confirmation', shares, wins),
                                                     'confirmation')['status'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            runner.decision(self.games('confirmation', [.8], [20]), 'confirmation')

    def test_telemetry_endpoint_rounding_and_late_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            common = dict(name='Breakwater', version='0.1.0', schemaVersion=1,
                          seed=531801, checkpointLoaded=False)
            totals = dict.fromkeys(runner.INTEGERS + runner.REALS, 0)
            row = dict(common, **totals, won=False, died=True)
            row['turns'] = 10
            rows = [dict(row, round=i + 1) for i in range(20)]
            # Engine independently detects bullet-bullet and bullet-bot contacts in one tick.
            contacts = dict(shotsFired=1, shieldShots=1, shieldIntercepts=1, shieldHits=1,
                            bulletHits=1, energySpent=.1, shieldDamage=.4, damageDealt=.4)
            rows[0].update(contacts)
            totals.update(contacts)
            (path / 'metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            official = dict(name='Breakwater', version='0.1.0', totalScore=2, survival=0,
                            firstPlaces=0, bulletDamage=2, id=1)
            enemy = dict(official, name='Diamond Port', version='1.8.28', firstPlaces=20, id=2)
            runner.save(path / 'battle.json', dict(numberOfRounds=20, results=[official, enemy]))
            totals['turns'] = 200
            model = dict(common, roundsCompleted=20, totals=totals, attackScores=[0, 0, 0], attackHistory=[], corrections=[])
            runner.save(path / 'model.json', model)
            self.assertEqual(.5, runner.validate(path, 'breakwater-v1-pilot', 531801)['scoreShare'])
            totals['matchedRays'] = 1
            runner.save(path / 'model.json', model)
            with self.assertRaisesRegex(ValueError, 'Endpoint sum'):
                runner.validate(path, 'breakwater-v1-pilot', 531801)
            rows[0]['unclassifiedShots'] = 1
            (path / 'metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, 'Unclassified'):
                runner.validate(path, 'breakwater-v1-pilot', 531801)

    def test_never_replace_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                runner.run(Path(tmp))


if __name__ == '__main__':
    unittest.main()
