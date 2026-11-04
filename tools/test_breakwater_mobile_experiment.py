import json
from pathlib import Path
import tempfile
import unittest

import breakwater_mobile_experiment as mobile


class MobileTests(unittest.TestCase):
    def games(self, stage, shares, wins, skipped=0):
        return [dict(gameId=gid, seed=seed, scoreShare=s, wins=w, metrics={'skippedTurns': skipped})
                for (gid, seed), s, w in zip(mobile.layout(stage), shares, wins)]

    def test_layout_and_strict_gates(self):
        for s, w, skips, expected in ((.20, 1, 0, 'reject'), (.21, 0, 0, 'reject'),
                                     (.21, 1, 1, 'reject'), (.21, 1, 0, 'advance')):
            self.assertEqual(mobile.decision(self.games('pilot', [s], [w], skips), 'pilot')['status'], expected)
        for shares, wins, status in (([.51, .51, .50], [10, 10, 10], 'reject'),
                                     ([.51, .51, .01], [10, 10, 11], 'reject'),
                                     ([.99, .49, .49], [10, 10, 11], 'reject'),
                                     ([.51, .51, .50], [10, 10, 11], 'accept')):
            self.assertEqual(mobile.decision(self.games('confirmation', shares, wins), 'confirmation')['status'], status)
        with self.assertRaises(ValueError):
            mobile.decision(self.games('transfer', [.7], [15]), 'pilot')
        with self.assertRaises(ValueError):
            mobile.decision(self.games('transfer', [.7], [15]), 'transfer')

    def test_validate_exact_rows_and_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            common = dict(name='Breakwater', version='0.2.0', schemaVersion=2, seed=532001, checkpointLoaded=False)
            numeric = dict.fromkeys(mobile.INTS + mobile.REALS, 0)
            numeric.update(turns=10, decisions=9, ccwDecisions=9)
            rows = [dict(common, **numeric, round=i, won=False, died=i != 20,
                         officialFirstPlaces=int(i == 20), officialRoundWins=int(i == 20)) for i in range(1, 21)]
            (path / 'metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            me = dict(name='Breakwater', version='0.2.0', id=1, totalScore=12, firstPlaces=1, survival=0, bulletDamage=12)
            enemy = dict(me, name='Diamond Port', version='1.8.28', id=2, firstPlaces=19)
            mobile.v1.save(path / 'battle.json', dict(numberOfRounds=20, results=[me, enemy]))
            model = dict(common, roundsCompleted=20, officialFirstPlaces=1, totals={k: v*20 for k, v in numeric.items()},
                         corrections=[], attackHistory=[], attackScores=[0, 0, 0, 0])
            mobile.v1.save(path / 'model.json', model)
            gid, seed = mobile.layout('pilot')[0]
            checked = mobile.validate(path, gid, seed, False)
            self.assertEqual(.5, checked['scoreShare'])
            self.assertEqual([20], checked['winFlagDisagreements'])
            for key, val in (('officialRoundWins', None), ('officialRoundWins', 0), ('officialFirstPlaces', 2)):
                original = rows[-1].copy()
                if val is None:
                    rows[-1].pop(key)
                else:
                    rows[-1][key] = val
                (path / 'metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
                with self.assertRaisesRegex(ValueError, 'official round wins'):
                    mobile.validate(path, gid, seed, False)
                rows[-1] = original
            (path / 'metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            model['totals']['waveDecisions'] = 1
            mobile.v1.save(path / 'model.json', model)
            with self.assertRaisesRegex(ValueError, 'endpoint'):
                mobile.validate(path, gid, seed, False)
            rows[0]['attackShots'] = 1
            (path / 'metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, 'shot outcomes'):
                mobile.validate(path, gid, seed, False)

    def test_legacy_ais_collection_can_be_validated_despite_missing_local_win_flags(self):
        path = mobile.ROOT / 'benchmarks/breakwater-vs-ais'
        rows = [json.loads(line) for line in (path / 'ais.jsonl').read_text().splitlines()]
        mobile.benchmark.validate_metrics(rows, 20, 'train', 531822, 0)
        self.assertEqual(20, mobile.compare.checkpoint_diagnostics(path / 'ais-model.dat')['trainingRounds'])
        self.assertEqual(15, sum(r['won'] for r in rows))


if __name__ == '__main__':
    unittest.main()
