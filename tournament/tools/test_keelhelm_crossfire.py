import gzip
import json
from pathlib import Path
import tempfile
import unittest

import keelhelm_crossfire as crossfire


class CrossfireContracts(unittest.TestCase):
    def test_registers_fresh_balanced_budget(self):
        rows = crossfire.make_schedule(list(range(16)))
        self.assertEqual((88, 520), (len(rows), sum(row['rounds'] for row in rows)))
        self.assertEqual(88, len({row['id'] for row in rows}))
        melee = [r for r in rows if r['scenario'] == 'melee' and r['arm'] == 'candidate']
        self.assertEqual([8]*4, [sum(r['order'].index(crossfire.CONTROL) == seat for r in melee) for seat in range(4)])
        for pair, (first, second) in enumerate(zip(rows[::2], rows[1::2])):
            self.assertEqual(2026120200 + pair, first['seed'])
            for field in ('scenario', 'seed', 'positions', 'order', 'rounds'):
                self.assertEqual(first[field], second[field])
            self.assertEqual({'candidate', 'control'}, {first['arm'], second['arm']})
        with self.assertRaises(ValueError):
            crossfire.make_schedule([0] * 16)

    def test_treatment_identity_replaces_only_one_native_slot(self):
        game = next(r for r in crossfire.make_schedule(list(range(16))) if r['scenario'] == 'SpinBot')
        translated = crossfire.risk.native_game(crossfire.transport_game(game))
        self.assertEqual({crossfire.CANDIDATE, 'Spin Bot'}, set(translated['order']))
        self.assertEqual(game['positions'][crossfire.CONTROL], translated['positions'][crossfire.CANDIDATE])
        self.assertIn(crossfire.CONTROL, game['order'])

    def test_stage_candidate_descriptor_folder_and_actual_jar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / 'Walls'
            sample.mkdir()
            (sample / 'Walls.java').touch()
            (sample / 'Walls.json').write_text('{"name":"Walls","version":"fixture"}')
            (root / 'samples' / 'lib').mkdir(parents=True)
            game = next(r for r in crossfire.make_schedule(list(range(16))) if r['scenario'] == 'Walls')
            freeze = {'source': str(root), 'bases': {'Walls': str(sample)},
                      'candidateFolder': crossfire.FOLDER,
                      'candidateProject': str(crossfire.ROOT / 'experiments' / crossfire.FOLDER)}
            folders, identities = crossfire.risk.stage(game, freeze, root, root / 'run')
            crossfire.verify_staging(game, folders, identities)
            self.assertIn((crossfire.CANDIDATE, '0.1.0'), identities)
            self.assertIn('/identity/KeelHelm%20Crossfire',
                          (Path(folders[0]) / (crossfire.FOLDER + '.sh')).read_text())

    def recording(self, path, *, previous_enemies=2, multiple_hits=False):
        name = crossfire.CANDIDATE
        shot = dict(ownerId=1, bulletId=7, power=1.6, direction=0)
        rows = [
            dict(type='GameStartedEventForObserver', participants=[dict(id=1, name=name), dict(id=2, name='Diamond Port'), dict(id=3, name='Prism')]),
            dict(type='TickEventForObserver', roundNumber=1, turnNumber=11,
                 botStates=[dict(id=1, energy=20, enemyCount=previous_enemies)], events=[]),
            dict(type='TickEventForObserver', roundNumber=1, turnNumber=12, botStates=[dict(id=1, energy=20, enemyCount=1)],
                 events=[dict(type='BulletFiredEvent', turnNumber=12, bullet=shot)]),
            dict(type='TickEventForObserver', roundNumber=1, turnNumber=13, botStates=[dict(id=1, energy=19)],
                 events=[dict(type='BulletHitBotEvent', bullet=shot, victimId=2, damage=7.6)]
                        + ([dict(type='BulletHitBotEvent', bullet=shot, victimId=3, damage=2.0)] if multiple_hits else [])),
            dict(type='RoundEndedEventForObserver', roundNumber=1,
                 results=[dict(id=1, firstPlaces=1), dict(id=2, firstPlaces=1)]),
        ]
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')

    def test_observer_confirms_actual_shot_and_distinguishes_hit_from_intent(self):
        telemetry = ['1 crossfire requests=4 accepted=3 confirmed=2 unknown=1',
                     '1 crossfireBullet turn=12 bullet=7 intendedTarget=2 power=1.6 direction=360.0',
                     '1 crossfireBullet turn=15 bullet=8 intendedTarget=2 power=1.6 direction=0']
        game = dict(arm='candidate', rounds=1, scenario='melee')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recording.gz'
            self.recording(path)
            result = crossfire.audit_alternates(path, telemetry, game)
            self.assertEqual(1, len(result['matched']))
            self.assertEqual('Diamond Port', result['matched'][0]['actualTarget'])
            self.assertEqual([[1, 8]], result['unmatchedCallback'])
            self.assertEqual(1, result['firstPlaceTieRounds'])
            self.assertEqual([20, 19], result['visibleRoundEnergy'][1])
            altered = [telemetry[0], telemetry[1].replace('turn=12', 'turn=13'), telemetry[2]]
            self.assertFalse(crossfire.audit_alternates(path, altered, game)['matched'])
            with self.assertRaises(ValueError):
                crossfire.audit_alternates(path, telemetry[:2], game)
            with self.assertRaises(ValueError):
                crossfire.audit_alternates(path, telemetry, dict(game, scenario='Diamond Port'))

    def test_request_phase_is_previous_tick_and_unknown_does_not_count(self):
        telemetry = ['1 crossfire requests=1 accepted=1 confirmed=1 unknown=0',
                     '1 crossfireBullet turn=12 bullet=7 intendedTarget=2 power=1.6 direction=0']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recording.gz'
            for enemies in (None, 1, 2):
                self.recording(path, previous_enemies=enemies)
                result = crossfire.audit_alternates(path, telemetry, dict(arm='candidate', rounds=1, scenario='melee'))
                self.assertEqual(int(enemies == 2), len(result['matched']))
                if enemies != 2:
                    self.assertEqual([[1,7]], result['unmatchedCallback'])
                    self.assertEqual('missing-or-nonmelee-request-state', result['unmatchedCallbackDetails'][0]['reason'])

    def test_multiple_physical_hits_retain_all_events_without_single_attribution(self):
        telemetry = ['1 crossfire requests=1 accepted=1 confirmed=1 unknown=0',
                     '1 crossfireBullet turn=12 bullet=7 intendedTarget=2 power=1.6 direction=0']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recording.gz'
            self.recording(path, multiple_hits=True)
            result = crossfire.audit_alternates(path, telemetry, dict(arm='candidate', rounds=1, scenario='melee'))
            shot = result['matched'][0]
            self.assertEqual(2, result['ownBulletHits'])
            self.assertEqual([7.6, 2.0], [hit['damage'] for hit in shot['hitEvents']])
            self.assertTrue(shot['actualTargetAmbiguous'])
            self.assertNotIn('actualTarget', shot)
            self.assertNotIn('damage', shot)

    def test_all_gates_with_pass_fixture_and_tied_melee_reject(self):
        assignment = list(range(16))
        games = []
        for game in crossfire.make_schedule(assignment):
            candidate, melee = game['arm'] == 'candidate', game['scenario'] == 'melee'
            name = crossfire.entrant_name(game)
            own_score = (40 if candidate else 20) if melee else 60
            own = dict(name=name, totalScore=own_score, survival=25, bulletDamage=20 if candidate else 10,
                       firstPlaces=(4 if melee else 6))
            names = ['Diamond Port', 'Prism', 'Keel'] if melee else [game['scenario']]
            roster = [own] + [dict(name=n, totalScore=(20 if melee else 40), survival=25) for n in names]
            games.append(dict(game, status='complete', helm=own, official={'results': roster},
                              crossfire={'matched': [None, None] if candidate and melee else []}, firstPlaceTieRounds=0))
        result = crossfire.summarize(games, assignment)
        self.assertTrue(all(result['gates'].values()))
        self.assertEqual(64, result['alternateConfirmed'])
        for game in games[:64]:
            if game['arm'] == 'candidate':
                game['official']['results'][1]['totalScore'] = game['helm']['totalScore']
        result = crossfire.summarize(games, assignment)
        self.assertFalse(result['gates']['meleeDiamondMean'])
        self.assertFalse(result['gates']['meleeAggregateFirst'])
        self.assertFalse(result['gates']['meleeUniqueFirstGames'])
        with self.assertRaises(ValueError):
            crossfire.summarize(games[:-1], assignment)


if __name__ == '__main__':
    unittest.main()
