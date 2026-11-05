import copy
import json
from pathlib import Path
import tempfile
import unittest

import tournament
import tournament_playoffs as playoffs


def result(a, b, score_a=100, score_b=50, places_a=12, places_b=8, game=1):
    return dict(game=game, status='complete', official=dict(numberOfRounds=20, results=[
        dict(name=a, totalScore=score_a, firstPlaces=places_a),
        dict(name=b, totalScore=score_b, firstPlaces=places_b)]))


def swiss_fixture(output):
    """Use the committed score-only evidence, never require private raw reports."""
    path = Path(__file__).resolve().parents[1] / 'benchmarks/tournament-2026-11-04/official-results.json'
    reports = [r for r in json.loads(path.read_text())['stages'] if r['stage'].startswith('swiss-')]
    if len(reports) != 4:
        raise ValueError('Expected four committed Swiss score reports')
    for report in reports:
        tournament.save(output / (report['stage']+'.json'), report)


class PlayoffTest(unittest.TestCase):
    def pair(self):
        return dict(matchId='U1', a='A', b='B', games=[result('A', 'B'), result('A', 'B', game=2)])

    def test_score_and_first_place_tiebreak(self):
        p = self.pair()
        self.assertEqual('A', playoffs.decision(p)['winner'])
        p['games'] = [result('A', 'B', 100, 100), result('A', 'B', 100, 100, game=2)]
        self.assertEqual('round-first-places', playoffs.decision(p)['resolution'])
        self.assertEqual('A', playoffs.decision(p)['winner'])

    def test_one_extra_game_then_registered_lottery(self):
        p = self.pair()
        p['games'] = [result('A', 'B', 100, 100, 10, 10, i) for i in [1, 2]]
        self.assertTrue(playoffs.decision(p)['needsExtra'])
        p['games'].append(result('A', 'B', 1, 2, 15, 5, 3))
        self.assertEqual('B', playoffs.decision(p)['winner'])
        p['games'][2] = result('A', 'B', 1, 1, 15, 5, 3)
        first = playoffs.decision(p)
        self.assertEqual('registered-draw-of-lots', first['resolution'])
        self.assertEqual(first, playoffs.decision(copy.deepcopy(p)))
        p['games'].append(result('A', 'B', game=4))
        with self.assertRaises(ValueError):
            playoffs.decision(p)

    def test_extra_game_cannot_override_decided_pairing(self):
        p = self.pair()
        p['games'].append(result('A', 'B', 0, 1000, game=3))
        with self.assertRaises(ValueError):
            playoffs.decision(p)

    def simulate(self, lose_final=False):
        seeds = [dict(seed=i+1, name=chr(65+i)) for i in range(8)]
        matches, reports, losses = {}, [], {s['name']: 0 for s in seeds}
        source = lambda ref: seeds[ref-1]['name'] if isinstance(ref, int) else (matches[ref[1:]]['winner'] if ref[0]=='W' else next(n for n in (matches[ref[1:]]['a'], matches[ref[1:]]['b']) if n != matches[ref[1:]]['winner']))
        for number, wave in enumerate(playoffs.WAVES, 1):
            if number == 7 and not lose_final:
                break
            draw = []
            for mid, a, b in wave:
                a, b = source(a), source(b)
                self.assertNotEqual(a, b)
                self.assertLess(losses[a], 2)
                self.assertLess(losses[b], 2)
                winner = b if mid == 'G1' and lose_final else a
                loser = b if winner == a else a
                games = [result(a, b, 100 if winner==a else 50, 100 if winner==b else 50, game=g) for g in [1, 2]]
                p = dict(matchId=mid, a=a, b=b, winner=winner, status='complete', games=games)
                matches[mid] = p
                losses[loser] += 1
                draw.append(p)
            reports.append(dict(stage=f'playoff-{number}', status='complete', completedGames=2*len(draw), draw=draw))
            rows, champion = playoffs.standings(seeds, reports)
            self.assertEqual(losses, {r['name']: r['losses'] for r in rows})
            self.assertTrue(all(r['eliminated'] == (losses[r['name']] == 2) for r in rows))
            if number < (7 if lose_final else 6):
                self.assertIsNone(champion)
        self.assertEqual(1, sum(n<2 for n in losses.values()))
        self.assertIn(champion, losses)
        return seeds, reports

    def test_full_bracket_without_reset_preserves_second_chance(self):
        seeds, reports = self.simulate()
        rows, _ = playoffs.standings(seeds, reports[:1])
        self.assertEqual(4, sum(r['losses'] == 1 for r in rows))
        self.assertFalse(any(r['eliminated'] for r in rows))
        self.assertEqual(14, sum(len(r['draw']) for r in reports))

    def test_lower_finalist_win_requires_reset(self):
        seeds, reports = self.simulate(lose_final=True)
        rows, champion = playoffs.standings(seeds, reports[:6])
        self.assertIsNone(champion)
        self.assertEqual(2, sum(not r['eliminated'] for r in rows))
        self.assertEqual(15, sum(len(r['draw']) for r in reports))

    def test_canonical_context_freeze_and_history_validation(self):
        # Generate only toy playoff results over the committed Swiss score fixture.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            swiss_fixture(root)
            pairs, meta = playoffs.context(root, 1)
            self.assertEqual(pairs, playoffs.context(root, 1)[0])
            self.assertEqual([('KeelHelm','AISurvivor'),('Breakwater','Adaptive Prime'),('Keel','Meridian'),('Diamond Port','Prism')], [(p['a'],p['b']) for p in pairs])
            for p in pairs:
                p.update(status='complete', winner=p['a'], games=[result(p['a'],p['b'],game=g) for g in [1,2]])
            tournament.save(root/'playoff-1.json', dict(stage='playoff-1',status='complete',completedGames=8,draw=pairs))
            next_pairs, _ = playoffs.context(root, 2)
            self.assertEqual(4, len(next_pairs))
            self.assertEqual(8, len({n for p in next_pairs for n in (p['a'], p['b'])}))
            (root/'swiss-4.json').write_text((root/'swiss-4.json').read_text()+'\n')
            with self.assertRaises(ValueError):
                playoffs.context(root, 2)

    def test_public_pair_keeps_resolution_but_drops_private_artifacts(self):
        p = self.pair()
        p.update(status='complete', winner='A', resolution='combined-score', private='secret')
        public = tournament.public_pair(p)
        self.assertEqual('U1', public['matchId'])
        self.assertEqual('combined-score', public['resolution'])
        self.assertNotIn('secret', json.dumps(public))

    def test_each_canonical_wave_and_reset_gate(self):
        for reset in (False, True):
            with self.subTest(reset=reset), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                swiss_fixture(root)
                for wave in range(1, 8 if reset else 7):
                    pairs, _ = playoffs.context(root, wave)
                    for p in pairs:
                        wins_b = p['matchId']=='G1' and reset
                        p.update(status='complete', winner=p['b'] if wins_b else p['a'],
                                 games=[result(p['a'],p['b'],50 if wins_b else 100,100 if wins_b else 50,game=g) for g in [1,2]])
                    tournament.save(root/f'playoff-{wave}.json',dict(stage=f'playoff-{wave}',status='complete',completedGames=2*len(pairs),draw=pairs))
                if not reset:
                    with self.assertRaisesRegex(ValueError,'reset not required'):
                        playoffs.context(root,7)


if __name__ == '__main__':
    unittest.main()
