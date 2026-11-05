import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import prepare_tournament
import tournament


def round_one():
    pairs = tournament.draw(202611041)
    for pair in pairs:
        pair.update(status='complete', scoreA=100, scoreB=0, pointsA=3, pointsB=0)
    return dict(stage='swiss-1', status='complete', completedGames=14, completedRounds=280,
                plannedGames=14, pointsApplied=True, draw=pairs)


class TournamentRulesTest(unittest.TestCase):
    def test_opening_draw_covers_each_bot_once(self):
        pairs = tournament.draw(202611041)
        self.assertEqual(7, len(pairs))
        self.assertCountEqual(tournament.NAMES, [n for p in pairs for n in (p['a'], p['b'])])

    def test_round_two_has_no_rematches_and_minimum_point_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = round_one()
            (root / 'swiss-1.json').write_text(json.dumps(previous))
            pairs, protocol = tournament.swiss_draw(root, 2)
            prior = {frozenset((p['a'], p['b'])) for p in previous['draw']}
            self.assertCountEqual(tournament.NAMES, [n for p in pairs for n in (p['a'], p['b'])])
            self.assertFalse(any(frozenset((p['a'], p['b'])) in prior for p in pairs))
            # Seven winners and seven losers force at least one 3-point cross-group pair.
            self.assertEqual(3, protocol['totalPointGap'])
            self.assertEqual((pairs, protocol), tournament.swiss_draw(root, 2))

    def test_next_round_requires_a_complete_previous_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = round_one()
            previous['status'] = 'running'
            (root / 'swiss-1.json').write_text(json.dumps(previous))
            with self.assertRaises(ValueError):
                tournament.swiss_draw(root, 2)

    def test_standings_use_current_opponent_points_and_mean_share(self):
        def pairing(a, b, sa, sb):
            return dict(a=a, b=b, status='complete', scoreA=sa, scoreB=sb,
                        pointsA=3 if sa > sb else 0, pointsB=3 if sb > sa else 0)
        reports = [dict(draw=[pairing('Keel', 'Retcon', 100, 0)]),
                   dict(draw=[pairing('Keel', 'Prism', 1, 3)])]
        rows = {r['name']: r for r in tournament.swiss_standings(reports)}
        self.assertEqual(3, rows['Keel']['points'])
        self.assertEqual(3, rows['Keel']['buchholz'])
        self.assertEqual(2, rows['Keel']['played'])
        self.assertAlmostEqual(.625, rows['Keel']['scoreShare'])

    def test_public_export_drops_artifacts_and_unexpected_result_fields(self):
        pair = dict(a='Keel', b='Retcon', status='complete', modelPath='private',
                    games=[dict(game=1, participants=['Keel', 'Retcon'], status='complete',
                                directory='private', retconStats={'private': 'telemetry'},
                                official=dict(numberOfRounds=20, private='secret', results=[dict(name='Keel', totalScore=100, private='secret')]))])
        public = json.dumps(tournament.public_pair(pair))
        for private in ['modelPath', 'directory', 'retconStats', 'private', 'secret']:
            self.assertNotIn(private, public)
        self.assertIn('totalScore', public)

    def test_repeated_site_export_does_not_double_award_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data.json').write_text(json.dumps(dict(rounds=[])))
            report = round_one()
            tournament.update_site(root, report)
            tournament.update_site(root, report)
            data = json.loads((root / 'data.json').read_text())
            self.assertEqual(1, len(data['rounds']))
            self.assertTrue(all(r['played'] == 1 for r in data['standings']))
            self.assertEqual(21, sum(r['points'] for r in data['standings']))

    def test_known_replay_exception_requires_all_and_only_known_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = ET.Element('testsuite', errors='0')
            for cls, name in prepare_tournament.KNOWN_REPLAY_FAILURES:
                case = ET.SubElement(suite, 'testcase', classname=cls, name=name)
                ET.SubElement(case, 'failure', message='gunTurnRate 2.0869864576583836 differs')
            path = root / 'TEST-fixture.xml'
            ET.ElementTree(suite).write(path)
            self.assertTrue(prepare_tournament.known_macos_failures(root))
            case = ET.SubElement(suite, 'testcase', classname='Other', name='broken')
            ET.SubElement(case, 'failure', message='unrelated failure')
            ET.ElementTree(suite).write(path)
            self.assertFalse(prepare_tournament.known_macos_failures(root))


if __name__ == '__main__':
    unittest.main()
