"""Synthetic proofs for the prospectively fixed candidate scorer."""
import copy
import unittest

from containment_forecast import BinaryMoments, CANDIDATES, decision, fit, predict, screen
from legal_scan_dataset import build_samples
from motion_forecast import FEATURES, HORIZONS
from test_motion_forecast import rows
from test_legal_scan_dataset import manifest


class ContainmentTests(unittest.TestCase):
    def test_binary_intercepts_and_constant_columns(self):
        moments = BinaryMoments()
        for _ in range(3):
            moments.add([7] * len(FEATURES), [1, 0, 1, 0])
        model = moments.model()
        self.assertEqual(model['intercept'], [1, 0, 1, 0])
        self.assertTrue(all(c == 0 for coefficients in model['coefficients'] for c in coefficients))

    def test_dev_and_spent_games_cannot_change_fit(self):
        records = rows()
        original = fit(records, manifest())
        for record in records:
            if record['gameId'] != 'A':
                record['scan']['x'] += 300
                record['own']['speed'] = -8
        self.assertEqual(original, fit(records, manifest()))

    def test_rank_ties_current_and_stale_fallback(self):
        model = fit(rows(), manifest())
        sample = list(build_samples(rows()[:64], manifest(), (8,), 8))[3]
        for m in model['models'].values():
            m['intercept'] = [0] * 4
            m['coefficients'] = [[0] * len(FEATURES) for _ in range(4)]
        self.assertEqual(predict(model, sample, manifest()['games'][0])[1], 'current')
        model['models']['8']['intercept'][3] = 2  # raw scores not clipped probabilities
        self.assertEqual(predict(model, sample, manifest()['games'][0])[1], 'headon')
        sample['anchorTurn'] += 3
        result = predict(model, sample, manifest()['games'][0])
        self.assertEqual(result[1], 'current')
        self.assertFalse(result[2])

    def test_full_mask_screen_coverage_and_selection_counts(self):
        model = fit(rows(), manifest())
        cells = screen(rows(), manifest(), model)
        self.assertEqual(set(cells), {'B'})
        self.assertEqual(cells['B']['40']['anchors'], 64)
        self.assertEqual(cells['B']['40']['observed'], 24)
        self.assertEqual(sum(cells['B']['40']['selections'].values()), 24)
        self.assertIn('eligibleContainment', cells['B']['40']['methods']['scorer'])
        self.assertEqual(decision(cells)['status'], 'inconclusive')

    def test_unrounded_gain_majority_short_and_missing_gates(self):
        cells = {g: {str(h): {'observed': 50, 'methods': {
            'current': {'containment': .1}, 'scorer': {'containment': .13}}} for h in HORIZONS} for g in ('a', 'b', 'c')}
        self.assertEqual(decision(cells)['status'], 'development-pass')
        for game in cells.values():
            for h in (24, 32, 40):
                game[str(h)]['methods']['scorer']['containment'] = .1199999999
        self.assertFalse(decision(cells)['checks']['primaryGain'])
        cells['a']['24']['observed'] = 49
        self.assertEqual(decision(cells)['status'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()
