import copy
import unittest

from own_wave_context_diagnostic import contexts, counts
from test_legal_scan_dataset import event, manifest


def shot(turn=1, bullet=1):
    return dict(turn=turn, bulletId=bullet, power=2, direction=0, x=0, y=100)


class OwnWaveTests(unittest.TestCase):
    def test_creation_plus_one_and_first_seen_not_backfilled(self):
        before, delivery = event(1, x=500), event(3, x=500)
        delivery['confirmedShots'] = [shot()]
        result = dict(contexts([delivery, before], manifest()))
        self.assertEqual(result[('A', 1, 1, 2)]['estimatedWaves'], [])
        wave = result[('A', 1, 3, 2)]['estimatedWaves'][0]
        self.assertEqual(wave['radius'], 14 * 3)
        self.assertEqual(wave['firstSeenTurn'], 3)
        self.assertEqual(wave['deliveryAge'], 0)

    def test_exact_window_boundary_and_delayed_delivery(self):
        data = [event(t, x=500) for t in (1, 8, 9, 10)]
        data[0]['confirmedShots'] = [shot()]
        data[-1]['confirmedShots'] = [shot(turn=1, bullet=2)]
        result = dict(contexts(data, manifest()))
        self.assertFalse(result[('A', 1, 8, 2)]['estimatedWaves'][0]['missingFromFirstDelivery8'])
        self.assertTrue(result[('A', 1, 9, 2)]['estimatedWaves'][0]['missingFromFirstDelivery8'])
        delayed = result[('A', 1, 10, 2)]['estimatedWaves'][1]
        self.assertGreater(delayed['eventAge'], 8)
        self.assertFalse(delayed['missingFromFirstDelivery8'])

    def test_future_records_spent_games_and_rounds_cannot_leak(self):
        data = [event(1, x=500), event(10, x=500)]
        data[0]['confirmedShots'] = [shot()]
        before = dict(contexts(data, manifest()))
        future = event(11, x=500)
        future['confirmedShots'] = [shot(turn=11, bullet=2)]
        other = event(1, round_number=2, x=500)
        result = dict(contexts(data + [future, other, event(2, game='C')], manifest()))
        self.assertEqual(before[('A', 1, 10, 2)], result[('A', 1, 10, 2)])
        self.assertEqual(result[('A', 2, 1, 2)]['estimatedWaves'], [])
        self.assertTrue(all(k[0] != 'C' for k in result))

    def test_dedup_first_seen_and_earliest_eligible_wave(self):
        data = [event(1, x=500), event(10, x=500)]
        data[0]['confirmedShots'] = [shot()]
        data[1]['confirmedShots'] = [shot(), shot(turn=10, bullet=2)]
        result = dict(contexts(data, manifest()))[('A', 1, 10, 2)]
        self.assertEqual(len(result['estimatedWaves']), 2)
        self.assertEqual(result['earliestEligible']['bulletId'], 1)
        self.assertTrue(result['earliestEligible']['missingFromFirstDelivery8'])
        self.assertFalse(result['estimatedWaves'][0]['missingFromRecorded8'])
        summary = counts(data, manifest())['A']
        self.assertEqual(summary['uniqueConfirmedShots'], 2)
        self.assertEqual(summary['visibilityDisagreements'], 1)


if __name__ == '__main__':
    unittest.main()
