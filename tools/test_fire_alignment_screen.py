import unittest

from fire_alignment_screen import screen_decision


def battle(share, shots=100, hits=10, damage=20):
    return dict(rounds=5, firstPlaces=0, totalScore=share, scoreShare=share,
        results=[{'totalScore': share}, {'totalScore': 1 - share}],
        metrics=dict(turns=1000, shotsFired=shots, bulletHits=hits, damageDealt=damage,
                     damageTaken=50, wallCollisions=2, skippedTurns=0))


class FireScreenTests(unittest.TestCase):
    def test_locked_screen_advances_but_never_retains(self):
        comparison = {'blocks': [{'previous': battle(.1), 'candidate': battle(.12, hits=15)} for _ in range(3)]}
        samples = [{'previous': battle(.8), 'candidate': battle(.8)} for _ in range(3)]
        for row in samples:
            row['previous']['rounds'] = row['candidate']['rounds'] = 3
        result = screen_decision(comparison, samples)
        self.assertEqual(result['status'], 'advance-to-confirmation')
        comparison['blocks'][2]['candidate']['scoreShare'] = -.00000001
        self.assertEqual(screen_decision(comparison, samples)['status'], 'reject')

    def test_zero_shot_mechanism_is_inconclusive(self):
        comparison = {'blocks': [{'previous': battle(.1), 'candidate': battle(.12, shots=0, hits=0)} for _ in range(3)]}
        self.assertEqual(screen_decision(comparison, [])['status'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()
