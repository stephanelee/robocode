import unittest
import aegis_experiment as e


def games(stage, share=.6, wins=15, skip=0):
    return [dict(**g, scoreShare=share, battle=dict(firstPlaces=wins), metrics=dict(skippedTurns=skip)) for g in e.layout(stage)]


def row(i, seed=20263801, **over):
    r = dict(schemaVersion=1, name=e.NAME, version=e.VERSION, mode='train', seed=seed, round=i, roundsCompleted=i, turns=500, checkpointLoaded=False,
             shotsFired=30, bulletHits=0, damageDealt=0.0, damageTaken=19.5, energySpent=3.0, wallCollisions=0, botCollisions=0, skippedTurns=0,
             won=False, died=True, enemyShotsDetected=33, shieldsFired=30, bulletsBlocked=25, hitsTaken=2, unmatchedRemovals=0, inactivityTurns=0,
             meanLabelError=2.0, maxDecisionMicros=20000, slowTurns=0, skippedTurnNumbers=[], shotOutcomes=dict(blocked=25, hit=2, missed=6))
    r.update(over)
    return r


class AegisExperimentTests(unittest.TestCase):
    def test_registered_budgets_and_fresh_disjoint_seeds(self):
        self.assertEqual((20, 60), tuple(sum(g['rounds'] for g in e.layout(s)) for s in ('pilot', 'confirmation')))
        self.assertTrue({g['seed'] for g in e.layout('pilot')}.isdisjoint(g['seed'] for g in e.layout('confirmation')))
        self.assertEqual(4, len({g['gameId'] for s in ('pilot', 'confirmation') for g in e.layout(s)}))

    def test_pilot_gate_exactly_as_registered(self):
        self.assertEqual('advance', e.decision(games('pilot', .200001, 1), 'pilot')['status'])
        for share, wins, skip in ((.2, 1, 0), (.5, 0, 0), (.8, 20, 1)):
            self.assertEqual('reject', e.decision(games('pilot', share, wins, skip), 'pilot')['status'])

    def test_confirmation_gate_exactly_as_registered(self):
        self.assertEqual('confirmed', e.decision(games('confirmation', .6, 11), 'confirmation')['status'])  # 33 > 30
        for data in (games('confirmation', .5, 15), games('confirmation', .7, 10), games('confirmation', .7, 15, 1)):
            self.assertEqual('reject', e.decision(data, 'confirmation')['status'])
        data = games('confirmation', .9, 15); data[1]['scoreShare'] = data[2]['scoreShare'] = .4
        self.assertFalse(e.decision(data, 'confirmation')['checks']['majorityGames'])
        with self.assertRaises(ValueError):
            e.decision(games('confirmation')[:-1], 'confirmation')

    def test_rows_validate_identity_reconciliation_and_skips(self):
        rows = [row(i) for i in range(1, 21)]
        e.check_rows(rows, 20, 20263801)
        self.assertEqual(500, e.summarize(rows)['bulletsBlocked'])
        for bad in (dict(name='AISurvivor'), dict(checkpointLoaded=True), dict(shotOutcomes=dict(blocked=24, hit=2, missed=6)),
                    dict(skippedTurns=1), dict(won=True), dict(bulletsBlocked=31, shotOutcomes=dict(blocked=31, hit=2, missed=0)), dict(shieldsFired=24), dict(meanLabelError=float('nan'))):
            broken = [row(i) for i in range(1, 21)]; broken[3] = row(4, **bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                e.check_rows(broken, 20, 20263801)
        with self.assertRaises(ValueError):
            e.check_rows(rows[:-1], 20, 20263801)
        # refused fire commands in a low-energy endgame must not invalidate the round
        e.check_rows([row(i, shieldsFired=60) for i in range(1, 21)], 20, 20263801)


if __name__ == '__main__':
    unittest.main()
