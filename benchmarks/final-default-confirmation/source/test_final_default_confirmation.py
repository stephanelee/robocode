import copy
import unittest
from final_default_confirmation import decision,expected_layout,JAR
import coherent_duel_v2_experiment as harness


def games(shares=None):
    shares=shares or [.7]*10;result=[]
    for registered,share in zip(expected_layout(),shares):
        battle=dict(rounds=20,firstPlaces=11,totalScore=1000*share,
            results=[dict(totalScore=1000*share),dict(totalScore=1000*(1-share))],
            metrics=dict(roundsRecorded=20,skippedTurns=0,checkpointLoaded=False),
            endpoint=dict(totalRounds=20),mechanism=dict(forcedDuelTurns=10))
        result.append(dict(**registered,rounds=20,kind='Retcon',arms=['candidate'],
            battles=dict(candidate=battle),native=dict(status='complete')))
    return result


class FinalDefaultTests(unittest.TestCase):
    def test_exact_budget_and_preserved_historical_helper(self):
        original=copy.deepcopy(harness.layout())
        result=decision(games())
        self.assertEqual(('confirmed',200,110,10),(result['status'],result['rounds'],result['wins'],result['gamesAboveHalf']))
        self.assertEqual(dict(harness.JARS)['candidate'],JAR)
        self.assertEqual(original,harness.layout())
        with self.assertRaises(ValueError):decision(games()[:-1])
        changed=games();changed[0]['rounds']=19
        with self.assertRaises(ValueError):decision(changed)
        changed=games();changed[0]['gameId']=changed[1]['gameId']
        with self.assertRaises(ValueError):decision(changed)

    def test_t_bound_is_binding_even_when_all_other_performance_gates_pass(self):
        result=decision(games([.2]+[.6]*9))
        self.assertEqual('not-confirmed',result['status'])
        self.assertEqual(['gameMeanTLower95'],[k for k,v in result['checks'].items() if not v])
        self.assertGreater(result['meanShare'],.5);self.assertLess(result['gameMeanTLower95'],.5)

    def test_both_batches_strict_majorities_and_evidence_are_required(self):
        result=decision(games([.7,.4,.4,.4,.4]+[.9]*5))
        self.assertFalse(result['checks']['bothBatches']);self.assertTrue(result['checks']['gameMeanTLower95'])
        same=games([.5]*10);self.assertFalse(decision(same)['checks']['meanShare'])
        changed=games();changed[0]['battles']['candidate']['endpoint']['totalRounds']=19
        self.assertFalse(decision(changed)['checks']['validEvidence'])
        changed=games();changed[0]['battles']['candidate']['metrics']['checkpointLoaded']=True
        self.assertFalse(decision(changed)['checks']['validEvidence'])
        changed=games();changed[0]['battles']['candidate']['firstPlaces']=21
        with self.assertRaises(ValueError):decision(changed)


if __name__=='__main__':unittest.main()
