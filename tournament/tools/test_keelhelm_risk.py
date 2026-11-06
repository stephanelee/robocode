import math,unittest
from fractions import Fraction
import keelhelm_risk as c
class RiskContracts(unittest.TestCase):
    def test_frozen_budget_geometry_assignments_and_arm_isolation(self):
        rows=c.make_schedule(list(range(16)))
        self.assertEqual((88,520),(len(rows),sum(r['rounds']for r in rows)))
        for a,b in zip(rows[::2],rows[1::2]):
            for key in ['pair','positions','rounds','seed','order','scenario']:self.assertEqual(a[key],b[key])
            self.assertEqual({'control','candidate'},{a['arm'],b['arm']})
            self.assertEqual([0,1],[a['slot'],b['slot']])
            points=list(a['positions'].values())
            self.assertTrue(all(math.dist(x[:2],y[:2])>=250 for i,x in enumerate(points)for y in points[:i]))
        self.assertEqual(16,sum(g['scenario']=='melee'and g['slot']==0 and g['arm']=='candidate'for g in rows))
    def test_reducer_detects_zero_effect_instead_of_promoting_ties(self):
        games=[]
        for g in c.make_schedule(list(range(16))):
            own=dict(totalScore=30,survival=30,firstPlaces=0)
            games.append(dict(**g,helm=own,official=dict(results=[own,dict(totalScore=70,survival=70)]),movementDifferences=[100]))
        result=c.summarize(games,list(range(16)))
        self.assertEqual('1',result['randomization']['pExact'])
        self.assertFalse(result['gates']['meleeGain']);self.assertFalse(result['gates']['randomization'])
        self.assertEqual('reject-improvement-claim',result['decision'])
if __name__=='__main__':unittest.main()
