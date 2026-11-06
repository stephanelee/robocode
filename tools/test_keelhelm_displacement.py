import unittest
from fractions import Fraction
import keelhelm_displacement as n
class DisplacementContracts(unittest.TestCase):
    def test_schedule_balanced_budget_and_cold_pairs(self):
        rows=n.schedule();self.assertEqual(40,len(rows));self.assertEqual(520,sum(g['rounds']for g in rows))
        for i in range(0,40,2):
            a,b=rows[i:i+2];self.assertEqual({a['arm'],b['arm']},{'control','candidate'});self.assertEqual(a['seed'],b['seed']);self.assertEqual(a['order'],b['order'])
        for arm in ('control','candidate'):
            self.assertEqual([8,4,4,2,2],[sum(g['arm']==arm and g['scenario']==s for g in rows)for s in ('melee','Diamond Port','Retcon','Walls','SpinBot')])
    def test_survival_and_actual_named_exposure_are_required(self):
        rows=[]
        for scenario in ('melee','Diamond Port','Retcon','Walls','SpinBot'):
            for arm,score in [('control',25),('candidate',30)]:
                own=dict(totalScore=score,survival=30,firstPlaces=1)
                rows.append(dict(scenario=scenario,arm=arm,helm=own,official=dict(results=[own,dict(totalScore=100-score,survival=70)]),exposure=dict(confirmedChanged=100,namedTargets={'Prism':50,'Diamond Port':50})))
        self.assertEqual('accept',n.summary(rows)['decision'])
        rows[1]['exposure']['namedTargets']={'Prism':100};self.assertEqual('reject',n.summary(rows)['decision'])
        rows[1]['exposure']['namedTargets']={'Prism':50,'Diamond Port':50};rows[1]['helm']['survival']=0
        self.assertFalse(n.summary(rows)['gates']['survival'])
if __name__=='__main__':unittest.main()
