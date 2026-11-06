import unittest
from fractions import Fraction
import keelhelm_outcome_power as n
class OutcomePowerContracts(unittest.TestCase):
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
    def make_trace(self,root,forged=False):
        import gzip,json
        raw=root/'battle.gz';events=[dict(type='GameStartedEventForObserver',participants=[dict(id=1,name='KeelHelm'),dict(id=2,name='Prism'),dict(id=3,name='Diamond Port')])];records=[];seq=0
        for i in range(34):
            request=1+4*i;bid=i+1;size=min(i,32);power=2 if size==32 else 1.6;context=seq
            seq+=1;fields=[seq,'F',context,request,request+1,bid,2,1,power,1.6,power,300,100,100,0,size,size]
            if forged and i==33:fields[2]=0
            records.append(','.join(map(str,fields)))
            bullet=dict(bulletId=bid,ownerId=1,power=power,direction=0)
            for t,ev in [(request,[]),(request+1,[dict(type='BulletFiredEvent',bullet=bullet)]),(request+2,[dict(type='BulletHitBulletEvent',bullet=bullet),dict(type='BulletHitBotEvent',bullet=bullet,damage=4)])]:
                events.append(dict(type='TickEventForObserver',roundNumber=1,turnNumber=t,botStates=[dict(id=1,energy=100,enemyCount=3)],events=ev))
            seq+=1;records.append(f'{seq},R,{request+3},{bid},2,1,{request+2},5,4,1,0')
        with gzip.open(raw,'wt')as f:
            for e in events:f.write(json.dumps(e)+'\n')
        telemetry=['1 stats powerTraceOverflow=0 outcomeConfirmed=34 outcomeResolved=34 outcomeSuccesses=34 outcomeCensored=0 powerConfirmed=2 powerTargets=2:2','1 powertrace '+';'.join(records)]
        return raw,telemetry
    def test_actual_power_and_prior_window_replay_accept_multi_outcome_tick(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory()as tmp:
            raw,lines=self.make_trace(Path(tmp));value=n.exposure(raw,lines,1)
            self.assertEqual(2,value['confirmedChanged']);self.assertEqual({'Prism':2},value['namedTargets']);self.assertEqual(34,value['resolvedFeedback'])
    def test_a_forged_request_window_context_is_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory()as tmp:
            raw,lines=self.make_trace(Path(tmp),True)
            with self.assertRaises(AssertionError):n.exposure(raw,lines,1)
if __name__=='__main__':unittest.main()
