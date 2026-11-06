import unittest,json,tempfile,gzip
from pathlib import Path
import keelhelm_calibration as c

class CalibrationContracts(unittest.TestCase):
    def test_budget_pair_balance_and_externally_labeled_identical_arms(self):
        rows=c.schedule();self.assertEqual(32,len(rows));self.assertEqual(240,sum(g['rounds']for g in rows))
        for i in range(0,32,2):
            a,b=rows[i:i+2];self.assertEqual({'A','B'},{a['arm'],b['arm']})
            for k in ['pair','layout','mode','scenario','rounds','seed','order']:self.assertEqual(a[k],b[k])
        for mode in ['random','fixed']:
            for scenario in ['melee','Diamond Port']:
                for arm in ['A','B']:self.assertEqual(4,sum(g['mode']==mode and g['scenario']==scenario and g['arm']==arm for g in rows))
    def test_fixed_layouts_are_legal_nonoverlapping_and_cover_each_corner(self):
        for g in c.schedule():
            poses=c.positions(g)
            if g['mode']=='random':self.assertIsNone(poses);continue
            w,h=(1000,1000)if g['scenario']=='melee'else(800,600)
            self.assertEqual(set(g['order']),set(poses));self.assertEqual(len(poses),len({tuple(v[:2])for v in poses.values()}))
            for x,y,d in poses.values():self.assertTrue(18<=x<=w-18 and 18<=y<=h-18 and 0<=d<360)
    def test_exact_continuation_limits_and_pair_divergence(self):
        games=[]
        for g in c.schedule():
            score=30 if g['arm']=='A'else 31
            own=dict(totalScore=score,firstPlaces=1)
            games.append(dict(**g,helm=own,official=dict(results=[own,dict(totalScore=100-score)]),startAudit=dict(trajectories={'1':['a','b']})))
        s=c.summarize(games);self.assertEqual('continue-to-new-movement-protocol',s['decision']);self.assertTrue(all(v is None for p in s['pairs']for v in p['firstStateDivergence'].values()))
        for g in games:
            if g['mode']=='fixed'and g['scenario']=='melee'and g['arm']=='B':g['helm']['totalScore']=40;g['official']['results'][1]['totalScore']=60
        self.assertEqual('stop-measurement-not-ready',c.summarize(games)['decision'])
    def test_native_first_tick_pose_mismatch_is_rejected(self):
        g=next(g for g in c.schedule()if g['mode']=='fixed'and g['scenario']=='Diamond Port');g=dict(g,rounds=1)
        names=sorted(g['order']);poses=c.positions(g);states=[]
        for i,n in enumerate(names,1):
            x,y,d=poses[n];states.append(dict(id=i,x=x,y=y,direction=d,gunDirection=d,radarDirection=d,speed=0,energy=100,gunHeat=0,enemyCount=1))
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'record.gz'
            def write():
                with gzip.open(p,'wt')as f:
                    f.write(json.dumps(dict(type='GameStartedEventForObserver',participants=[dict(id=i,name=n)for i,n in enumerate(names,1)]))+'\n')
                    f.write(json.dumps(dict(type='TickEventForObserver',roundNumber=1,turnNumber=1,botStates=states))+'\n')
            write();self.assertEqual(2,len(c.starts(p,g)['ids']))
            states[0]['x']+=1;write()
            with self.assertRaises(AssertionError):c.starts(p,g)
    def test_lost_handshake_send_is_rejected_even_if_posthandshake_hashes_match(self):
        g=next(g for g in c.schedule()if g['mode']=='fixed'and g['scenario']=='Diamond Port');rows=[]
        for session,name in zip(['a','b'],sorted(g['order'])):
            before=dict(type='BotHandshake',name=name,version='1',sessionId=session);after=dict(before,initialPosition=dict(zip(('x','y','direction'),c.positions(g)[name])))
            rows.append(dict(path='/identity/'+name,name=name,handshakeBefore=json.dumps(before),handshakeAfter=json.dumps(after),serverSessionId=session,rewrites=1,downInput='h',downOutput='h',downInputCount=1,downOutputCount=1,upAfterHandshakeInput='e',upAfterHandshakeOutput='e',upInputCount=1,upOutputCount=1))
        value=dict(complete=True,cleanupErrors=[],relayErrors=[],runnerReturnCode=0,connections=rows)
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'transport.json';p.write_text(json.dumps(value));c.inspect_transport(p,g,[(n,'1')for n in g['order']])
            rows[0]['upOutputCount']=0;p.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):c.inspect_transport(p,g,[(n,'1')for n in g['order']])
if __name__=='__main__':unittest.main()
