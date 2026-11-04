"""Meaningful integrity/corruption fixtures, not gameplay or competitive evidence."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import hashlib

spec = importlib.util.spec_from_file_location('independent_audit', Path(__file__).with_name('audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    people = [dict(id=1, name='KeelHelm', version='0.1.0'), dict(id=2, name='Diamond Port', version='1.8.28')]
    result = [dict(**p, totalScore=s, firstPlaces=w) for p, s, w in zip(people, [75, 50], [1, 0])]
    bullet = lambda owner, power, bid: dict(ownerId=owner, power=power, bulletId=bid)
    return [dict(type='GameStartedEventForObserver', participants=people, gameSetup=dict(gameType='1v1', arenaWidth=800, arenaHeight=600, numberOfRounds=1, turnTimeout=100000)),
        dict(type='TickEventForObserver', roundNumber=1, turnNumber=1, botStates=people, events=[dict(type='BulletFiredEvent', bullet=bullet(o, p, i)) for i, (o, p) in enumerate([(1, .1), (2, 2), (1, 1), (2, .5)], 1)]),
        dict(type='TickEventForObserver', roundNumber=1, turnNumber=2, botStates=people, events=[dict(type='BulletHitBotEvent', bullet=bullet(1, .1, 1), victimId=2, damage=.4),dict(type='BulletHitBotEvent', bullet=bullet(2, 2, 2), victimId=1, damage=10),dict(type='BulletHitBulletEvent', bullet=bullet(1, 1, 3), hitBullet=bullet(2, .5, 4)),dict(type='BotHitWallEvent', victimId=1),dict(type='BotHitBotEvent', botId=1, victimId=2)]),
        dict(type='RoundEndedEventForObserver', roundNumber=1, turnNumber=2, results=result),
        dict(type='GameEndedEventForObserver', numberOfRounds=1, results=result)]


class IntegrityTests(unittest.TestCase):
    def reduce(self, events=None, stats=None):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp);recording = p/'synthetic.battle.gz';marker = p/'markers.txt'
            with gzip.open(recording, 'wt') as f:
                for row in fixture() if events is None else events:
                    f.write(json.dumps(row)+'\n')
            marker.write_text(stats or '1 start\n1 stats maxMs=5 stoppedRolloutSteps=8 headingChangedSteps=4 secondWaveEvaluations=2 executedStopNow=1\n1 end\n')
            return audit.reduce_game(recording, marker, 'KeelHelm', 'Diamond Port', '1.8.28', 1, True)

    def test_full_metrics_and_exact_ratio(self):
        r=self.reduce();self.assertTrue(r['valid'],r);self.assertTrue(r['timingPass'])
        self.assertEqual([r[k] for k in ['shots','hits','enemyShots','enemyHits','wallContacts','botContacts','bulletCollisions']], [2,1,2,1,1,1,1])
        self.assertEqual(r['exactShare']['numerator'],3);self.assertEqual(r['exactShare']['denominator'],5)
        self.assertEqual(r['exposure']['headingChangedSteps'],4)

    def test_missing_tick_retains_physical_lower_bound(self):
        rows=fixture();rows[2]['turnNumber']=3;r=self.reduce(rows)
        self.assertFalse(r['valid']);self.assertEqual(r['observedPhysicalRoundIds'],[1]);self.assertIn('tick',r['error'])

    def test_duplicate_endpoint_invalid(self):
        rows=fixture();rows.insert(4,rows[3]);self.assertFalse(self.reduce(rows)['valid'])

    def test_unknown_participant_and_bullet_invalid(self):
        rows=fixture();rows[1]['events'][0]['bullet']['ownerId']=9;self.assertFalse(self.reduce(rows)['valid'])
        rows=fixture();rows[0]['participants'][1]['version']='unregistered';self.assertFalse(self.reduce(rows)['valid'])

    def test_max75_is_rejection_not_rewritten_value(self):
        stats='1 start\n1 stats maxMs=75 stoppedRolloutSteps=8 headingChangedSteps=4 secondWaveEvaluations=2 executedStopNow=1\n1 end\n'
        r=self.reduce(stats=stats);self.assertTrue(r['valid']);self.assertFalse(r['timingPass']);self.assertEqual(r['maxMs'],75)

    def test_duplicate_stats_and_negative_timing_invalid(self):
        stats='1 start\n1 stats maxMs=-1\n1 end\n';self.assertFalse(self.reduce(stats=stats)['valid'])
        stats='1 start\n1 stats maxMs=1\n1 stats maxMs=2\n1 end\n';self.assertFalse(self.reduce(stats=stats)['valid'])

    def test_skip_is_retained(self):
        stats='1 start\n1 skip 1\n1 stats maxMs=1 stoppedRolloutSteps=8 headingChangedSteps=4 secondWaveEvaluations=2 executedStopNow=1\n1 end\n'
        r=self.reduce(stats=stats);self.assertTrue(r['valid']);self.assertEqual(r['skips'],1);self.assertFalse(r['timingPass'])

    def test_incomplete_recording_not_filled(self):
        r=self.reduce(fixture()[:3]);self.assertFalse(r['valid']);self.assertEqual(r['observedPhysicalRoundIds'],[1]);self.assertEqual(r['observedRoundEndpointIds'],[])


class GateBoundaryTests(unittest.TestCase):
    def groups(self, stage):
        rows=[]
        configs=[('diamond',3 if stage=='screen' else 6,53 if stage=='screen' else 52,50,6 if stage=='screen' else 11,10 if stage=='screen' else 20),('retcon',2 if stage=='screen' else 3,90,93 if stage=='screen' else 92,9 if stage=='screen' else 8,10)]
        if stage=='screen':configs += [('walls',1,90,95,4,5),('ais',1,90,95,4,5)]
        for opponent,n,c,b,w,rounds in configs:
            for arm,score in [('candidate',c),('baseline',b)]:
                for _ in range(n):rows.append(dict(game=dict(bot=arm,opponent=opponent,rounds=rounds),ownScore=score,totalScore=100,wins=w,timingPass=True))
        return rows

    def test_screen_fractional_boundaries(self):
        rows=self.groups('screen');self.assertTrue(audit.stage_gates('screen',rows)['allPass'])
        for r in rows:
            if r['game']['opponent']=='diamond' and r['game']['bot']=='candidate':r.update(ownScore=529,totalScore=1000)
        self.assertFalse(audit.stage_gates('screen',rows)['checks']['diamondDeltaAtLeast3pp'])

    def test_confirmation_fractional_boundaries(self):
        rows=self.groups('confirmation');self.assertTrue(audit.stage_gates('confirmation',rows)['allPass'])
        rows[0]['timingPass']=False;self.assertFalse(audit.stage_gates('confirmation',rows)['allPass'])


class ArchiveTests(unittest.TestCase):
    def test_exact_zip_members_and_corrupt_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'files';out.mkdir();(out/'raw.txt').write_bytes(b'unchanged raw\n')
            z=p/'original.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('raw.txt',b'unchanged raw\n')
            h=hashlib.sha256(z.read_bytes()).hexdigest()
            self.assertTrue(audit.audit_archive(z,h,out)['pass'])
            (out/'raw.txt').write_bytes(b'edited')
            with self.assertRaises(ValueError):audit.audit_archive(z,h,out)

    def test_unsafe_or_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);z=p/'original.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../outside.txt',b'bad')
            h=hashlib.sha256(z.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):audit.audit_archive(z,h,p)


if __name__=='__main__':unittest.main()
