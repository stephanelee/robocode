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
    people = [dict(id=1, name='KeelBrace', version='0.1.0'), dict(id=2, name='Diamond Port', version='1.8.28')]
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
            marker.write_text(stats or '1 start\n1 stats maxMs=5 rawRamEvents=1 eligibleRamCorrectionEvents=1 eligibleRamCorrectedScans=1 ramChangedPowerScans=1 ramChangedGateScans=1\n1 end\n')
            return audit.reduce_game(recording, marker, 'KeelBrace', 'Diamond Port', '1.8.28', 1, True)

    def test_full_metrics_and_exact_ratio(self):
        r=self.reduce();self.assertTrue(r['valid'],r);self.assertTrue(r['timingPass'])
        self.assertEqual([r[k] for k in ['shots','hits','enemyShots','enemyHits','wallContacts','botContacts','bulletCollisions']], [2,1,2,1,1,1,1])
        self.assertEqual(r['exactShare']['numerator'],3);self.assertEqual(r['exactShare']['denominator'],5)
        self.assertEqual(r['exposure']['eligibleRamCorrectedScans'],1)

    def test_missing_tick_retains_physical_lower_bound(self):
        rows=fixture();rows[2]['turnNumber']=3;r=self.reduce(rows)
        self.assertFalse(r['valid']);self.assertEqual(r['observedPhysicalRoundIds'],[1]);self.assertIn('tick',r['error'])

    def test_duplicate_endpoint_invalid(self):
        rows=fixture();rows.insert(4,rows[3]);self.assertFalse(self.reduce(rows)['valid'])

    def test_unknown_participant_and_bullet_invalid(self):
        rows=fixture();rows[1]['events'][0]['bullet']['ownerId']=9;self.assertFalse(self.reduce(rows)['valid'])
        rows=fixture();rows[0]['participants'][1]['version']='unregistered';self.assertFalse(self.reduce(rows)['valid'])

    def test_max75_is_rejection_not_rewritten_value(self):
        stats='1 start\n1 stats maxMs=75 rawRamEvents=1 eligibleRamCorrectionEvents=1 eligibleRamCorrectedScans=1 ramChangedPowerScans=1 ramChangedGateScans=1\n1 end\n'
        r=self.reduce(stats=stats);self.assertTrue(r['valid']);self.assertFalse(r['timingPass']);self.assertEqual(r['maxMs'],75)

    def test_duplicate_stats_and_negative_timing_invalid(self):
        stats='1 start\n1 stats maxMs=-1\n1 end\n';self.assertFalse(self.reduce(stats=stats)['valid'])
        stats='1 start\n1 stats maxMs=1\n1 stats maxMs=2\n1 end\n';self.assertFalse(self.reduce(stats=stats)['valid'])

    def test_skip_is_retained(self):
        stats='1 start\n1 skip 1\n1 stats maxMs=1 rawRamEvents=1 eligibleRamCorrectionEvents=1 eligibleRamCorrectedScans=1 ramChangedPowerScans=1 ramChangedGateScans=1\n1 end\n'
        r=self.reduce(stats=stats);self.assertTrue(r['valid']);self.assertEqual(r['skips'],1);self.assertFalse(r['timingPass'])

    def test_incomplete_recording_not_filled(self):
        r=self.reduce(fixture()[:3]);self.assertFalse(r['valid']);self.assertEqual(r['observedPhysicalRoundIds'],[1]);self.assertEqual(r['observedRoundEndpointIds'],[])


class GlobalMarkerOrderTests(unittest.TestCase):
    def test_rounds_cannot_interleave_or_reverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'markers.txt'
            for text in ['1 start\n2 start\n1 stats maxMs=1\n1 end\n2 stats maxMs=1\n2 end\n','2 start\n2 stats maxMs=1\n2 end\n1 start\n1 stats maxMs=1\n1 end\n']:
                p.write_text(text)
                with self.assertRaises(ValueError):audit.markers(p,2,False)
            p.write_text('1 start\n1 stats maxMs=1\n1 end\n2 start\n2 stats maxMs=1\n2 end\n');self.assertEqual(len(audit.markers(p,2,False)),2)


class GateBoundaryTests(unittest.TestCase):
    def groups(self, stage):
        rows=[]
        configs=[('diamond',3 if stage=='screen' else 6,52,55 if stage=='screen' else 54,6 if stage=='screen' else 11,10 if stage=='screen' else 20),('retcon',2 if stage=='screen' else 3,94,93,9 if stage=='screen' else 8,10)]
        if stage=='screen':configs += [('walls',1,90,95,4,5),('ais',1,90,95,4,5)]
        for opponent,n,c,b,w,rounds in configs:
            for arm,score in [('candidate',c),('baseline',b)]:
                for _ in range(n):rows.append(dict(game=dict(bot=arm,opponent=opponent,rounds=rounds),ownScore=score,totalScore=100,wins=w,timingPass=True,exposure={k:1 for k in audit.EXPOSURE} if arm=='candidate' else None))
        return rows

    def test_screen_fractional_boundaries(self):
        rows=self.groups('screen');self.assertTrue(audit.stage_gates('screen',rows)['allPass'])
        for r in rows:
            if r['game']['opponent']=='diamond' and r['game']['bot']=='candidate':r.update(ownScore=519,totalScore=1000)
        self.assertFalse(audit.stage_gates('screen',rows)['checks']['diamondDeltaAtLeastMinus3pp'])

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



class BraceCounterTests(unittest.TestCase):
    reduce=IntegrityTests.reduce
    def stats(self, **values):
        d={k:1 for k in audit.EXPOSURE};d.update(values)
        return '1 start\n1 stats maxMs=1 '+' '.join(k+'='+str(v) for k,v in d.items())+'\n1 end\n'

    def test_counter_cap_negative_duplicate_noninteger(self):
        for value in [-1,1_000_001,'1.0','true']:
            self.assertFalse(self.reduce(stats=self.stats(rawRamEvents=value))['valid'])
        bad=self.stats().replace('rawRamEvents=1','rawRamEvents=1 rawRamEvents=1')
        self.assertFalse(self.reduce(stats=bad)['valid'])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'markers';p.write_text(self.stats(**{k:1_000_000 for k in audit.EXPOSURE}))
            self.assertEqual(audit.markers(p,1,True)[1]['exposure']['rawRamEvents'],1_000_000)

    def test_counter_chain_overflow_fault_and_official_contact_bound(self):
        for values in [dict(ramChangedGateScans=2),dict(eligibleRamCorrectedScans=2),dict(eligibleRamCorrectionEvents=2)]:
            self.assertFalse(self.reduce(stats=self.stats(**values))['valid'])
        self.assertFalse(self.reduce(stats=self.stats().replace('1 end','1 diagnosticOverflow rawRamEvents\n1 end'))['valid'])

    def test_raw_contact_corrob_and_no_private_eligibility_invention(self):
        self.assertFalse(self.reduce(stats=self.stats(rawRamEvents=2,eligibleRamCorrectionEvents=2))['valid'])
        # A delivered event need not reach an eligible scan; raw count alone is not exposure.
        r=self.reduce(stats=self.stats(eligibleRamCorrectionEvents=0,eligibleRamCorrectedScans=0,ramChangedPowerScans=0,ramChangedGateScans=0))
        self.assertTrue(r['valid'],r);self.assertEqual(r['exposure']['eligibleRamCorrectedScans'],0)

    def test_duplicate_start_missing_end_and_baseline_na(self):
        self.assertFalse(self.reduce(stats=self.stats().replace('1 start','1 start\n1 start'))['valid'])
        self.assertFalse(self.reduce(stats=self.stats().replace('1 end\n',''))['valid'])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'markers';p.write_text(self.stats())
            with self.assertRaises(ValueError):audit.markers(p,1,False)

    def test_ties_preserve_official_firstplaces(self):
        rows=fixture();rows[-2]['results'][1]['firstPlaces']=1
        rows[-1]['results']=rows[-2]['results']
        r=self.reduce(rows);self.assertTrue(r['valid'],r);self.assertEqual(r['wins'],1)

    def test_corrupt_gzip_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);recording=p/'bad.gz';marker=p/'m';marker.write_text(self.stats())
            data=gzip.compress(('\n'.join(json.dumps(x) for x in fixture())+'\n').encode());data=data[:-8]+bytes([data[-8]^1])+data[-7:];recording.write_bytes(data)
            self.assertFalse(audit.reduce_game(recording,marker,'KeelBrace','Diamond Port','1.8.28',1,True)['valid'])
            with gzip.open(recording,'wt') as f:f.write('{"type":"x","type":"y"}\n')
            self.assertFalse(audit.reduce_game(recording,marker,'KeelBrace','Diamond Port','1.8.28',1,True)['valid'])


class BraceGateTests(unittest.TestCase):
    groups=GateBoundaryTests.groups
    def test_retcon_exact_plus1_below_boundary_and_ceiling(self):
        rows=self.groups('screen');self.assertTrue(audit.stage_gates('screen',rows)['allPass'])
        for r in rows:
            if r['game']['bot']=='candidate' and r['game']['opponent']=='retcon':r.update(ownScore=939,totalScore=1000)
        self.assertFalse(audit.stage_gates('screen',rows)['checks']['retconDeltaAtLeast1pp'])
        for r in rows:
            if r['game']['opponent']=='retcon':r.update(ownScore=100,totalScore=100)
        self.assertFalse(audit.stage_gates('screen',rows)['allPass']) # no post-result ceiling waiver

    def test_confirmation_minus2_exact_then_below(self):
        rows=self.groups('confirmation');self.assertTrue(audit.stage_gates('confirmation',rows)['allPass'])
        for r in rows:
            if r['game']['bot']=='candidate' and r['game']['opponent']=='diamond':r.update(ownScore=519,totalScore=1000)
        self.assertFalse(audit.stage_gates('confirmation',rows)['checks']['diamondDeltaAtLeastMinus2pp'])

    def test_exposure_is_stage_local_retcon_not_raw_contacts_or_diamond(self):
        rows=self.groups('confirmation')
        for r in rows:
            if r['game']['bot']=='candidate' and r['game']['opponent']=='retcon':
                r['exposure']={k:0 for k in audit.EXPOSURE};r['exposure']['rawRamEvents']=100
        g=audit.stage_gates('confirmation',rows);self.assertEqual(g['rawDisposition'],'INCONCLUSIVE');self.assertFalse(g['allPass']);self.assertTrue(g['scoreAndTimingPass'])
        next(r for r in rows if r['game']['bot']=='candidate' and r['game']['opponent']=='retcon')['exposure']={k:1 for k in audit.EXPOSURE}
        self.assertTrue(audit.stage_gates('confirmation',rows)['allPass'])

    def test_inconclusive_precedes_score_rejection_but_not_integrity(self):
        rows=self.groups('screen')
        for r in rows:
            if r['game']['bot']=='candidate' and r['game']['opponent']=='retcon':r['exposure']={k:0 for k in audit.EXPOSURE};r['ownScore']=1
        self.assertEqual(audit.stage_gates('screen',rows)['rawDisposition'],'INCONCLUSIVE')
        rows[0]['timingPass']=False
        self.assertEqual(audit.stage_gates('screen',rows)['rawDisposition'],'INVALID/INCOMPLETE')

    def test_strict_half_and_win_count_boundaries(self):
        rows=self.groups('confirmation')
        for r in rows:
            if r['game']['bot']=='candidate' and r['game']['opponent']=='diamond':r.update(ownScore=50,wins=10)
        g=audit.stage_gates('confirmation',rows)
        self.assertFalse(g['checks']['candidateDiamondMeanAboveHalf']);self.assertFalse(g['checks']['diamondWinsMoreThan60'])

    def test_equal_game_not_pooled_score_arithmetic(self):
        rows=self.groups('confirmation')
        group=[r for r in rows if r['game']['bot']=='candidate' and r['game']['opponent']=='diamond']
        group[0].update(ownScore=10000,totalScore=10000);group[1].update(ownScore=0,totalScore=1)
        mean=audit.group_summary(rows)['candidate/diamond']['mean']
        self.assertEqual((mean['numerator'],mean['denominator']),(77,150))


class StreamingArchiveTests(unittest.TestCase):
    def test_streaming_replay_without_materializing_recording(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);z=p/'z.zip';raw=gzip.compress(('\n'.join(json.dumps(x) for x in fixture())+'\n').encode())
            stats='1 start\n1 stats maxMs=1 '+' '.join(k+'=1' for k in audit.EXPOSURE)+'\n1 end\n'
            with zipfile.ZipFile(z,'w') as f:f.writestr('stage/raw.battle.gz',raw);f.writestr('stage/markers.txt',stats)
            digest=hashlib.sha256(z.read_bytes()).hexdigest();proof=audit.audit_archive(z,digest)
            self.assertEqual(len(proof['wholeRecordings']),1);self.assertEqual(proof['fileCount'],2)
            store=audit.ZipStore(z)
            try:
                v=store.root('stage');r=audit.reduce_game(v/'raw.battle.gz',v/'markers.txt','KeelBrace','Diamond Port','1.8.28',1,True)
                self.assertTrue(r['valid'],r);self.assertEqual(r['wins'],1)
            finally:store.close()
            self.assertFalse((p/'raw.battle.gz').exists())

    def test_duplicate_alias_absolute_drive_symlink_and_digest_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for name in ['a/../b','a//b','./a','/a','C:bad','a\\b']:
                z=p/'bad.zip'
                with zipfile.ZipFile(z,'w') as f:f.writestr(name,b'x')
                with self.assertRaises(ValueError):audit.audit_archive(z,hashlib.sha256(z.read_bytes()).hexdigest())
            z=p/'bad.zip';info=zipfile.ZipInfo('link');info.external_attr=0o120777<<16
            with zipfile.ZipFile(z,'w') as f:f.writestr(info,b'target')
            with self.assertRaises(ValueError):audit.audit_archive(z,hashlib.sha256(z.read_bytes()).hexdigest())
            with zipfile.ZipFile(z,'w') as f:f.writestr('ok',b'x')
            with self.assertRaises(ValueError):audit.audit_archive(z,'0'*64)

    def test_corrupt_zip_crc_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            z=Path(t)/'bad.zip'
            with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_STORED) as f:f.writestr('raw.txt',b'original raw data')
            data=z.read_bytes().replace(b'original raw data',b'changed! raw data');z.write_bytes(data)
            with self.assertRaises(zipfile.BadZipFile):audit.audit_archive(z,hashlib.sha256(data).hexdigest())


class AuthorityFailClosedTests(unittest.TestCase):
    def test_missing_authority_and_storage_are_not_acceptance(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            self.assertFalse(audit.audit_authority(p,p,'screen','a'*40,1)['pass'])
            self.assertFalse(audit.audit_storage(p,'screen',1)['pass'])

    def test_strict_body_parser_duplicate_nonfinite_and_typed_json(self):
        for data in ['{"a":1,"a":2}','{"a":NaN}']:
            with self.assertRaises(ValueError):audit.json_data(data)
        self.assertIs(audit.json_data('{"schemaVersion":true}')['schemaVersion'],True) # typed validation remains separate

    def test_timestamp_offset_required(self):
        with self.assertRaises(ValueError):audit.timestamp('2026-11-04T20:00:00')
        self.assertEqual(audit.timestamp('2026-11-04T20:00:00Z'),audit.timestamp('2026-11-04 20:00:00 UTC'))


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')


def authority_fixture(root, approval_overrides=None):
    evidence=root/'evidence';source=root/'source';provider=root/'provider';evidence.mkdir();provider.mkdir();head='c'*40;runid=7
    at=lambda m:f'2026-11-04T00:{m:02d}:00Z'
    reviewer='/independent';record=dict(reviewerTask=reviewer,activeBeforeDispatch=True,tool='recorded independent assignment')
    dispatch=dict(schemaVersion=1,repository='stephanelee/robocode',issue=114,stage='screen',headSHA=head,reviewerTask=reviewer,reviewerAccount='stephanelee',implementationTasks=['/implementation'],assignmentToolRecord=record,assignmentToolRecordSHA256=hashlib.sha256(audit.canonical(record)).hexdigest())
    def comment(cid,prefix,payload,minute):return dict(id=cid,node_id='IC_'+str(cid),url='https://api.github.com/repos/stephanelee/robocode/issues/comments/'+str(cid),issue_url='https://api.github.com/repos/stephanelee/robocode/issues/114',html_url='https://github.com/stephanelee/robocode/issues/114#issuecomment-'+str(cid),body=prefix+'\n'+json.dumps(payload),created_at=at(minute),updated_at=at(minute),user=dict(login='stephanelee',id=30698997,type='User'),author_association='OWNER',reactions={'total_count':0})
    dc=comment(1,'KEEL_BRACE_DISPATCH_ASSIGNMENT_V1',dispatch,0)
    approval={k:'a'*64 for k in audit.APPROVAL_KEYS};approval.update(schemaVersion=1,verdict='CLEAR',repository='stephanelee/robocode',issue=114,stage='screen',headSHA=head,runId=runid,runAttempt=1,bootId='synthetic-boot',workspace='/synthetic',artifactId=99,reviewerTask=reviewer,dispatchAssignmentSHA256=hashlib.sha256(dc['body'].encode()).hexdigest())
    if approval_overrides:approval.update(approval_overrides)
    payload_hash=hashlib.sha256(audit.canonical(approval)).hexdigest()
    report=dict(approvalPayloadSHA256=payload_hash,reviewerTask=reviewer,reviewReport='Pure synthetic authority fixture, not runtime approval.',agentToolProvenance=dict(reviewerTask=reviewer,reportGeneratedBeforeGames=True))
    rc=comment(2,'KEEL_BRACE_RUNTIME_REVIEW_V1',report,4);ac=comment(3,'KEEL_BRACE_RUNTIME_APPROVAL_V1',approval,5)
    comments={'approval-comment.json':ac,'review-report-comment.json':rc,'dispatch-assignment-comment.json':dc}
    for n,c in comments.items():dump(evidence/n,c);dump(provider/'comments'/(str(c['id'])+'.json'),c)
    authority=dict(schemaVersion=1,approvalPayload=approval,approvalPayloadSHA256=payload_hash,comments={name:audit.comment_identity(c) for name,c in comments.items()},originalAcceptedCopiesSHA256={n:audit.digest(evidence/n) for n in comments})
    dump(evidence/'accepted-authority.json',authority);dump(evidence/'approval.json',approval)
    artifact=dict(id=99,digest='sha256:'+approval['artifactDigestSHA256'],updated_at=at(3),workflow_run=dict(id=runid,head_sha=head));dump(evidence/'artifact.json',artifact)
    payload=dict(acceptedAuthority=authority,acceptedAuthoritySHA256=hashlib.sha256(audit.canonical(authority)).hexdigest());tag=dict(sha='b'*40,object=dict(sha=head,type='commit'),message=json.dumps(payload))
    ref=dict(object=dict(sha=tag['sha'],type='tag'));dump(evidence/'receipt.json',dict(payload=payload,annotatedTag=tag,createdRef=ref));dump(provider/'receipt-ref.json',ref);dump(provider/'receipt-tag.json',tag)
    game=dict(id='brace-s01',bot='baseline',opponent='diamond',rounds=10,seed=910001,port=33511);dump(source/'benchmarks/keel-brace-v1/games.json',{'screen':[game]})
    for phase,minute in [('before-receipt',6),('before-brace-s01',7)]:
        v=dict(schemaVersion=1,stage='screen',phase=phase,checkedAtUTC=at(minute),status='valid',acceptedAuthoritySHA256=payload['acceptedAuthoritySHA256'],currentComments=comments,currentArtifact=artifact,currentPullRequest=dict(head=dict(sha=head),body='KEEL-BRACE-SOURCE-CLEARED '+head),scannedComments=list(comments.values()))
        if phase!='before-receipt':v.update(currentReceiptRef=ref,currentReceiptTag=tag)
        dump(evidence/'approval-revalidations'/(phase+'.json'),v)
    prepared=dict(command=['synthetic-java'],environment={},cwd='/synthetic');dump(evidence/'games/brace-s01/prepared-game.json',prepared)
    dump(evidence/'games/brace-s01/attempt.json',dict(prepared,game=game,headSHA=head,startUTC=at(8),freshProcess=True,freshWritableDirectory=True,importedModels=[]))
    dump(evidence/'games/brace-s01/teardown.json',dict(ownedProcessGroup=1234,termSent=False,killSent=False,hardDeadlineRespected=True,hardLimitSeconds=1800,runStopReserveSeconds=16,endUTC='2026-11-04T00:08:30Z',elapsedMonotonicSeconds=30.0))
    run=dict(id=runid,head_sha=head,run_attempt=1,status='completed',conclusion='success',created_at=at(1),updated_at=at(10));dump(provider/'run.json',run);dump(provider/'workflow-runs.json',dict(workflow_runs=[run]));dump(provider/'jobs.json',dict(jobs=[dict(conclusion='success',started_at=at(1),completed_at=at(10),steps=[dict(name='Run fixed population',completed_at=at(9))])]))
    return evidence,source,provider,head,runid


class AuthorityPositiveTests(unittest.TestCase):
    def test_complete_original_live_authority_and_current_provider(self):
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));result=audit.audit_authority(e,s,'screen',h,r,p)
            self.assertTrue(result['pass'],result)
            self.assertFalse(audit.audit_authority(e,s,'screen',h,r,None)['pass'])

    def test_normalized_pin_keeps_full_api_and_exact_stable_fields(self):
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));authority=json.loads((e/'accepted-authority.json').read_text());original=json.loads((e/'approval-comment.json').read_text());pin=authority['comments']['approval-comment.json']
            self.assertNotEqual(pin,original);self.assertEqual(set(pin),{'id','node_id','url','issue_url','html_url','created_at','updated_at','body','author','bodySHA256'})
            self.assertEqual(pin['author'],{'id':30698997,'login':'stephanelee','type':'User'})
            self.assertEqual(pin['bodySHA256'],hashlib.sha256(original['body'].encode('utf-8')).hexdigest())
            self.assertTrue(audit.audit_authority(e,s,'screen',h,r,p)['pass'])
    def test_pin_mutations_and_current_provider_stable_identity_fail(self):
        for mutation in ['bodyhash','unknown','author','node','providerurl','originalbytes']:
            with tempfile.TemporaryDirectory() as t:
                e,s,p,h,r=authority_fixture(Path(t));f=e/'accepted-authority.json';d=json.loads(f.read_text());pin=d['comments']['approval-comment.json']
                if mutation=='bodyhash':pin['bodySHA256']='0'*64
                elif mutation=='unknown':pin['extra']=True
                elif mutation=='author':pin['author']['type']='Bot'
                elif mutation=='node':pin['node_id']='wrong'
                elif mutation=='providerurl':
                    f=p/'comments/3.json';d=json.loads(f.read_text());d['url']='https://api.github.com/wrong'
                else:
                    f=e/'approval-comment.json';d=json.loads(f.read_text());d['reactions']['total_count']=1
                dump(f,d);self.assertFalse(audit.audit_authority(e,s,'screen',h,r,p)['pass'],mutation)

    def test_provider_replay_and_live_edited_timestamp_fail(self):
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));f=p/'workflow-runs.json';d=json.loads(f.read_text());d['workflow_runs']*=2;dump(f,d)
            self.assertFalse(audit.audit_authority(e,s,'screen',h,r,p)['pass'])
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));f=e/'approval-revalidations/before-brace-s01.json';d=json.loads(f.read_text());d['currentComments']['approval-comment.json']['updated_at']='2026-11-04T00:06:01Z';dump(f,d)
            self.assertFalse(audit.audit_authority(e,s,'screen',h,r,p)['pass'])

    def test_wrong_issue_account_dispatch_order_and_revocation_fail(self):
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));f=p/'run.json';d=json.loads(f.read_text());d['created_at']='2026-11-03T23:00:00Z';dump(f,d)
            self.assertFalse(audit.audit_authority(e,s,'screen',h,r,p)['pass'])
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));f=e/'approval-revalidations/before-brace-s01.json';d=json.loads(f.read_text());d['scannedComments'].append(dict(user=dict(login='stephanelee',id=30698997),body='KEEL_BRACE_RUNTIME_REVOCATION_V1\n'+json.dumps(dict(runId=r,headSHA=h,stage='screen',approvalCommentId=3))));dump(f,d)
            self.assertFalse(audit.audit_authority(e,s,'screen',h,r,p)['pass'])
        for field,value in [('issue',107),('schemaVersion',True),('verdict','REJECT')]:
            with tempfile.TemporaryDirectory() as t:
                e,s,p,h,r=authority_fixture(Path(t),{field:value});result=audit.audit_authority(e,s,'screen',h,r,p)
                self.assertFalse(result['pass']);self.assertIn('accepted authority exact original approval payload',result['checked'])


def storage_fixture(root):
    e=root/'evidence';p=root/'provider';c=root/'companion';e.mkdir();p.mkdir();c.mkdir();at=lambda m:f'2026-11-04T00:{m:02d}:00Z';runid=7;head='c'*40
    pre=dict(host='fixture-host',volume='/fixture-volume',observedAtUTC=at(0),freeBytes=8*1024**3,toolEvidence={'scope':'synthetic recorded free-space observation'})
    dump(e/'dispatch-assignment.json',dict(assignmentToolRecord=dict(storagePreflight=pre)));dump(e/'dispatch-assignment-comment.json',dict(created_at=at(1)))
    for name,minute,minimum in [('before-preparation',3,20*1024**3),('before-receipt',4,8*1024**3),('before-brace-s01',5,8*1024**3)]:
        dump(e/'storage-admissions'/(name+'.json'),dict(schemaVersion=1,phase=name,observedAtUTC=at(minute),volume='/fixture-worker',freeBytes=minimum,minimumFreeBytes=minimum,stageOutputBytes=100,stageOutputLimitBytes=4*1024**3))
    dump(e/'commands.json',[dict(id='brace-s01')]);dump(e/'games/brace-s01/attempt.json',dict(startUTC=at(6)))
    dump(e/'storage-monitor.json',dict(schemaVersion=1,firstObservedAtUTC=at(3),lastObservedAtUTC=at(6),observations=2,maximumStageOutputBytes=100,limitBytes=4*1024**3,limitExceeded=False))
    run=dict(id=runid,head_sha=head,created_at=at(2),updated_at=at(10),status='completed',conclusion='success');dump(p/'run.json',run)
    original=dict(id=10,size_in_bytes=123,digest='sha256:'+'a'*64,workflow_run=dict(id=runid,head_sha=head),updated_at=at(7))
    companion=dict(id=11,name='keel-brace-v1-screen-custody-validation-7',size_in_bytes=124,digest='sha256:'+'b'*64,workflow_run=dict(id=runid,head_sha=head))
    dump(p/'evidence-artifact.json',original);dump(p/'custody-artifact.json',companion);dump(c/'evidence-artifact.json',original);dump(e/'decision.json',{})
    dump(c/'export-validation.json',dict(schemaVersion=1,stage='screen',status='valid',observedAtUTC=at(8),originalZIPMaximumBytes=2*1024**3,rawDecisionSHA256=audit.digest(e/'decision.json'),artifact=original))
    transport=dict(originalZIPBytes=123,originalZIPSHA256='a'*64,custodyZIPBytes=124,custodyZIPSHA256='b'*64,custodyRoot=c)
    return e,p,transport


class RevocationPolicyTests(unittest.TestCase):
    def check_case(self,which):
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));approval=json.loads((e/'approval.json').read_text());accepted=json.loads((e/'approval-comment.json').read_text())
            rev={k:approval[k] for k in ('repository','issue','stage','headSHA','runId','runAttempt','bootId','artifactId','reviewerTask')};rev.update(schemaVersion=1,verdict='REVOKE',approvalCommentId=3,reason='Synthetic targeted revocation')
            comment=dict(user=dict(login='stephanelee',id=30698997),created_at='2026-11-04T00:06:30Z')
            if which.startswith('foreign'):comment['user']=dict(login='unrelated',id=1)
            if which=='wrong-target':rev['approvalCommentId']=99
            if which=='missing-target':rev.pop('approvalCommentId')
            if which=='authorized-bad':rev['bootId']='wrong-worker'
            if which=='authorized-early':comment['created_at']=accepted['created_at']
            comment['body']='KEEL_BRACE_RUNTIME_REVOCATION_V1\n'+('malformed' if which=='foreign-malformed' else json.dumps(rev))
            f=e/'approval-revalidations/before-brace-s01.json';d=json.loads(f.read_text());d['scannedComments'].append(comment);dump(f,d)
            return audit.audit_authority(e,s,'screen',h,r,p)
    def test_foreign_malformed_and_matching_run_ignored(self):
        for case in ['foreign-malformed','foreign-matching']:self.assertTrue(self.check_case(case)['pass'],case)
    def test_different_approval_target_ignored(self):self.assertTrue(self.check_case('wrong-target')['pass'])
    def test_missing_approval_target_ignored(self):self.assertTrue(self.check_case('missing-target')['pass'])
    def test_authorized_target_inconsistent_or_early_fails_closed(self):
        for case in ['authorized-bad','authorized-early']:self.assertFalse(self.check_case(case)['pass'],case)
    def test_current_valid_targeted_revoke_removes_authority(self):self.assertFalse(self.check_case('valid')['pass'])


class StoragePositiveTests(unittest.TestCase):
    def test_exact_registered_boundaries_and_separate_companion(self):
        with tempfile.TemporaryDirectory() as t:
            e,p,x=storage_fixture(Path(t));r=audit.audit_storage(e,'screen',7,p,x);self.assertTrue(r['pass'],r)
            self.assertFalse(audit.audit_storage(e,'screen',7,p,None)['pass'])

    def test_missing_or_incomplete_companion_never_passes(self):
        for value in ['incomplete','valid']:
            with tempfile.TemporaryDirectory() as t:
                e,p,x=storage_fixture(Path(t));f=x['custodyRoot']/'export-validation.json';d=json.loads(f.read_text());d['status']=value;d['error']='synthetic error';dump(f,d)
                self.assertFalse(audit.audit_storage(e,'screen',7,p,x)['pass'])
        with tempfile.TemporaryDirectory() as t:
            e,p,x=storage_fixture(Path(t));(p/'custody-artifact.json').unlink();self.assertFalse(audit.audit_storage(e,'screen',7,p,x)['pass'])

    def test_byte_overlimit_stale_preflight_and_original_decision_rewrite(self):
        for which in ['worker','output','zip','stale','decision']:
            with tempfile.TemporaryDirectory() as t:
                e,p,x=storage_fixture(Path(t))
                if which=='worker':
                    f=e/'storage-admissions/before-brace-s01.json';d=json.loads(f.read_text());d['freeBytes']=8*1024**3-1;dump(f,d)
                elif which=='output':
                    f=e/'storage-monitor.json';d=json.loads(f.read_text());d['maximumStageOutputBytes']=4*1024**3+1;dump(f,d)
                elif which=='zip':
                    f=p/'evidence-artifact.json';d=json.loads(f.read_text());d['size_in_bytes']=2*1024**3+1;dump(f,d)
                elif which=='stale':
                    f=e/'dispatch-assignment.json';d=json.loads(f.read_text());d['assignmentToolRecord']['storagePreflight']['observedAtUTC']='2026-11-03T22:00:00Z';dump(f,d)
                else:dump(e/'decision.json',{'rewritten':True})
                self.assertFalse(audit.audit_storage(e,'screen',7,p,x)['pass'],which)



class CustodyFailureReportingTests(unittest.TestCase):
    def test_cli_preserves_custody_error_without_battle_result(self):
        from unittest import mock
        import contextlib,io,sys
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);z=p/'original.zip';output=p/'audit.json'
            with zipfile.ZipFile(z,'w') as f:f.writestr('raw',b'original')
            with mock.patch.object(sys,'argv',['audit.py','verify-archive','--zip',str(z),'--sha256','0'*64,'--output',str(output)]),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(),1)
            result=json.loads(output.read_text());self.assertEqual(result['decision'],'AUDIT-CUSTODY-INCOMPLETE');self.assertIn('never opens replacement',result['limit'])

    def test_unknown_strict_approval_key_fails_even_with_consistent_hash_chain(self):
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t),{'unknownField':'not authorized'});result=audit.audit_authority(e,s,'screen',h,r,p)
            self.assertFalse(result['pass']);self.assertIn('accepted authority exact original approval payload',result['checked']);self.assertIn('new issue114 strict20-key typed approval',result['failures'])



def image_log(version='20261026.180.1', name='ubuntu-26.04'):
    lines=['##[group]Runner Image Provisioner','Version: 99999999.999','##[endgroup]',
           '##[group]Operating System','Ubuntu','26.04.1','LTS','##[endgroup]',
           '##[group]Runner Image','Image: '+name,'Version: '+version,
           'Included Software: https://github.com/actions/runner-images/blob/example/Readme.md',
           'Image Release: https://github.com/actions/runner-images/releases/tag/example','##[endgroup]',
           'Prepare workflow directory']
    return ''.join('registered-stage\tSet up job\t2026-11-04T00:01:00.000Z '+s+'\n' for s in lines)


class AuthoritativeImageTests(unittest.TestCase):
    def fixture(self,root,header='20261026.180.1',env='20261026.180.1'):
        e=root/'e';p=root/'p';e.mkdir();p.mkdir();h='c'*40;r=7
        dump(e/'profile.json',dict(identity=dict(image=env)));(p/'job.log').write_text(image_log(header))
        dump(p/'job-log-api.json',dict(schemaVersion=1,jobId=19,runId=r,headSHA=h,logSHA256=audit.digest(p/'job.log'),sourceURL='https://api.github.com/repos/stephanelee/robocode/actions/jobs/19/logs',retrievedAtUTC='2026-11-04T00:10:00Z',collectorTask='/synthetic',toolEvidence={'scope':'PURE SYNTHETIC NOT RUNTIME PROOF'}))
        dump(p/'jobs.json',{'jobs':[{'id':19}]});dump(p/'run.json',dict(created_at='2026-11-04T00:00:00Z'))
        return e,p,h,r

    def test_agreeing_allowed_header_fixture_only(self):
        with tempfile.TemporaryDirectory() as t:
            e,p,h,r=self.fixture(Path(t));result=audit.audit_legacy_runner_image(e,h,r,p);self.assertTrue(result['pass'],result)
            self.assertFalse(audit.audit_legacy_runner_image(e,h,r,None)['pass'])

    def test_actual_dispute_shape_and_unapproved_agreement_fail_closed(self):
        for env in ['20261026.180.1','20261025.179.1']:
            with tempfile.TemporaryDirectory() as t:
                e,p,h,r=self.fixture(Path(t),'20261025.179.1',env);result=audit.audit_legacy_runner_image(e,h,r,p)
                self.assertFalse(result['pass']);self.assertIn('authoritative Runner Image version is prospectively allowed',result['failures'])
                if env!='20261025.179.1':self.assertIn('NO conflict between ENV/profile and independent authoritative Runner Image header',result['failures'])

    def test_missing_duplicate_and_later_env_cannot_rescue_header(self):
        with self.assertRaises(ValueError):audit.runner_image_header(['##[group]Runner Image Provisioner','Version: 20261026.180.1','##[endgroup]'])
        duplicated=image_log().replace('Prepare workflow directory','##[group]Runner Image\nImage: ubuntu-26.04\nVersion: 20261026.180.1\nIncluded Software: u\nImage Release: u\n##[endgroup]')
        with self.assertRaises(ValueError):audit.runner_image_header(duplicated.splitlines())
        # Later shell/ENV output is not the original Setup-job header and cannot overwrite it.
        original=image_log('20261025.179.1')+'registered-stage\tLater shell step\t2026-11-04T00:02:00Z ##[group]Runner Image\n'
        self.assertEqual(audit.runner_image_header(original.splitlines())['version'],'20261025.179.1')

    def test_collector_digest_and_job_identity_tamper_fail(self):
        with tempfile.TemporaryDirectory() as t:
            e,p,h,r=self.fixture(Path(t));(p/'job.log').write_text(image_log('20261101.186.1'));self.assertFalse(audit.audit_legacy_runner_image(e,h,r,p)['pass'])
        with tempfile.TemporaryDirectory() as t:
            e,p,h,r=self.fixture(Path(t));d=json.loads((p/'job-log-api.json').read_text());d['jobId']=20;dump(p/'job-log-api.json',d);self.assertFalse(audit.audit_legacy_runner_image(e,h,r,p)['pass'])


class ActualChildTimingTests(unittest.TestCase):
    def test_in_between_work_does_not_redefine_child_limit(self):
        with tempfile.TemporaryDirectory() as t:
            e,s,p,h,r=authority_fixture(Path(t));j=json.loads((p/'jobs.json').read_text());j['jobs'][0]['completed_at']='2026-11-04T00:50:00Z';j['jobs'][0]['steps'][0]['completed_at']='2026-11-04T00:49:00Z';dump(p/'jobs.json',j)
            # Child itself ended at00:08:30; later reduction/export is not child runtime.
            result=audit.audit_authority(e,s,'screen',h,r,p);self.assertTrue(result['pass'],result)
    def test_actual_child_overrun_or_failed_teardown_cannot_pass(self):
        for change in [{'elapsedMonotonicSeconds':1800.01},{'hardDeadlineRespected':False},{'status':'failed'},{'hardLimitSeconds':1801},{'runStopReserveSeconds':15}]:
            with tempfile.TemporaryDirectory() as t:
                e,s,p,h,r=authority_fixture(Path(t));f=e/'games/brace-s01/teardown.json';d=json.loads(f.read_text());d.update(change);dump(f,d)
                self.assertFalse(audit.audit_authority(e,s,'screen',h,r,p)['pass'])

class CanonicalDecisionTests(unittest.TestCase):
    def stage(self,root,stage='screen'):
        import copy
        rows=GateBoundaryTests().groups(stage);population=[]
        for n,row in enumerate(rows):
            game=dict(id=f'synthetic-{n}',stage=stage,bot=row['game']['bot'],opponent=row['game']['opponent'],rounds=row['game']['rounds'],seed=n,port=33000+n);row['game']=game;population.append(game)
            row.update(valid=True,share=row['ownScore']/row['totalScore'],skips=0,maxMs=1,turns=100,recordingSHA256='a'*64,hitRate=.5,enemyHitRate=.25,firingPer1000Turns=10.0,powerMix={'1':1},enemyPowerMix={'1':1})
            row.update({k:1 for k in audit.COUNTS});row['perRound']=[dict(round=r,turns=10,ownScore=5,totalScore=10,share=.5,win=1,maxMs=1,exposure=row['exposure'],**{k:1 for k in audit.COUNTS}) for r in range(1,game['rounds']+1)]
        raw=audit.stage_gates(stage,rows);self.assertTrue(raw['allPass']);games=[]
        for row in rows:
            g=dict(row['game'],status='valid',**{k:copy.deepcopy(v) for k,v in row.items() if k not in ('game','valid','timingPass')});g.update(observedPhysicalRoundIDs=list(range(1,g['rounds']+1)),completedPhysicalRoundIDs=list(range(1,g['rounds']+1)),observedPhysicalRounds=g['rounds'],completedPhysicalRounds=g['rounds'],partialRecordingError=None);games.append(g)
        decision=dict(stage=stage,headSHA='a'*40,games=games,unplayedIDs=[],decision='progress; confirmation open' if stage=='screen' else 'confirmed',observedPhysicalRounds=sum(g['rounds'] for g in population),completedPhysicalRounds=sum(g['rounds'] for g in population),checks=audit.canonical_gate_checks(stage,rows,raw),diamondMean=raw['groups']['candidate/diamond']['mean']['decimal'],diamondDelta=raw['diamondDifference']['decimal'],retconMean=raw['groups']['candidate/retcon']['mean']['decimal'],retconDelta=raw['retconDifference']['decimal'],candidateRetconEligibleRamScans=raw['candidateRetconEligibleRamCorrectedScans'],betweenGameShareSD={k:v['sampleSDPercentagePoints']/100 if v['sampleSDPercentagePoints'] is not None else None for k,v in raw['groups'].items() if k.rsplit('/',1)[1] in ('diamond','retcon')})
        dump(root/'decision.json',decision);return population,rows,raw,decision
    def test_positive_complete_original_screen_and_confirmation(self):
        for stage in ('screen','confirmation'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p,r,g,d=self.stage(root,stage);result=audit.reconcile_canonical(root,stage,'a'*40,p,r,g,True);self.assertTrue(result['pass'],result)
    def test_complete_raw_scores_never_waive_last_child_failure(self):
        for failure in ('invalid','gamefailure','phasefailure','integrity','infrastructure'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p,r,g,d=self.stage(root)
                if failure=='invalid':d['decision']='invalid';d['games'][-1]['status']='invalid';d['games'][-1]['error']='child exited nonzero after full recording'
                elif failure=='gamefailure':dump(root/'games'/p[-1]['id']/'failure.json',dict(status='invalid',error='last child nonzero'))
                elif failure=='phasefailure':dump(root/'failure-run.json',dict(error='acquisition failure'))
                elif failure=='integrity':d['integrityFailure']='child failed'
                else:d['infrastructureError']='worker failed'
                dump(root/'decision.json',d);self.assertTrue(g['allPass']);self.assertFalse(audit.reconcile_canonical(root,'screen','a'*40,p,r,g,True)['pass'],failure)
    def test_full_stage_cannot_pass_after_canonical_last_child_failure(self):
        from unittest import mock
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e=root/'evidence';source=root/'source';p,rows,raw,declared=self.stage(e,'confirmation')
            for game,row,original in zip(p,rows,declared['games']):
                extra=dict(participantName='KeelBrace' if game['bot']=='candidate' else 'Keel',participantVersion='0.1.0',opponentName='Diamond Port' if game['opponent']=='diamond' else 'Retcon',opponentVersion='1.8.28' if game['opponent']=='diamond' else '0.1.0')
                game.update(extra);original.update(extra);d=e/'games'/game['id'];d.mkdir(parents=True);(d/'opaque-mocked.battle.gz').write_bytes(b'pure mock transport; never decoded')
            dump(source/'benchmarks/keel-brace-v1/games.json',{'confirmation':p});dump(e/'decision.json',declared)
            real_digest=audit.digest
            def fake_digest(path):return audit.GAMES_SHA256 if str(path).endswith('games.json') else real_digest(path)
            with mock.patch.object(audit,'digest',side_effect=fake_digest),mock.patch.object(audit,'reduce_game',side_effect=lambda *a:copy.deepcopy(rows[int(Path(str(a[0])).parent.name.split('-')[-1])])),mock.patch.object(audit,'verify_exported_provenance',return_value={'pass':True}),mock.patch.object(audit,'audit_authority',return_value={'pass':True}),mock.patch.object(audit,'audit_storage',return_value={'pass':True}),mock.patch.object(audit,'audit_runner_image',return_value={'pass':True}),mock.patch.object(audit,'audit_upstream_screen',return_value={'pass':True}):
                self.assertEqual(audit.audit_stage(e,source,'confirmation','a'*40,7)['decision'],'PASS')
                declared['decision']='invalid';declared['games'][-1]['status']='invalid';declared['games'][-1]['error']='nonzero exit after final full recording';dump(e/'decision.json',declared);dump(e/'games'/p[-1]['id']/'failure.json',declared['games'][-1])
                result=audit.audit_stage(e,source,'confirmation','a'*40,7);self.assertEqual(result['rawDisposition'],'PASS');self.assertEqual(result['decision'],'INVALID/INCOMPLETE')

    def test_decision_population_budget_and_metric_mutations_fail(self):
        for mutation in ('head','ids','budget','unplayed','gate','score','round','mean','variation','exposure','pending'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p,r,g,d=self.stage(root)
                if mutation=='head':d['headSHA']='b'*40
                elif mutation=='ids':d['games'].reverse()
                elif mutation=='budget':d['completedPhysicalRounds']-=1
                elif mutation=='unplayed':d['unplayedIDs']=[p[-1]['id']]
                elif mutation=='gate':d['checks']['retconDeltaAtLeast1pp']=False
                elif mutation=='score':d['games'][0]['ownScore']+=1
                elif mutation=='round':d['games'][0]['perRound'][0]['win']=0
                elif mutation=='mean':d['diamondMean']+=.01
                elif mutation=='variation':d['betweenGameShareSD']['candidate/diamond']=.1
                elif mutation=='exposure':d['candidateRetconEligibleRamScans']=0
                else:d['decision']='pending'
                dump(root/'decision.json',d);self.assertFalse(audit.reconcile_canonical(root,'screen','a'*40,p,r,g,True)['pass'],mutation)


class QualifiedRuntimePrimitiveTests(unittest.TestCase):
    def native(self,upper=False,claim=None):
        c=audit.QUALIFIED_NATIVE if claim is None else claim
        detail='\n'.join(k+': '+c[v] for k,v in [('Image','image'),('Version','version'),('Included Software','includedSoftwareURL'),('Image Release','imageReleaseURL')])
        return json.dumps([{('Group' if upper else 'group'):'Runner Image',('Detail' if upper else 'detail'):detail}]).encode()
    def test_separate_qualified_native_and_env_observations(self):
        self.assertEqual(audit.strict_native_claim(self.native(),'imagedata'),audit.QUALIFIED_NATIVE)
        self.assertEqual(audit.strict_native_claim(self.native(True),'setup_info'),audit.QUALIFIED_NATIVE)
        self.assertEqual(audit.strict_environment_claim(b'ImageOS=ubuntu26\nImageVersion="20261026.180.1"\nPATH=/unrelated\n'),audit.QUALIFIED_ENV)
        self.assertNotEqual(audit.QUALIFIED_NATIVE['version'],audit.QUALIFIED_ENV['ImageVersion'])
    def test_native_duplicates_aliases_extra_fields_and_unknown_tuple_fail(self):
        data=json.loads(self.native())
        bad=[data*2,[dict(data[0],Group='Runner Image')],[{'group':'Runner Image','detail':data[0]['detail']+'\nVersion: 20261025.179.1'}],[{'group':'Runner Image','detail':data[0]['detail']+'\nExtra: x'}]]
        for value in bad:
            with self.assertRaises(ValueError):audit.strict_native_claim(json.dumps(value).encode(),'imagedata')
        for key in audit.QUALIFIED_NATIVE:
            c=dict(audit.QUALIFIED_NATIVE);c[key]+='changed'
            with self.assertRaises(ValueError):audit.strict_native_claim(self.native(claim=c),'imagedata')
    def test_environment_missing_duplicate_expansion_and_nonliteral_fail(self):
        for text in ['ImageOS=ubuntu26','ImageOS=ubuntu26\nImageVersion=20261026.180.1\nImageOS=ubuntu26','ImageOS=ubuntu26\nImageVersion=$REV','export ImageOS=ubuntu26\nImageVersion=20261026.180.1','ImageOS=ubuntu26\nImageVersion=20261025.179.1']:
            with self.assertRaises(ValueError):audit.strict_environment_claim(text.encode())
    def test_startup_scope_ignores_later_echoes_and_requires_exact_four_claims(self):
        detail=json.loads(self.native())[0]['detail'];log="Current runner version: '2.338.2'\n"+'##[group]Operating System\nUbuntu\n26.04.1\nLTS\n##[endgroup]\n##[group]Runner Image\n'+detail+'\n##[endgroup]\nComplete job name: Synthetic pure metadata job\n';self.assertEqual(audit.qualified_provider_header(log.encode()),audit.QUALIFIED_NATIVE)
        later=log+'##[group]Runner Image\nImage: malicious later echo\n##[endgroup]\n';self.assertEqual(audit.qualified_provider_header(later.encode()),audit.QUALIFIED_NATIVE)
        for bad in ['\n'.join(log.split('\n')[1:]),log.replace('Complete job name: Synthetic pure metadata job',''),log.replace('Image Release: '+audit.QUALIFIED_NATIVE['imageReleaseURL'],'Image Release: normalized-but-different'),log.replace('Complete job name: Synthetic pure metadata job','##[group]Runner Image\nImage: spoof\n##[endgroup]\nComplete job name: Synthetic pure metadata job')]:
            with self.assertRaises(ValueError):audit.qualified_provider_header(bad.encode())
    def test_metadata_caps_and_original_raw_format_required(self):
        with self.assertRaises(ValueError):audit.strict_native_claim(b' '*1048577,'imagedata')
        with self.assertRaises(ValueError):audit.strict_environment_claim(b' '*1048577)
        with self.assertRaises(ValueError):audit.qualified_provider_header(('later-step\tLater output\t'+image_log('20261025.179.1')).encode())


def qualified_image_fixture(root,early_status=404):
    import copy
    e=root/'evidence';p=root/'provider';pro=e/'provenance';image=pro/'image-identity';head='c'*40;run=7;job=83;workspace='/synthetic/workspace';boot='synthetic-worker-boot';collector=b'PURE SYNTHETIC COLLECTOR BYTES; NOT EXECUTED';sha=lambda b:hashlib.sha256(b).hexdigest();at=lambda minute,second=0:f'2026-11-05T00:{minute:02d}:{second:02d}+00:00'
    freeze={'assets':{'tools/keel_brace_image_probe.py':sha(collector)}}
    source_root=Path(__file__).resolve().parents[3]
    for rel,wanted in [('docs/improvements/keel-brace-runtime-identity-amendment.md',audit.AMENDMENT_DOCUMENT_SHA256),('benchmarks/keel-brace-v1/runtime-identity-amendment.json',audit.AMENDMENT_POLICY_SHA256)]:
        data=(source_root/rel).read_bytes();assert sha(data)==wanted;target=pro/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);freeze['assets'][rel]=wanted
    game=dict(id='synthetic-only',rounds=1);dump(pro/'source/benchmarks/keel-brace-v1/games.json',dict(screen=[game]));dump(pro/'freeze.json',freeze)
    native_detail='\n'.join(k+': '+audit.QUALIFIED_NATIVE[v] for k,v in [('Image','image'),('Version','version'),('Included Software','includedSoftwareURL'),('Image Release','imageReleaseURL')])
    raws=dict(imagedata=json.dumps([dict(group='Runner Image',detail=native_detail)]).encode(),setup_info_0=json.dumps([dict(Group='Runner Image',Detail=native_detail)]).encode(),etc_environment=b'ImageOS=ubuntu26\nImageVersion=20261026.180.1\n',etc_os_release=b'ID=ubuntu\nVERSION_ID="26.04"\n',usr_lib_os_release=b'ID=ubuntu\nVERSION_ID="26.04"\n',etc_lsb_release=b'DISTRIB_ID=Ubuntu\n',boot_id=(boot+'\n').encode())
    runner='/observed/runner';exe=runner+'/bin/Runner.Worker';comm='Runner.Worker\n';paths=dict(audit.IMAGE_LABEL_PATHS,setup_info_0=runner+'/.setup_info')
    source_identity=dict(stage='screen',runId=run,runAttempt=1,headSHA=head,workspace=workspace,logicalJobName='registered-stage')
    claims=dict(imagedataNative=dict(audit.QUALIFIED_NATIVE),setupInfoNative=dict(audit.QUALIFIED_NATIVE),processEnvironment=dict(audit.QUALIFIED_ENV),etcEnvironment=dict(audit.QUALIFIED_ENV))
    def snapshot(target,phase,minute,collector_phase):
        c=target/'collector'/collector_phase;c.mkdir(parents=True,exist_ok=True);files={};captures={};stable_raw={}
        def retain(rel,data,raw=False):
            f=c/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data);row=dict(artifactPath=rel,bytes=len(data),suppliedBytes=len(data),complete=True,sha256=sha(data))
            if raw:row['sourceBytesComplete']=True
            files[rel]=row;return copy.deepcopy(row)
        collector_ref=retain('collector.py',collector)
        for label,data in raws.items():
            stat=dict(size=0 if label=='boot_id' else len(data),mode=33188,uid=1001,gid=1001,device=1,inode=3,mtimeNs=4,regular=True)
            raw=retain('raw/'+label+'.bin',data,True);record=dict(requestedPath=paths[label],resolvedPath=paths[label],exists=True,errors=[],bytesComplete=True,raw=raw,rawSHA256Scope='whole-observed-file',utcStart=at(minute),utcEnd=at(minute,1),durationNs=1,lstatBefore=stat,statBefore=stat,openedStat=stat,statAfter=stat)
            if label in ('imagedata','setup_info_0','etc_environment'):
                parsed=claims['imagedataNative' if label=='imagedata' else 'setupInfoNative'] if label!='etc_environment' else {k:dict(present=True,value=v) for k,v in audit.QUALIFIED_ENV.items()}
                record.update(parsedArtifact=retain('parsed/'+label+'.json',json.dumps({'claims':parsed}).encode()),parsedClaims=copy.deepcopy(parsed),parseErrors=[])
            captures[label]=record;stable_raw[label]=dict(sourcePath=paths[label],resolvedPath=paths[label],size=stat['size'],wholeSHA256=sha(data),mode=stat['mode'],uid=1001,gid=1001)
        discovery=dict(procRoot='/proc',scannedEntries=1,nonmatchingCount=0,truncated=False,errors=[],roots=[runner],matches=[dict(pid=42,comm='Runner.Worker',commRawASCII=comm,commSHA256=sha(comm.encode()),commPath='/proc/42/comm',commStat={},exeSymlinkStat={},exeReadlink=exe,exePath=exe,exeReadlinkSHA256=sha(exe.encode()),runnerRoot=runner,setupInfoPath=runner+'/.setup_info',errors=[])])
        report=dict(schemaVersion=1,phase=collector_phase,errors=[],collectorPython=dict(executable='/usr/bin/python3'),durationNs=10**9,utcStart=at(minute),utcEnd=at(minute,1),limits=dict(eachFileBytes=1048576,wholeTwoPhaseArtifactBytes=10485760,procEntries=512,runnerMatches=16,collectionSeconds=120),collectorSource=collector_ref,runnerDiscovery=discovery,captures=captures,processImageEnvironment={k:dict(present=True,value=v) for k,v in audit.QUALIFIED_ENV.items()},platform=dict(architecture='x86_64',kernel='synthetic-kernel',kernelVersion='synthetic-kernel-version'),context=dict(GITHUB_RUN_ID=str(run),GITHUB_RUN_ATTEMPT='1',GITHUB_JOB='registered-stage',GITHUB_REPOSITORY='stephanelee/robocode'),rawArtifactIndex=copy.deepcopy(files))
        report_ref=retain('report.json',json.dumps(report,indent=2).encode());status=dict(schemaVersion=1,phase=collector_phase,diagnosticErrorCount=0,files=files,report=report_ref,retainedBytesBeforeStatus=sum(v['bytes'] for v in files.values()));dump(c/'capture-status.json',status)
        stable=dict(rawSources=stable_raw,claims=copy.deepcopy(claims),runnerRoot=runner,setupInfoPath=runner+'/.setup_info',bootId=boot,architecture='x86_64',kernel='synthetic-kernel',kernelVersion='synthetic-kernel-version',collectorSourceSHA256=sha(collector),sourceIdentity=dict(source_identity))
        snap=dict(schemaVersion=1,policyID=audit.QUALIFIED_POLICY_ID,policySHA256=audit.AMENDMENT_POLICY_SHA256,**source_identity,phase=phase,bootId=boot,architecture='x86_64',kernel='synthetic-kernel',kernelVersion='synthetic-kernel-version',collectorSourceSHA256=sha(collector),snapshotManifestSHA256=audit.digest(c/'capture-status.json'),reportSHA256=audit.digest(c/'report.json'),stable=stable);dump(target/'snapshot.json',snap);return snap
    b=snapshot(image/'before-tools','before-tools',2,'before-tools');prepared=snapshot(image/'prepared','prepared',4,'after-tools')
    provider_job=dict(id=job,run_id=run,status='in_progress',name='registered-stage');dump(image/'prepared-jobs-api.json',dict(jobs=[provider_job]))
    hashes=dict(beforeToolsSnapshotSHA256=audit.digest(image/'before-tools/snapshot.json'),preparedSnapshotSHA256=audit.digest(image/'prepared/snapshot.json'))
    wrapper=dict(schemaVersion=1,policyID=audit.QUALIFIED_POLICY_ID,policySHA256=audit.AMENDMENT_POLICY_SHA256,runId=run,runAttempt=1,headSHA=head,logicalJobName='registered-stage',preparedNumericJobId=job,beforeToolsSnapshotFile='before-tools/snapshot.json',preparedSnapshotFile='prepared/snapshot.json',jobsAPIFile='prepared-jobs-api.json',jobsAPISHA256=audit.digest(image/'prepared-jobs-api.json'),**hashes);dump(image/'prepared-wrapper.json',wrapper);hashes['preparedWrapperSHA256']=audit.digest(image/'prepared-wrapper.json')
    profile=dict(identity=dict(workspace=workspace,bootId=boot,image=audit.QUALIFIED_ENV['ImageVersion']),imageIdentity=dict(policyID=audit.QUALIFIED_POLICY_ID,policySHA256=audit.AMENDMENT_POLICY_SHA256,**hashes,preparedNumericJobId=job,stable=prepared['stable']));dump(pro/'profile.json',profile);dump(e/'approval.json',dict(stage='screen'))
    artifact=dict(id=99,digest='sha256:'+'a'*64,updated_at=at(6),workflow_run=dict(id=run,head_sha=head));dump(e/'artifact.json',artifact)
    log="Current runner version: '2.338.2'\n"+'##[group]Operating System\nUbuntu\n26.04.1\nLTS\n##[endgroup]\n##[group]Runner Image\n'+native_detail+'\n##[endgroup]\nComplete job name: Synthetic pure metadata job\n'
    response=b'{"message":"Not Found"}' if early_status==404 else log.encode();(p/'early').mkdir(parents=True);(p/'early/early-header-response.bin').write_bytes(response)
    early=dict(status='unavailable-404-deferred' if early_status==404 else 'available-matched',endpoint=f'https://api.github.com/repos/stephanelee/robocode/actions/jobs/{job}/logs',runId=run,runAttempt=1,jobId=job,headSHA=head,observedAtUTC=at(7),httpStatus=early_status,responseBytes=len(response),responseSHA256=sha(response),responseFile='early-header-response.bin',scope='actual-job-log-request' if early_status==404 else 'complete-startup-header',runAccessVerified=True,jobsAccessVerified=True,artifactAccessVerified=True,jobInProgress=True,nativeClaim=None if early_status==404 else dict(audit.QUALIFIED_NATIVE))
    review=dict(schemaVersion=1,policyID=audit.QUALIFIED_POLICY_ID,policySHA256=audit.AMENDMENT_POLICY_SHA256,**hashes,primaryPregameAuthority='native-only-qualified-observation',earlyHeaderEvidence=early,finalExternalReconciliationRequired=True)
    comment=dict(user=dict(login='stephanelee',id=30698997),created_at=at(8),body='KEEL_BRACE_RUNTIME_REVIEW_V1\n'+json.dumps(dict(agentToolProvenance=dict(imageIdentityReview=review))));dump(e/'review-report-comment.json',comment)
    early_run=dict(id=run,head_sha=head,run_attempt=1,status='in_progress');dump(p/'early/run.json',early_run);dump(p/'early/jobs.json',dict(jobs=[provider_job]));dump(p/'early/artifact.json',artifact)
    for phase,minute in [('before-receipt',9),('before-synthetic-only',10)]:
        rel=f'image-identity-revalidations/revalidation-{minute}/snapshot.json';snapshot((e/rel).parent,f'revalidation-{minute}',minute,'after-tools');dump(e/'approval-revalidations'/(phase+'.json'),dict(checkedAtUTC=at(minute,2),imageIdentityRevalidation=dict(snapshotFile=rel,snapshotSHA256=audit.digest(e/rel),preparedSnapshotSHA256=hashes['preparedSnapshotSHA256'],runtimeProfileSHA256=audit.digest(pro/'profile.json'))))
    steps=[]
    for name,start,end in [('Capture original qualified image observations before tools',2,3),('Prepare exact sources/runtime/commands on this worker (zero games)',4,5),('Finalize immutable zero-game provenance before independent review',5,6)]:steps.append(dict(name=name,status='completed',conclusion='success',started_at=at(start),completed_at=at(end)))
    final_job=dict(provider_job,status='completed',conclusion='success',completed_at=at(11),steps=steps);dump(p/'jobs.json',dict(jobs=[final_job]));dump(p/'run.json',dict(early_run,status='completed',conclusion='success'))
    (p/'job.log').write_bytes(log.encode());dump(p/'job-log-api.json',dict(schemaVersion=1,jobId=job,runId=run,headSHA=head,logSHA256=sha(log.encode()),sourceURL=early['endpoint'],retrievedAtUTC=at(12),collectorTask='/synthetic-independent-collector',toolEvidence=dict(scope='PURE FIXTURE, NOT RUNTIME CLEARANCE')))
    return e,p,head,run


class QualifiedRuntimeIntegrationTests(unittest.TestCase):
    def test_provider_step_recorded_second_boundary(self):
        step=dict(started_at='2026-11-05T00:02:00Z',completed_at='2026-11-05T00:02:01Z')
        self.assertTrue(audit.provider_step_contains(step,'2026-11-05T00:02:00.5Z','2026-11-05T00:02:01.999999Z'))
        self.assertFalse(audit.provider_step_contains(step,'2026-11-05T00:02:00Z','2026-11-05T00:02:02Z'))
        self.assertFalse(audit.provider_step_contains(step,'2026-11-05T00:01:59.999999Z','2026-11-05T00:02:01Z'))

    def test_complete_qualified_tuple_404_and_available_header(self):
        for status in (404,200):
            with tempfile.TemporaryDirectory() as tmp:
                e,p,h,r=qualified_image_fixture(Path(tmp),status);result=audit.audit_runner_image(e,h,r,p);self.assertTrue(result['pass'],result);self.assertEqual(result['status'],'PASS')
    def test_missing_final_provider_pending_then_completed_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,p,h,r=qualified_image_fixture(Path(tmp));result=audit.audit_runner_image(e,h,r,None);self.assertFalse(result['pass']);self.assertEqual(result['status'],'PROVENANCE-PENDING')
            (p/'job.log').unlink();result=audit.audit_runner_image(e,h,r,p);self.assertFalse(result['pass']);self.assertEqual(result['status'],'PROVENANCE-INCOMPLETE')
    def test_early_actual_bytes_api_status_and_report_deferral_mutations(self):
        for case in ('response','earlyjob','earlyartifact','http403','declaration','endpoint'):
            with tempfile.TemporaryDirectory() as tmp:
                e,p,h,r=qualified_image_fixture(Path(tmp));f=e/'review-report-comment.json';comment=json.loads(f.read_text());payload=json.loads(comment['body'].split('\n',1)[1]);early=payload['agentToolProvenance']['imageIdentityReview']['earlyHeaderEvidence']
                if case=='response':(p/'early/early-header-response.bin').write_bytes(b'changed')
                elif case=='earlyjob':dump(p/'early/jobs.json',{'jobs':[]})
                elif case=='earlyartifact':dump(p/'early/artifact.json',{'id':1})
                elif case=='http403':early['httpStatus']=403
                elif case=='declaration':early['artifactAccessVerified']=False
                else:early['endpoint']='https://api.github.com/repos/stephanelee/robocode/actions/jobs/1/logs'
                comment['body']='KEEL_BRACE_RUNTIME_REVIEW_V1\n'+json.dumps(payload);dump(f,comment);self.assertFalse(audit.audit_runner_image(e,h,r,p)['pass'],case)
    def test_snapshot_boot_source_hash_graph_and_live_binding_mutations(self):
        for case in ('snapshot','raw','collector','live','profile','policy'):
            with tempfile.TemporaryDirectory() as tmp:
                e,p,h,r=qualified_image_fixture(Path(tmp));base=e/'provenance/image-identity'
                if case=='snapshot':f=base/'before-tools/snapshot.json';d=json.loads(f.read_text());d['bootId']='different';dump(f,d)
                elif case=='raw':(base/'prepared/collector/after-tools/raw/imagedata.bin').write_bytes(b'changed')
                elif case=='collector':(base/'prepared/collector/after-tools/collector.py').write_bytes(b'different source')
                elif case=='live':f=e/'approval-revalidations/before-synthetic-only.json';d=json.loads(f.read_text());d['imageIdentityRevalidation']['runtimeProfileSHA256']='0'*64;dump(f,d)
                elif case=='profile':f=e/'provenance/profile.json';d=json.loads(f.read_text());d['imageIdentity']['stable']['claims']['processEnvironment']['ImageVersion']='fake';dump(f,d)
                else:(e/'provenance/source/benchmarks/keel-brace-v1/runtime-identity-amendment.json').write_bytes(b'changed')
                self.assertFalse(audit.audit_runner_image(e,h,r,p)['pass'],case)
    def rebind_snapshot_report(self,root,report,snapshot):
        c=root/'collector/before-tools';dump(c/'report.json',report);status=json.loads((c/'capture-status.json').read_text());row=status['files']['report.json'];row.update(bytes=(c/'report.json').stat().st_size,suppliedBytes=(c/'report.json').stat().st_size,sha256=audit.digest(c/'report.json'));status['report']=row;status['retainedBytesBeforeStatus']=sum(v['bytes'] for v in status['files'].values());dump(c/'capture-status.json',status);snapshot.update(reportSHA256=audit.digest(c/'report.json'),snapshotManifestSHA256=audit.digest(c/'capture-status.json'));dump(root/'snapshot.json',snapshot)
    def test_rehashed_discovery_stat_cap_and_complete_flags_still_rejected(self):
        for case in ('discoverycap','discoveryerror','ambiguousroot','truncated','statmutation','prefix','readdeadline','reporterror','processenv','booleanstat','nonUTC'):
            with tempfile.TemporaryDirectory() as tmp:
                e,p,h,r=qualified_image_fixture(Path(tmp));base=e/'provenance/image-identity/before-tools';report=json.loads((base/'collector/before-tools/report.json').read_text());snapshot=json.loads((base/'snapshot.json').read_text())
                if case=='discoverycap':report['runnerDiscovery']['scannedEntries']=513
                elif case=='discoveryerror':report['runnerDiscovery']['errors']=['failed read']
                elif case=='ambiguousroot':report['runnerDiscovery']['roots'].append('/another/root')
                elif case=='truncated':report['runnerDiscovery']['truncated']=True
                elif case=='statmutation':report['captures']['setup_info_0']['statAfter']['mtimeNs']+=1
                elif case=='prefix':report['captures']['imagedata']['bytesComplete']=False
                elif case=='readdeadline':report['durationNs']=120*10**9+1
                elif case=='reporterror':report['errors']=['diagnostic failed']
                elif case=='booleanstat':report['captures']['setup_info_0']['openedStat']['uid']=True
                elif case=='nonUTC':report['utcStart']='2026-11-05T01:02:00+01:00'
                else:report['processImageEnvironment']['ImageVersion']['present']=False
                self.rebind_snapshot_report(base,report,snapshot);freeze=json.loads((e/'provenance/freeze.json').read_text());expected={k:snapshot[k] for k in audit.IMAGE_SOURCE_KEYS}
                with self.assertRaises(ValueError,msg=case):audit.audit_image_snapshot(base,expected,'before-tools',freeze)
    def test_exact_optional_lsb_absence_and_other_errors_rejected(self):
        for error in ('FileNotFoundError','PermissionError'):
            with tempfile.TemporaryDirectory() as tmp:
                e,p,h,r=qualified_image_fixture(Path(tmp));base=e/'provenance/image-identity/before-tools';c=base/'collector/before-tools';report=json.loads((c/'report.json').read_text());snapshot=json.loads((base/'snapshot.json').read_text());label='etc_lsb_release';record=report['captures'][label];record.update(exists=False,raw=None,bytesComplete=False,resolvedPath=None,errors=[dict(operation='lstat',path='/etc/lsb-release',type=error,message='synthetic missing source',messageWasTruncated=False)]);report['rawArtifactIndex'].pop('raw/'+label+'.bin');status=json.loads((c/'capture-status.json').read_text());status['files'].pop('raw/'+label+'.bin');status['diagnosticErrorCount']=1;dump(c/'capture-status.json',status);snapshot['stable']['rawSources'][label]=dict(sourcePath='/etc/lsb-release',absent=True);self.rebind_snapshot_report(base,report,snapshot);freeze=json.loads((e/'provenance/freeze.json').read_text());expected={k:snapshot[k] for k in audit.IMAGE_SOURCE_KEYS}
                if error=='FileNotFoundError':audit.audit_image_snapshot(base,expected,'before-tools',freeze)
                else:
                    with self.assertRaises(ValueError):audit.audit_image_snapshot(base,expected,'before-tools',freeze)

    def test_final_job_header_source_and_step_order_mutations(self):
        for case in ('header','job','step','capturetime','collector','loghash'):
            with tempfile.TemporaryDirectory() as tmp:
                e,p,h,r=qualified_image_fixture(Path(tmp))
                if case=='header':
                    (p/'job.log').write_text((p/'job.log').read_text().replace('Version: 20261025.179.1','Version: 20261026.180.1'));f=p/'job-log-api.json';d=json.loads(f.read_text());d['logSHA256']=audit.digest(p/'job.log');dump(f,d)
                elif case=='job':f=p/'jobs.json';d=json.loads(f.read_text());d['jobs'][0]['id']=84;dump(f,d)
                elif case in ('step','capturetime'):
                    f=p/'jobs.json';d=json.loads(f.read_text())
                    if case=='step':d['jobs'][0]['steps'].reverse()
                    else:d['jobs'][0]['steps'][0]['completed_at']='2026-11-05T00:01:59+00:00'
                    dump(f,d)
                else:f=p/'job-log-api.json';d=json.loads(f.read_text());d['collectorTask']='' if case=='collector' else d['collectorTask'];d['logSHA256']='0'*64 if case=='loghash' else d['logSHA256'];dump(f,d)
                self.assertFalse(audit.audit_runner_image(e,h,r,p)['pass'],case)


class UpstreamCustodyTests(unittest.TestCase):
    def parts(self,root,data=b'original ZIP range bytes'):
        path=root/'part';path.write_bytes(data);sha=hashlib.sha256(data).hexdigest()
        part=dict(path='part',offset=0,bytes=len(data),sha256=sha,gitBlobSHA1=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest())
        return dict(artifactId=42,bytes=len(data),sha256=sha,parts=[part]),{'part':sha}
    def test_seekable_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original,retained=self.parts(root);r=audit.PartReader(root,original,retained)
            self.assertEqual(r.read(8),b'original');r.seek(-5,2);self.assertEqual(r.read(),b'bytes');r.seek(0);self.assertEqual(r.read(),b'original ZIP range bytes')
    def test_noncontiguous_or_changed_part_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original,retained=self.parts(root);original['parts'][0]['offset']=1
            with self.assertRaises(ValueError):audit.PartReader(root,original,retained)
            original['parts'][0]['offset']=0;(root/'part').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):audit.PartReader(root,original,retained)
    def test_bad_original_digest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original,retained=self.parts(root);original['sha256']='0'*64
            with self.assertRaises(ValueError):audit.PartReader(root,original,retained)
    def test_confirmation_without_screen_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:self.assertFalse(audit.audit_upstream_screen(Path(tmp),'confirmation')['pass'])
        self.assertTrue(audit.audit_upstream_screen(Path('/unused'),'screen')['pass'])

if __name__=='__main__':unittest.main()
