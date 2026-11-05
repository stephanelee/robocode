"""Required pure source/event-order/fraction/compile-proof corruption fixtures.

All full-reader paths below are actual original-ZIP/CLI executions without core
function patches. No engine, bot.start, server, JVM or opponent is launched.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import evidence_fixture as f
import phase_order as phase
from test_full_reader import invoke,edit

a=f.audit


def derive(x):
    path=x['provider']/'phase-order-proof.json'
    if path.exists():path.unlink() # controlled synthetic fixture, never actual evidence
    previous=sys.argv
    try:
        sys.argv=['audit.py','derive-phase-order','--evidence-zip',str(x['zip']),'--zip-sha256',a.digest(x['zip']),'--source-root',str(x['source']),'--provider-evidence',str(x['provider']),'--stage',x['stage'],'--head',x['head'],'--run',str(x['run']),'--output',str(path)]
        with contextlib.redirect_stdout(io.StringIO()):code=a.main()
    finally:sys.argv=previous
    return code,path


class ExactProviderPrecisionTests(unittest.TestCase):
    def test_100ns_distinct_and_no_truncation(self):
        lo='2026-11-05T00:02:00.0000001Z';hi='2026-11-05T00:02:00.0000009Z'
        self.assertEqual(phase.instant(hi)[0]-phase.instant(lo)[0],phase.Fraction(8,10**7))
        self.assertFalse(phase.instant('2026-11-05T00:02:00.0000000Z')[1])
        step=dict(started_at='2026-11-05T00:01:59Z',completed_at=lo)
        with self.assertRaises(ValueError):phase.contains_capture(step,'2026-11-05T00:01:59.9Z',hi,causal=True)
        step['completed_at']=hi
        self.assertEqual(phase.contains_capture(step,'2026-11-05T00:01:59.9Z',lo,causal=False),'direct-nonoverlap')
        with self.assertRaises(ValueError):a.timestamp(lo) # explicit rejection outside exact phase math

    def test_whole_bucket_only_and_known_contradiction_never_rescued(self):
        step=dict(started_at='2026-11-05T00:02:00Z',completed_at='2026-11-05T00:02:00Z')
        with self.assertRaises(ValueError):phase.contains_capture(step,'2026-11-05T00:02:00Z','2026-11-05T00:02:00.5Z',causal=False)
        self.assertEqual(phase.contains_capture(step,'2026-11-05T00:02:00Z','2026-11-05T00:02:00.5Z',causal=True),phase.KIND)
        for end in ['2026-11-05T00:02:01Z','2026-11-05T00:02:01.0000001Z']:
            with self.assertRaises(ValueError):phase.contains_capture(step,'2026-11-05T00:02:00Z',end,causal=True)
        step['completed_at']='2026-11-05T00:02:00.0000000Z'
        with self.assertRaises(ValueError):phase.contains_capture(step,'2026-11-05T00:02:00Z','2026-11-05T00:02:00.0000001Z',causal=True)

    def test_approval_deadline_and_order_do_not_use_truncated_whole_bucket(self):
        self.assertTrue(phase.strictly_before('2026-11-05T00:02:00Z','2026-11-05T00:02:01Z'))
        self.assertFalse(phase.strictly_before('2026-11-05T00:02:00Z','2026-11-05T00:02:00.9Z'))
        deadline=phase.instant('2026-11-05T00:32:00Z')[0]
        self.assertFalse(phase.no_later_than_exact_deadline('2026-11-05T00:32:00Z',deadline))
        self.assertTrue(phase.no_later_than_exact_deadline('2026-11-05T00:31:59Z',deadline))


class FullPhaseProofWireTests(unittest.TestCase):
    def test_real_cli_derivation_and_whole_screen_with_7digit_log(self):
        with tempfile.TemporaryDirectory() as t:
            x=f.full_fixture(Path(t),capture_end_second=.5,causal_overlap=True)
            code,path=derive(x);self.assertEqual(code,0,path.with_name(path.name+'.failure.json').read_text() if code else '')
            proof=a.read_json(path);self.assertEqual(set(proof),phase.PROOF_KEYS)
            self.assertTrue(all(len(v['observedProviderTimestamp'].split('.')[1][:-1])==7 for v in proof['phaseRecords']))
            self.assertNotIn('reviewerTask',proof)
            code,result=invoke(x);self.assertEqual((code,result['decision']),(0,'PASS'),{k:v.get('failures') for k,v in result.items() if isinstance(v,dict) and 'failures' in v})
            self.assertEqual(result['registeredRoundBudget'],120)

    def test_real_cli_fresh_confirmation_and_original_screen_causal_replay(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);screen=f.full_fixture(root/'s',capture_end_second=.5,causal_overlap=True);x=f.full_fixture(root/'c',stage='confirmation',run_id=8,job_id=84,capture_end_second=.5,causal_overlap=True)
            gate=f.attach_upstream_screen(screen,x);f.bind_fixture_upstream_inputs(x,gate)
            code,path=derive(x);self.assertEqual(code,0)
            code,result=invoke(x);self.assertEqual((code,result['decision']),(0,'PASS'),{k:v.get('failures') for k,v in result.items() if isinstance(v,dict) and 'failures' in v})
            self.assertEqual(result['registeredRoundBudget'],300);self.assertTrue(result['upstreamScreen']['pass'])

    def test_available_complete_200_is_verified_not_pending_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            x=f.full_fixture(Path(t),early_status=200);code,result=invoke(x);self.assertEqual((code,result['decision']),(0,'PASS'))
            body=x['provider']/'early/early-header-response.bin';raw=body.read_bytes();body.write_bytes(b'\n'.join(line for line in raw.split(b'\n') if phase.PREFIX.encode() not in line));code,result=invoke(x);self.assertEqual(code,1)

    def test_full_pipeline_required_phase_corruptions(self):
        cases=['missing','duplicate','quoted-echo','wrong-phase','wrong-run','wrong-attempt','wrong-head','wrong-snapshot','late-segment','next-step-missing','next-step-reversed','next-step-failed','command-hash','workflow-background','workflow-parallel','shell-detach','emitter-bytes','raw-collector-role','fractional-100ns','exact-fractional-zero','definitely-late','pending-falsely-observed','final-proof-missing','phase-proof-bool','partial-log']
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                x=f.full_fixture(Path(t),capture_end_second=.5,causal_overlap=True);e=x['evidence'];p=x['provider'];image=e/'provenance/image-identity';log=p/'job.log';raw=log.read_text();lines=raw.splitlines();index=next(i for i,line in enumerate(lines) if phase.PREFIX in line)
                if case=='missing':del lines[index];log.write_text('\n'.join(lines)+'\n')
                elif case=='duplicate':lines.insert(index,lines[index]);log.write_text('\n'.join(lines)+'\n')
                elif case=='quoted-echo':lines[index]=lines[index].replace(phase.PREFIX,'"'+phase.PREFIX,1)+'"';log.write_text('\n'.join(lines)+'\n')
                elif case in ('wrong-phase','wrong-run','wrong-attempt','wrong-head','wrong-snapshot'):
                    prefix,message=lines[index].split(' '+phase.PREFIX,1);payload=json.loads(message)
                    key={'wrong-phase':'phase','wrong-run':'runId','wrong-attempt':'runAttempt','wrong-head':'headSHA','wrong-snapshot':'snapshotSHA256'}[case];payload[key]={'wrong-phase':'prepared','wrong-run':999,'wrong-attempt':2,'wrong-head':'0'*40,'wrong-snapshot':'0'*64}[case]
                    lines[index]=prefix+' '+phase.PREFIX+phase.canonical(payload).decode();log.write_text('\n'.join(lines)+'\n')
                elif case=='late-segment':marker=lines.pop(index);lines.append(marker);log.write_text('\n'.join(lines)+'\n')
                elif case.startswith('next-step-'):
                    def mutation(d):
                        row=d['jobs'][0];target=next(v for v in row['steps'] if v.get('number')==4)
                        if case=='next-step-missing':row['steps'].remove(target)
                        elif case=='next-step-reversed':target['number']=2
                        else:target['conclusion']='failure'
                    edit(p/'jobs.json',mutation)
                elif case=='command-hash':edit(image/'phase-order-plan.json',lambda d:d['phasePlans'][0].update(foregroundCommandSHA256='0'*64))
                elif case in ('workflow-background','workflow-parallel','shell-detach'):
                    path=x['source']/phase.WORKFLOW;text=path.read_text();addition='        background: true\n' if case=='workflow-background' else '        parallel: true\n' if case=='workflow-parallel' else ''
                    text=text.replace('        run: |\n',addition+'        run: |\n',1) if addition else text.replace('--stage "$STAGE"','--stage "$STAGE" &',1);path.write_text(text)
                elif case=='emitter-bytes':(x['source']/phase.EMITTER).write_bytes(b'wrong unreviewed helper')
                elif case=='raw-collector-role':edit(image/'before-tools/snapshot.json',lambda d:d.update(collectorSourceSHA256='0'*64))
                elif case in ('fractional-100ns','exact-fractional-zero','definitely-late'):
                    complete='2026-11-05T00:02:00.4999999Z' if case=='fractional-100ns' else '2026-11-05T00:02:00.0000000Z' if case=='exact-fractional-zero' else '2026-11-05T00:01:59Z'
                    edit(p/'jobs.json',lambda d:next(v for v in d['jobs'][0]['steps'] if v.get('number')==3).update(completed_at=complete))
                elif case=='pending-falsely-observed':
                    path=e/'review-report-comment.json';comment=a.read_json(path);body=json.loads(comment['body'].split('\n',1)[1]);body['agentToolProvenance']['imageIdentityReview']['phaseOrderReview']['providerMarkersObserved']=True;comment['body']='KEEL_WAKE_RUNTIME_REVIEW_V1\n'+json.dumps(body);f.dump(path,comment)
                elif case=='final-proof-missing':(p/'phase-order-proof.json').unlink()
                elif case=='phase-proof-bool':edit(p/'phase-order-proof.json',lambda d:d.update(schemaVersion=True))
                elif case=='partial-log':log.write_text('\n'.join(lines[index:])+'\n')
                if log.exists():edit(p/'job-log-api.json',lambda d:d.update(logSHA256=a.digest(log)))
                f.repack(x);code,result=invoke(x);self.assertEqual(code,1,(case,result.get('decision')));self.assertNotEqual(result.get('decision'),'PASS')

    def test_observer_typecheck_corruptions_reach_whole_reader(self):
        for case in ('missing','compiler','classpath','source','version','proc','duration','classes','output'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                x=f.full_fixture(Path(t));root=x['evidence']/'provenance/launcher-compilation';profile_path=x['evidence']/'provenance/profile.json'
                if case=='missing':(root/'proof.json').unlink()
                elif case=='classes':(root/'classes/KeelWakeBattle.class').write_bytes(b'wrong class')
                elif case=='output':(root/'stderr.bin').write_bytes(b'changed original diagnostic')
                else:
                    proof=a.read_json(root/'proof.json')
                    if case in ('compiler','classpath','source'):proof[case]['sha256']='0'*64
                    elif case=='version':proof['compilerVersion']='javac 17.0.21'
                    elif case=='proc':proof['command'].remove('-proc:none')
                    else:proof['elapsedMonotonicSeconds']=120.000001
                    f.dump(root/'proof.json',proof);profile=a.read_json(profile_path);profile['launcherCompilation']=dict(proof=proof,proofSHA256=a.digest(root/'proof.json'));f.dump(profile_path,profile)
                f.repack(x);code,result=invoke(x);self.assertEqual(code,1);self.assertNotEqual(result.get('decision'),'PASS')


if __name__=='__main__':unittest.main()
