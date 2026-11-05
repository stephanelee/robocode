"""UNMOCKED complete original-ZIP/CLI/source/authority/image/raw-gate fixtures.

No bot/server/engine/JAR/network starts. Synthetic participants/events/API/tool
records are explicitly fixture inputs and never actual authority or strength.
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

a=f.audit


def invoke(x,identity=True):
    output=x['evidence'].parent/'reader-output.json'
    argv=['audit.py','audit-stage','--evidence-zip',str(x['zip']),'--zip-sha256',a.digest(x['zip']),'--custody-zip',str(x['custodyZip']),'--custody-sha256',a.digest(x['custodyZip']),'--source-root',str(x['source']),'--provider-evidence',str(x['provider']),'--stage',x.get('stage','screen'),'--head',x['head'],'--run',str(x['run']),'--output',str(output)]
    if identity:argv+=['--reviewer-task',x['reviewer'],'--reviewer-identity-file',str(x['identity'])]
    previous=sys.argv
    try:
        sys.argv=argv
        with contextlib.redirect_stdout(io.StringIO()):code=a.main()
    finally:sys.argv=previous
    return code,json.loads(output.read_text())


def edit(path,fn):
    d=a.read_json(path);fn(d);f.dump(path,d)


class FullOriginalZIPReaderTests(unittest.TestCase):
    def test_actual_cli_canonical_original_zip_all_validators_unmocked(self):
        with tempfile.TemporaryDirectory() as t:
            x=f.full_fixture(Path(t));code,r=invoke(x)
            self.assertEqual(code,0,r);self.assertEqual(r['decision'],'PASS')
            self.assertEqual(r['registeredRoundBudget'],120);self.assertEqual(len(r['validGames']),14)
            self.assertEqual(r['rawGates']['groups']['candidate/diamond']['mean']['numerator'],3)
            self.assertEqual(r['rawGates']['groups']['candidate/diamond']['mean']['denominator'],5)
            self.assertAlmostEqual(r['rawGates']['diamondDifference']['decimal'],0.05)
            self.assertEqual(r['rawGates']['candidateDiamondDensityChangedWaves'],30)
            for k in ('provenance','authority','storage','runnerImage','upstreamScreen','canonicalReconciliation','actualReviewerIdentity'):self.assertTrue(r[k]['pass'],(k,r[k]))
            self.assertEqual(r['runnerImage']['finalReconciliation']['reviewerTask'],x['reviewer'])
            self.assertEqual(r['runnerImage']['finalReconciliation']['status'],'PASS')
            self.assertTrue(r['originalArchiveAudit']['pass']);self.assertTrue(r['companionArchiveAudit']['pass'])

    def test_equal_compatibility_aliases_still_use_original_canonical_profile(self):
        with tempfile.TemporaryDirectory() as t:
            x=f.full_fixture(Path(t));e=x['evidence']
            for name in ('profile.json','freeze.json'):(e/name).write_bytes((e/'provenance'/name).read_bytes())
            f.repack(x);code,r=invoke(x);self.assertEqual((code,r['decision']),(0,'PASS'),r)

    def test_without_actual_review_proof_machine_cannot_impersonate_reviewer(self):
        with tempfile.TemporaryDirectory() as t:
            x=f.full_fixture(Path(t));code,r=invoke(x,identity=False)
            self.assertEqual(code,1);self.assertEqual(r['decision'],'UNAUDITED');self.assertNotIn('finalReconciliation',r['runnerImage']);self.assertEqual(r['runnerImage']['providerFacts']['status'],'UNAUDITED')

    def test_full_unmocked_pipeline_corruptions_fail_closed(self):
        # Repack transport hashes after each mutation: failures must not be hidden
        # behind an unrelated outer-ZIP hash mismatch.
        cases=['root-profile','missing-canonical','origin','outer-approval','counter-missing','counter-duplicate','counter-fraction','counter-negative','counter-overflow','counter-relations','no-label-refresh','baseline-counter','epoch-exposure-foreign','checkedAt-before-image','over180','fetch-outside-window','fetch-prefix','late-initial-approval','job-reserve','receipt','edited-authority','mixed-image','raw-custody','installed-graph','early-header','final-header','provider-time-ambiguity','reviewer-author-fiction','storage','teardown','population','raw-decision']
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                x=f.full_fixture(Path(t));e=x['evidence'];p=x['provider'];pro=e/'provenance';live=e/'approval-revalidations/before-wake-s02.json';marker=e/'games/wake-s02/markers.txt';text=marker.read_text()
                if case=='root-profile':f.dump(e/'profile.json',{'validLookingButWrong':'original root alias'})
                elif case=='missing-canonical':(e/'profile.json').write_bytes((pro/'profile.json').read_bytes());(pro/'profile.json').unlink()
                elif case=='origin':edit(pro/'effective-freeze-origin.json',lambda d:d.update(effectiveFreezePath='wrong/freeze.json'))
                elif case=='outer-approval':edit(e/'approval.json',lambda d:d.update(unknown=True))
                elif case=='counter-missing':marker.write_text(text.replace(' densityChangedWaves=1','',1))
                elif case=='counter-duplicate':marker.write_text(text.replace(' densityChangedWaves=1',' densityChangedWaves=1 densityChangedWaves=1',1))
                elif case=='counter-fraction':marker.write_text(text.replace('densityChangedWaves=1','densityChangedWaves=1.0',1))
                elif case=='counter-negative':marker.write_text(text.replace('matchedDangerLabels=1','matchedDangerLabels=-1',1))
                elif case=='counter-overflow':marker.write_text(text.replace('refreshedWaves=1','refreshedWaves=1000001',1))
                elif case=='counter-relations':marker.write_text(text.replace('refreshEligiblePlans=1','refreshEligiblePlans=2',1))
                elif case=='no-label-refresh':marker.write_text(text.replace('matchedDangerLabels=1','matchedDangerLabels=0',1))
                elif case=='baseline-counter':b=e/'games/wake-s01/markers.txt';b.write_text(b.read_text().replace('stats maxMs=1','stats maxMs=1 densityChangedWaves=1',1))
                elif case=='epoch-exposure-foreign':
                    for g in a.read_json(x['source']/'benchmarks/keel-wake-v1/games.json')['screen']:
                        if g['bot']=='candidate' and g['opponent']=='diamond':m=e/'games'/g['id']/'markers.txt';m.write_text(m.read_text().replace('densityChangedWaves=1','densityChangedWaves=0'))
                elif case=='checkedAt-before-image':edit(live,lambda d:d.update(checkedAtUTC=d['checkStartedAtUTC']))
                elif case=='over180':edit(live,lambda d:d.update(elapsedMonotonicSeconds=180.000001))
                elif case=='fetch-outside-window':edit(live,lambda d:d['fetchObservations'][0].update(fetchCompletedAtUTC='2026-11-05T09:00:00Z'))
                elif case=='fetch-prefix':edit(live,lambda d:d['fetchObservations'][0].update(transportComplete=False))
                elif case=='late-initial-approval':edit(e/'approval-comment.json',lambda d:d.update(created_at='2026-11-05T01:00:00Z'))
                elif case=='job-reserve':edit(live,lambda d:d.update(remainingJobSecondsAtEnd=2099))
                elif case=='receipt':edit(p/'receipt-ref.json',lambda d:d['object'].update(sha='0'*40))
                elif case=='edited-authority':edit(p/'comments/3.json',lambda d:d.update(updated_at='2026-11-05T03:00:00Z'))
                elif case=='mixed-image':edit(pro/'image-identity/prepared/snapshot.json',lambda d:d['stable']['claims']['processEnvironment'].update(ImageVersion='20261101.186.1'))
                elif case=='raw-custody':(pro/'image-identity/prepared/collector/after-tools/raw/imagedata.bin').write_bytes(b'truncated')
                elif case=='installed-graph':edit(live,lambda d:d['sourceInputsRevalidation'].update(installedGraphUnchanged=False))
                elif case=='early-header':(p/'early/early-header-response.bin').write_bytes(b'changed original404 response')
                elif case=='final-header':(p/'job.log').write_bytes(b'No provider startup header; missing proof')
                elif case=='provider-time-ambiguity':
                    edit(p/'jobs.json',lambda d:d['jobs'][0]['steps'][0].update(completed_at='2026-11-05T00:02:00Z'))
                elif case=='reviewer-author-fiction':edit(x['identity'],lambda d:d.update(reviewerTask='/decoder-author-not-assigned'))
                elif case=='storage':edit(e/'storage-admissions/before-wake-s02.json',lambda d:d.update(freeBytes=0))
                elif case=='teardown':edit(e/'games/wake-s02/teardown.json',lambda d:d.update(elapsedMonotonicSeconds=1801))
                elif case=='population':(e/'games/wake-s02/synthetic.battle.gz').unlink()
                elif case=='raw-decision':edit(e/'decision.json',lambda d:d.update(decision='reject'))
                f.repack(x);code,r=invoke(x)
                self.assertEqual(code,1,(case,r));self.assertNotEqual(r.get('decision'),'PASS',(case,r))


class WakeCounterAdmissionTests(unittest.TestCase):
    def test_each_of_four_fields_all_invalid_wire_variants(self):
        valid='1 start\n1 stats maxMs=1 matchedDangerLabels=1 refreshEligiblePlans=1 refreshedWaves=1 densityChangedWaves=1\n1 end\n'
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'markers.txt'
            for key in a.EXPOSURE:
                for bad in ('-1','1.0','true','NaN','1000001'):
                    with self.subTest(key=key,bad=bad):
                        p.write_text(valid.replace(key+'=1',key+'='+bad))
                        with self.assertRaises(ValueError):a.markers(p,1,True)
                for malformed in (valid.replace(' '+key+'=1',''),valid.replace(key+'=1',key+'=1 '+key+'=1')):
                    p.write_text(malformed)
                    with self.assertRaises(ValueError):a.markers(p,1,True)

    def test_exact_gate_boundaries_and_stage_local_exposure(self):
        rows=f.t.GateBoundaryTests().groups('confirmation')
        for row in rows:
            if row['game']['bot']=='candidate' and row['game']['opponent']=='diamond':row['ownScore']=57;row['totalScore']=100;row['wins']=10
            if row['game']['bot']=='baseline' and row['game']['opponent']=='diamond':row['ownScore']=55;row['totalScore']=100
        self.assertTrue(a.stage_gates('confirmation',rows)['allPass']) # exact+2pp,60wins inclusive
        for row in rows:
            if row['game']['bot']=='candidate' and row['game']['opponent']=='diamond':row['exposure']['densityChangedWaves']=0
        result=a.stage_gates('confirmation',rows);self.assertEqual(result['rawDisposition'],'INCONCLUSIVE')



class FullFreshConfirmationReaderTests(unittest.TestCase):
    def test_actual_cli_fresh_confirmation_with_full_original_screen_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);screen=f.full_fixture(root/'screen');confirmation=f.full_fixture(root/'confirmation',stage='confirmation',run_id=8,job_id=84)
            gate=f.attach_upstream_screen(screen,confirmation)
            f.bind_fixture_upstream_inputs(confirmation,gate)
            code,result=invoke(confirmation)
            self.assertEqual((code,result['decision']),(0,'PASS'),result)
            self.assertEqual((result['registeredRoundBudget'],len(result['validGames'])),(300,18))
            self.assertTrue(result['upstreamScreen']['pass'],result['upstreamScreen'])
            self.assertEqual(result['expectedRun'],8)


if __name__=='__main__':unittest.main()
