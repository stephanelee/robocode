"""Fresh serial isolated melee comparisons after independently audited timing diagnostics."""
import argparse,json,shutil,hashlib
from pathlib import Path
import keelhelm_followups as legacy
import keelhelm_melee as core
BASE='19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b'
POLICIES={1:('predicted-risk',8,'riskDifferences'),2:('radar-revisit',16,'radarRevisits'),3:('shot-obstruction',32,'obstructedShots'),4:('duel-transfer',64,'transferredScans')}
legacy_summary=legacy.summary
def summary(games,stage):
 import re
 s=legacy_summary(games,None);key=POLICIES[stage][2]
 exposure=sum(int(re.search(r'\b'+key+r'=(\d+)',line)[1])for g in games if g['arm']=='candidate'and g['scenario']=='melee'for line in g['telemetry']if ' stats 'in line)
 s['policyChangeExposure']=exposure;s['gates']['observedPolicyChange']=exposure>0;s['decision']='accept'if all(s['gates'].values())else'reject';return s

def prepare(a):
 out=a.evidence.resolve();out.mkdir(parents=True,exist_ok=False);name,bit,key=POLICIES[a.stage]
 previous=json.loads(a.previous_audit.read_text());assert previous['experimentStatus']=='complete'and previous['decision']in ('accept','reject')
 assert previous['nextAcceptedMask']==a.control_mask
 assert a.control_mask&bit==0
 assert not a.control_mask & 7,'Rejected or invalid earlier policies must stay off'
 control=a.control.resolve();candidate=a.candidate_project.resolve()
 controlsha=core.sha(control/'KeelHelm.jar');assert previous['nextAcceptedJarSHA256']==controlsha
 feature_sha=core.sha(candidate/'KeelHelm.jar')
 manifest={str(p.relative_to(candidate)):core.sha(p)for p in core.files(candidate)}
 manifest_sha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if a.stage>1:
  assert previous['study']=='next-serial-v1'and previous['stage']==a.stage-1
  assert previous['featureJarSHA256']==feature_sha and previous['featureProjectManifestSHA256']==manifest_sha
 for label,src in [('control-source',control),('candidate-source',candidate)]:shutil.copytree(src,out/label,ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
 shutil.copy2(a.previous_audit,out/'previous-outcome-review.json')
 games=legacy.schedule(a.stage,True)
 for i,g in enumerate(games):g['seed']=2026111600+100*a.stage+i//2
 config=dict(study='next-serial-v1',featureJarSHA256=feature_sha,featureProjectManifestSHA256=manifest_sha,stage=a.stage,name=name,newBit=bit,controlMask=a.control_mask,candidateMask=a.control_mask|bit,concurrency=1,loadPlan='next-serial',controlJarSHA256=controlsha,candidateJarSHA256=core.sha(candidate/'KeelHelm.jar'),previousAuditSHA256=core.sha(a.previous_audit))
 core.save(out/'registration.json',config)
 extras=[Path(__file__),Path(legacy.__file__),core.ROOT/'tools/keelhelm_resources.py',core.ROOT/'docs/improvements/keelhelm-next-protocol.md',core.ROOT/'docs/improvements/keelhelm-next-implementations.md',core.ROOT/'tools/test_keelhelm_next.py',out/'registration.json',out/'previous-outcome-review.json',a.timing_audit,core.ROOT/'docs/improvements/keelhelm-timing-d2-results.md']
 f=core.collect(a.source.resolve(),out,a.pins.resolve(),a.java.absolute(),a.python.absolute(),control_jar_sha=controlsha,schedule_rows=games,extra_paths=extras,candidate_project=candidate);f['stageConfig']=config;core.save(out/'freeze.json',f)
 for g in games:
  p=out/'preflight'/g['id'];p.mkdir(parents=True);dirs,ids=core.stage_bots(g,a.source.resolve(),out,f,p);assert len(set(dirs))==len(ids)==len(g['order'])
 print('PREPARED',core.sha(out/'freeze.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--approval-sha');p.add_argument('--stage',type=int,choices=range(1,5));p.add_argument('--control-mask',type=int,default=0)
 for n in ('control','previous-audit','timing-audit','pins','java','python'):p.add_argument('--'+n,type=Path)
 p.add_argument('--candidate-project',type=Path,default=core.ROOT/'benchmarks/keelhelm-next/candidate-project')
 a=p.parse_args()
 if a.prepare:prepare(a)
 else:legacy.summary=summary;legacy.run(a)
