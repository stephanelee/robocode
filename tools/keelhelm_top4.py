"""Issue127: four cold latest-published top-four melee games; prepare/review/run once."""
import argparse,json,os,shutil
from fractions import Fraction
from pathlib import Path
import keelhelm_melee as core
import keelhelm_resources as resources

def schedule():
 names=['KeelHelm','Diamond Port','Keel','Prism']
 return [dict(id=f'melee-{i+1}',arm='candidate',scenario='melee',rounds=20,seed=2026112100+i,order=names[i:]+names[:i])for i in range(4)]

def standings(games):
 rows={}
 for game in games:
  total=sum(b['totalScore']for b in game['official']['results'])
  for b in game['official']['results']:
   r=rows.setdefault(b['name'],dict(name=b['name'],version=b['version'],totalScore=0,survival=0,bulletDamage=0,firstPlaces=0,shares=[]))
   assert r['version']==b['version']
   for k in ['totalScore','survival','bulletDamage','firstPlaces']:r[k]+=b[k]
   r['shares'].append(Fraction(b['totalScore'],total))
 for r in rows.values():
  mean=sum(r.pop('shares'),Fraction())/len(games);r.update(meanScoreShare=float(mean),meanScoreShareExact=str(mean))
 return sorted(rows.values(),key=lambda r:(-r['totalScore'],r['name']))

def prepare(a):
 out=a.evidence.resolve();out.mkdir(parents=True,exist_ok=False);project=core.ROOT/'experiments/KeelHelm'
 assert json.loads((project/'bot/KeelHelm.json').read_text())['version']=='0.2.1'
 shutil.copytree(project,out/'control-source',ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
 shutil.copytree(project,out/'entrant-source',ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
 extra=[Path(__file__),core.ROOT/'tools/keelhelm_resources.py',core.ROOT/'tools/test_keelhelm_top4.py',core.ROOT/'docs/improvements/keelhelm-top4-latest-protocol.md']
 f=core.collect(a.source.resolve(),out,a.pins.resolve(),a.java.absolute(),a.python.absolute(),control_jar_sha=core.sha(project/'KeelHelm.jar'),schedule_rows=schedule(),extra_paths=extra,candidate_project=project)
 f['benchmark']=dict(name='latest-published-top4',games=4,rounds=80,concurrency=1,featureMask=0,entrantJarSHA256=core.sha(project/'KeelHelm.jar'),publishedMain='255f3230e4854a1db00e0aa214030db5f7c28f2d')
 core.save(out/'freeze.json',f)
 for g in f['schedule']:
  run=out/'preflight'/g['id'];run.mkdir(parents=True);dirs,ids=core.stage_bots(g,a.source.resolve(),out,f,run);assert len(set(dirs))==len(ids)==4
 print('PREPARED',core.sha(out/'freeze.json'),flush=True)

def play(g,f,out,monitor):
 run=out/'games'/g['id'];run.mkdir(parents=True);source=Path(f['source']);java=Path(f['java'])
 dirs,ids=core.stage_bots(g,source,out,f,run);(run/'home').mkdir();(run/'recordings').mkdir()
 env={k:os.environ[k]for k in ('TMPDIR','LANG','LC_ALL','TZ')if k in os.environ}
 env.update(HOME=str(run/'home'),JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+':/usr/bin:/bin:/opt/homebrew/bin',ROBOCODE_PYTHON_BIN=f['python'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',ROBOCODE_ENV_FILE=str(run/'absent.env'),ROBOCODE_GUN_ENV_FILE=str(run/'absent-guns.env'),RUMBLE_JAVA_CACHE=str(source/'.benchmark-cache/retcon-api'),KEEL_HELM_METRICS=str(run/'helm-metrics.txt'),KEEL_HELM_MELEE_FEATURES='0',KEEL_HELM_SEED=str(g['seed']),KEEL_SEED=str(g['seed']),PRISM_SEED=str(g['seed']))
 cmd=[f['java'],'-Dtournament.recordingDir='+str(run/'recordings'),'-cp',str(source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),str(core.ROOT/'tournament/scripts/TournamentBattle.java'),'20','900',str(run/'battle.json'),*dirs]
 core.save(run/'invocation.json',dict(command=cmd,environment=env,effectiveFeatureMask=0))
 row=dict(**g,status='attempted');print('START',g['id'],flush=True)
 try:
  def check():
   if monitor.error:raise ValueError(monitor.error)
  core.execute(cmd,env,run,monitor.register,failure_check=check)
  official=json.loads((run/'battle.json').read_text());assert official['numberOfRounds']==20
  assert sorted((b['name'],b['version'])for b in official['results'])==sorted(ids)
  recordings=list((run/'recordings').glob('*.gz'));assert len(recordings)==1
  audit=core.audit(recordings[0],official,g,ids);assert audit['metrics']['skippedTurns']==0
  telemetry=core.telemetry(run/'helm-metrics.txt',20)
  row.update(status='complete',official=official,audit=audit,telemetry=telemetry,recording=str(recordings[0].relative_to(out)))
 except BaseException as e:row['error']=repr(e)
 core.save(run/'result.json',row);print('END',g['id'],row['status'],row.get('error'),flush=True);return row

def run(a):
 out=a.evidence.resolve();assert a.approval_sha==core.sha(out/'freeze.json')
 f=json.loads((out/'freeze.json').read_text());assert f['source']==str(a.source.resolve())and f['evidence']==str(out)
 assert f['benchmark']['concurrency']==1 and f['benchmark']['featureMask']==0
 assert not(out/'results.json').exists(),'No resume/replacement'
 report=dict(status='running',freezeSHA256=a.approval_sha,plannedGames=4,plannedRounds=80,games=[],waves=[]);core.save(out/'results.json',report)
 try:
  for g in f['schedule']:
   core.verify_inputs(f);ad=resources.read_resources();ad['diskFreeBytes']=shutil.disk_usage(out).free;core.save(out/(g['id']+'-admission.json'),ad)
   if ad['availablePercent']<35 or ad['diskFreeBytes']<2*1024**3:raise ValueError('Admission failed; no game launched')
   monitor=resources.Monitor(out/(g['id']+'-resources.jsonl'));monitor.start()
   try:row=play(g,f,out,monitor)
   finally:monitor.close()
   report['games'].append(row);report['waves'].append(dict(game=g['id'],samples=monitor.samples,resourceError=monitor.error));core.save(out/'results.json',report)
   if row['status']!='complete'or monitor.error:raise ValueError(monitor.error or row.get('error'))
  report.update(status='complete',completedGames=4,completedRounds=80,standings=standings(report['games']))
 except BaseException as e:report.update(status='invalid',error=repr(e));raise
 finally:core.save(out/'results.json',report)
 print(json.dumps(report['standings'],indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--approval-sha')
 for n in ['pins','java','python']:p.add_argument('--'+n,type=Path)
 a=p.parse_args()
 if a.prepare:prepare(a)
 else:run(a)
