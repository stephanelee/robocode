"""D1: bounded serial JFR diagnostics, no policy change or strength inference."""
import argparse,json,os,shutil,time
from pathlib import Path
import keelhelm_melee as core
import keelhelm_resources as resources
BASE='19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b'
LAUNCHER='''#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec java -XX:+UseSerialGC -Xms128m -Xmx512m "-XX:StartFlightRecording=filename=$KEEL_HELM_JFR,settings=profile,disk=true,maxsize=64m,dumponexit=true" -jar KeelHelm.jar
'''
def main(a):
 out=a.evidence.resolve();source=a.source.resolve()
 if a.prepare:
  out.mkdir(parents=True,exist_ok=False)
  shutil.copytree(a.control,out/'control-source',ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
  games=[dict(id=f'diagnostic-{i+1}',arm='control',scenario='melee',rounds=20,seed=2026111500+i,order=['KeelHelm','Diamond Port','Keel','Prism'][i:]+['KeelHelm','Diamond Port','Keel','Prism'][:i])for i in range(2)]
  f=core.collect(source,out,a.pins.resolve(),a.java.absolute(),a.python.absolute(),control_jar_sha=BASE,schedule_rows=games,extra_paths=[Path(__file__),core.ROOT/'tools/keelhelm_resources.py',core.ROOT/'docs/improvements/keelhelm-next-protocol.md'])
  f['diagnostic']=dict(name='D1',launcher=LAUNCHER,concurrency=1,plannedRounds=40,competitive=False)
  core.save(out/'freeze.json',f)
  for g in games:
   p=out/'preflight'/g['id'];p.mkdir(parents=True);core.stage_bots(g,source,out,f,p);(p/'bots/KeelHelm/KeelHelm.sh').write_text(LAUNCHER)
  print('PREPARED',core.sha(out/'freeze.json'));return
 if a.approval_sha!=core.sha(out/'freeze.json'):raise ValueError('Exact independent approval required')
 f=json.loads((out/'freeze.json').read_text());assert f['evidence']==str(out)and f['source']==str(source)
 assert not (out/'results.json').exists(),'No resume/replacement'
 report=dict(status='running',diagnostic='D1',competitive=False,freezeSHA256=a.approval_sha,games=[]);core.save(out/'results.json',report)
 try:
  for g in f['schedule']:
   core.verify_inputs(f);ad=resources.read_resources();ad['diskFreeBytes']=shutil.disk_usage(out).free;core.save(out/(g['id']+'-admission.json'),ad)
   if ad['availablePercent']<35 or ad['diskFreeBytes']<2*1024**3:raise ValueError('Admission failure; no game')
   run=out/'games'/g['id'];run.mkdir(parents=True);dirs,ids=core.stage_bots(g,source,out,f,run)
   launcher=run/'bots/KeelHelm/KeelHelm.sh';launcher.write_text(f['diagnostic']['launcher']);launcher.chmod(0o755)
   (run/'home').mkdir();(run/'recordings').mkdir()
   java=Path(f['java']);env={k:os.environ[k]for k in ('TMPDIR','LANG','LC_ALL','TZ')if k in os.environ}
   env.update(HOME=str(run/'home'),JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+':/usr/bin:/bin:/opt/homebrew/bin',ROBOCODE_PYTHON_BIN=f['python'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',ROBOCODE_ENV_FILE=str(run/'absent.env'),ROBOCODE_GUN_ENV_FILE=str(run/'absent-guns.env'),RUMBLE_JAVA_CACHE=str(source/'.benchmark-cache/retcon-api'),KEEL_HELM_METRICS=str(run/'helm-metrics.txt'),KEEL_HELM_MELEE_FEATURES='0',KEEL_HELM_SEED=str(g['seed']),KEEL_SEED=str(g['seed']),PRISM_SEED=str(g['seed']),KEEL_HELM_JFR=str(run/'helm.jfr'))
   cmd=[f['java'],'-Dtournament.recordingDir='+str(run/'recordings'),'-cp',str(source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),str(core.ROOT/'tournament/scripts/TournamentBattle.java'),'20','900',str(run/'battle.json'),*dirs]
   core.save(run/'invocation.json',dict(command=cmd,environment=env,launcherSHA256=core.sha(launcher)))
   row=dict(**g,status='attempted');monitor=resources.Monitor(run/'resources.jsonl');monitor.start();print('START',g['id'],flush=True)
   try:
    core.execute(cmd,env,run,monitor.register)
    official=json.loads((run/'battle.json').read_text());rec=list((run/'recordings').glob('*.gz'));assert len(rec)==1
    audit=core.audit(rec[0],official,g,ids);telemetry=core.telemetry(run/'helm-metrics.txt',20)
    if audit['metrics']['skippedTurns']:raise ValueError('Own observer skip')
    assert(run/'helm.jfr').stat().st_size>0,'JFR missing'
    row.update(status='complete',official=official,audit=audit,telemetry=telemetry,jfrSHA256=core.sha(run/'helm.jfr'))
   except BaseException as e:row['error']=repr(e)
   finally:monitor.close()
   row['resourceError']=monitor.error;report['games'].append(row);core.save(run/'result.json',row);core.save(out/'results.json',report)
   print('END',g['id'],row['status'],row.get('error'),flush=True)
   if row['status']!='complete'or monitor.error:raise ValueError('Diagnostic stop: invalid evidence')
  report['status']='complete-diagnostic-only'
 except BaseException as e:report.update(status='stopped',error=repr(e));raise
 finally:core.save(out/'results.json',report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--approval-sha')
 for n in ('control','pins','java','python'):p.add_argument('--'+n,type=Path)
 main(p.parse_args())
