"""Issue125: four independently reviewed feature interventions, two games per wave."""
import argparse
import concurrent.futures
from fractions import Fraction
import json
import os
import re
from pathlib import Path
import shutil
import time
import keelhelm_melee as core
import keelhelm_resources as resources

ROOT=core.ROOT
BASELINE_JAR='19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b'
NAMES={1:'adaptive-gun',2:'accuracy-power',3:'target-selection',4:'predicted-risk'}

def schedule(stage):
    games=core.schedule()
    diamond=[]
    for slot in range(2):
        for arm in (['control','candidate'] if slot==0 else ['candidate','control']):
            diamond.append(dict(id=f'diamond-{slot+1}-{arm}',arm=arm,scenario='Diamond Port',rounds=10,
                seed=2026110600+stage*100+30+slot,order=['KeelHelm','Diamond Port'] if slot==0 else ['Diamond Port','KeelHelm']))
    games[8:8]=diamond
    for i,g in enumerate(games):g['seed']=2026110600+stage*100+i//2
    assert len(games)==20 and sum(g['rounds'] for g in games)==260
    return games

def summary(games,stage=None):
    scenarios={};ratios={}
    for name in ('melee','Diamond Port','Retcon','Walls','SpinBot'):
        scenarios[name]={}
        for arm in ('control','candidate'):
            rows=[g for g in games if g['scenario']==name and g['arm']==arm]
            values=[Fraction(g['helm']['totalScore'],sum(r['totalScore'] for r in g['official']['results'])) for g in rows]
            mean=sum(values,Fraction())/len(values);ratios[name,arm]=mean
            scenarios[name][arm]=dict(meanShare=float(mean),meanShareExact=str(mean),
                gameShares=[float(v) for v in values],firstPlaces=sum(g['helm']['firstPlaces'] for g in rows))
        scenarios[name]['deltaPP']=float(100*(ratios[name,'candidate']-ratios[name,'control']))
    gates=dict(meleeShare=ratios['melee','candidate']>=Fraction(1,5),
        meleeGain=ratios['melee','candidate']-ratios['melee','control']>=Fraction(1,50))
    for name in ('Diamond Port','Retcon','Walls','SpinBot'):
        gates[name]=ratios[name,'candidate']-ratios[name,'control']>=-Fraction(1,20)
    exposure=0
    if stage is not None:
        for game in games:
            if game['scenario']!='melee' or game['arm']!='candidate':continue
            for line in game['telemetry']:
                if ' stats ' not in line:continue
                if stage==1:
                    match=re.search(r'meleeGunChoices=\[(\d+), (\d+), (\d+)\]',line)
                    exposure+=int(match[1])+int(match[2])
                else:exposure+=int(re.search({2:r'powerChanges=(\d+)',3:r'smartDifferences=(\d+)',4:r'riskDifferences=(\d+)'}[stage],line)[1])
        gates['observedPolicyChange']=exposure>0
    return dict(scenarios=scenarios,gates=gates,policyChangeExposure=exposure,decision='accept' if all(gates.values()) else 'reject')

def prepare(args):
    output=args.evidence.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'freeze.json').exists() or (output/'registration.json').exists():raise ValueError('Never replace registration')
    bit=1<<(args.stage-1);candidate_mask=args.control_mask|bit
    if args.control_mask&bit:raise ValueError('New bit already enabled')
    if args.control_mask<0 or args.control_mask&~(bit-1):raise ValueError('Control mask contains untested later bits')
    previous=None
    if args.stage==1:
        if args.control_mask!=0:raise ValueError('First control must be0.2.1')
    else:
        if args.previous_audit is None:raise ValueError('Previous independent outcome audit required')
        previous=json.loads(args.previous_audit.read_text())
        if previous['stage']!=args.stage-1 or previous['nextAcceptedMask']!=args.control_mask:
            raise ValueError('Previous independently accepted mask differs')
    control=args.control_project.resolve();candidate=ROOT/'experiments/KeelHelm'
    feature_jar=core.sha(candidate/'KeelHelm.jar')
    feature_manifest={str(p.relative_to(candidate)):core.sha(p) for p in core.files(candidate)}
    import hashlib
    feature_manifest_sha=hashlib.sha256(json.dumps(feature_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    expected=BASELINE_JAR if args.control_mask==0 else feature_jar
    if previous is not None:
        if previous['featureJarSHA256']!=feature_jar or previous['featureProjectManifestSHA256']!=feature_manifest_sha:
            raise ValueError('The single frozen feature JAR/source project changed')
        if previous['nextAcceptedJarSHA256']!=expected:raise ValueError('Prior independently accepted control JAR differs')
    if core.sha(control/'KeelHelm.jar')!=expected:raise ValueError('Control JAR not last accepted version')
    for name,src in [('control-source',control),('candidate-source',candidate)]:
        shutil.copytree(src,output/name,ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
    registration=dict(stage=args.stage,name=NAMES[args.stage],newBit=bit,controlMask=args.control_mask,
        candidateMask=candidate_mask,controlJarSHA256=expected,candidateJarSHA256=core.sha(candidate/'KeelHelm.jar'),
        featureJarSHA256=feature_jar,featureProjectManifestSHA256=feature_manifest_sha,
        previousAuditSHA256=core.sha(args.previous_audit) if args.previous_audit else None)
    core.save(output/'registration.json',registration)
    extra=[Path(__file__),ROOT/'tools/keelhelm_resources.py',ROOT/'tools/test_keelhelm_followups.py',
        ROOT/'experiments/KeelHelm/test/MeleeGunTest.java',ROOT/'docs/improvements/keelhelm-melee-followups-protocol.md',
        output/'registration.json']
    if args.previous_audit:
        shutil.copy2(args.previous_audit,output/'previous-outcome-review.json');extra.append(output/'previous-outcome-review.json')
    freeze=core.collect(args.source.resolve(),output,args.pins.resolve(),args.java.absolute(),args.python.absolute(),
        control_jar_sha=expected,schedule_rows=schedule(args.stage),extra_paths=extra)
    freeze['stageConfig']=registration;core.save(output/'freeze.json',freeze)
    for game in freeze['schedule']:
        run=output/'staging-preflight'/game['id'];run.mkdir(parents=True)
        dirs,ids=core.stage_bots(game,args.source.resolve(),output,freeze,run)
        if len(set(dirs))!=len(game['order']) or len(ids)!=len(game['order']):raise ValueError('Staging collision')
    print('PREPARED20ROSTERS '+core.sha(output/'freeze.json'),flush=True)

def play(game,freeze,output,monitor):
    try:return _play(game,freeze,output,monitor)
    except BaseException as error:
        row=dict(**game,status='setup-failed',error=repr(error),featureMask=freeze['stageConfig'][game['arm']+'Mask'])
        run=output/'games'/game['id']
        try:
            if run.exists():core.save(run/'result.json',row)
        except OSError:pass # The top-level report still retains this slot if its staging disk filled.
        print('FAILED SETUP '+game['id']+' '+repr(error),flush=True)
        return row

def _play(game,freeze,output,monitor):
    source=Path(freeze['source']);java=Path(freeze['java']);python=Path(freeze['python'])
    run=output/'games'/game['id'];run.mkdir(parents=True)
    dirs,identities=core.stage_bots(game,source,output,freeze,run)
    home=run/'home';home.mkdir();rec=run/'recordings';rec.mkdir()
    env={k:os.environ[k] for k in ('TMPDIR','LANG','LC_ALL','TZ') if k in os.environ}
    mask=freeze['stageConfig'][game['arm']+'Mask']
    env.update(HOME=str(home),JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+':/usr/bin:/bin:/opt/homebrew/bin',
        ROBOCODE_PYTHON_BIN=str(python),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        ROBOCODE_ENV_FILE=str(run/'absent.env'),ROBOCODE_GUN_ENV_FILE=str(run/'absent-guns.env'),
        RUMBLE_JAVA_CACHE=str(source/'.benchmark-cache/retcon-api'),KEEL_HELM_METRICS=str(run/'helm-metrics.txt'),
        KEEL_HELM_MELEE_FEATURES=str(mask),KEEL_HELM_SEED=str(game['seed']),KEEL_SEED=str(game['seed']),PRISM_SEED=str(game['seed']))
    command=[str(java),'-Dtournament.recordingDir='+str(rec),'-cp',str(source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),
        str(ROOT/'tournament/scripts/TournamentBattle.java'),str(game['rounds']),'900',str(run/'battle.json'),*dirs]
    core.save(run/'invocation.json',dict(command=command,environment=env,effectiveFeatureMask=mask))
    print('START '+game['id']+' mask='+str(mask),flush=True)
    row=dict(**game,status='attempted',featureMask=mask)
    try:
        core.execute(command,env,run,monitor.register)
        official=json.loads((run/'battle.json').read_text())
        if official['numberOfRounds']!=game['rounds'] or sorted((r['name'],r['version']) for r in official['results'])!=sorted(identities):raise ValueError('Official identities/rounds differ')
        recordings=list(rec.glob('*.gz'))
        if len(recordings)!=1:raise ValueError('One original recording required')
        audit=core.audit(recordings[0],official,game,identities)
        if audit['metrics']['skippedTurns']:raise ValueError('Own observer skip')
        telemetry=core.telemetry(run/'helm-metrics.txt',game['rounds'])
        own=next(r for r in official['results'] if r['name']=='KeelHelm')
        row.update(status='complete',official=official,helm=own,scoreShare=own['totalScore']/sum(r['totalScore'] for r in official['results']),
            recording=str(recordings[0].relative_to(output)),audit=audit,telemetry=telemetry)
    except BaseException as error:row['error']=repr(error)
    core.save(run/'result.json',row);print('END '+game['id']+' '+row['status']+' '+str(row.get('scoreShare')),flush=True)
    return row

def run(args):
    output=args.evidence.resolve();path=output/'freeze.json'
    if args.approval_sha!=core.sha(path):raise ValueError('Exact independently reviewed freeze required')
    freeze=json.loads(path.read_text())
    if freeze['source']!=str(args.source.resolve()) or freeze['evidence']!=str(output):raise ValueError('Execution path changed')
    result=output/'results.json'
    if result.exists():raise ValueError('No resume or replacement')
    report=dict(status='running',stageConfig=freeze['stageConfig'],freezeSHA256=core.sha(path),plannedGames=20,plannedRounds=260,games=[],waves=[])
    core.save(result,report)
    try:
        for index in range(0,20,2):
            core.verify_inputs(freeze)
            while True:
                admission=resources.read_resources();admission['diskFreeBytes']=shutil.disk_usage(output).free
                with (output/'admission.jsonl').open('a') as log:log.write(json.dumps(admission)+'\n')
                if admission['availablePercent']>=35 and admission['diskFreeBytes']>=2*1024**3:break
                print('WAIT resource admission '+str(admission['availablePercent'])+'%',flush=True);time.sleep(10)
            wave=index//2+1;monitor=resources.Monitor(output/f'wave-{wave}-resources.jsonl');monitor.start()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(play,g,freeze,output,monitor) for g in freeze['schedule'][index:index+2]]
                    rows=[]
                    for game,future in zip(freeze['schedule'][index:index+2],futures):
                        try:rows.append(future.result())
                        except BaseException as error:rows.append(dict(**game,status='setup-failed',error=repr(error)))
            finally:monitor.close()
            report['games']+=rows;report['waves'].append(dict(wave=wave,resourceError=monitor.error,samples=monitor.samples))
            core.save(result,report)
            if monitor.error or any(r['status']!='complete' for r in rows):raise ValueError(monitor.error or 'Wave evidence failure')
        report.update(status='complete',completedGames=20,completedRounds=260,summary=summary(report['games'],freeze['stageConfig']['stage']))
    except BaseException as error:
        report.update(status='invalid',error=repr(error));raise
    finally:core.save(result,report)
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--stage',type=int,choices=range(1,5));p.add_argument('--control-mask',type=int)
    p.add_argument('--control-project',type=Path);p.add_argument('--previous-audit',type=Path)
    for name in ('pins','java','python'):p.add_argument('--'+name,type=Path)
    p.add_argument('--approval-sha');a=p.parse_args()
    if a.prepare:
        if any(getattr(a,k) is None for k in ('stage','control_mask','control_project','pins','java','python')):p.error('Preparation inputs required')
        prepare(a)
    else:run(a)
