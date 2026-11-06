"""Issue134: isolated predictive movement, 88 serial games; exact paired randomization."""
import argparse,gzip,hashlib,json,math,os,random,re,secrets,shutil,sys,zipfile
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import keelhelm_melee as core
import keelhelm_resources as resources
import keelhelm_calibration as calibration
from keelhelm_randomization import exact_tail,unrank
BASE=calibration.BASE
NAMES=calibration.NAMES

def make_schedule(assignment):
    assert len(assignment)==len(set(assignment))==16 and all(i in range(32)for i in assignment)
    rng=random.Random(2026112601);rows=[];pair=0
    for scenario,count,rounds in [('melee',32,5),('Diamond Port',4,10),('Retcon',4,10),('Walls',2,5),('SpinBot',2,5)]:
        for j in range(count):
            names=NAMES if scenario=='melee'else['KeelHelm',scenario]
            while True:
                points=[[rng.randint(100,900),rng.randint(100,900),rng.randrange(360)] if scenario=='melee' else [rng.randint(80,720),rng.randint(80,520),rng.randrange(360)]for name in names]
                if all(math.dist(a[:2],b[:2])>=250 for i,a in enumerate(points)for b in points[:i]):break
            order=list(names[j%len(names):]+names[:j%len(names)])
            first='candidate'if (pair in assignment if scenario=='melee'else j%2==0)else'control'
            for slot,arm in enumerate([first,'control'if first=='candidate'else'candidate']):
                rows.append(dict(id=f'{pair+1:02d}-{slot+1}-{arm}',pair=pair,slot=slot,arm=arm,mode='fixed',scenario=scenario,rounds=rounds,seed=2026112700+pair,order=order,positions=dict(zip(names,points))))
            pair+=1
    assert len(rows)==88 and sum(g['rounds']for g in rows)==520
    return rows

def stage(g,f,out,run):
    dirs,ids=core.stage_bots(g,Path(f['source']),out,f,run)
    for directory,(name,version)in zip(dirs,ids):
        folder=Path(directory);launch=folder/(folder.name+'.sh');assert launch.exists()
        launch.rename(folder/'.pairing-original.sh')
        launch.write_text('#!/bin/sh\nset -eu\nexport SERVER_URL="${SERVER_URL%/}/identity/'+quote(name,safe='')+'"\nexec "$(dirname "$0")/.pairing-original.sh"\n');launch.chmod(0o755)
    return dirs,ids

def inspect_transport(path,game,ids):
    # Reuse the reviewed stream/handshake checks, with the manifest geometry.
    original=calibration.positions
    try:
        calibration.positions=lambda g:g['positions']
        return calibration.inspect_transport(path,game,ids)
    finally:calibration.positions=original

def starts(recording,game):
    original=calibration.positions
    try:
        calibration.positions=lambda g:g['positions']
        return calibration.starts(recording,game)
    finally:calibration.positions=original

def summarize(games,assignment):
    means={};exact={};survival={};pairs=[]
    def share(g,key):
        denominator=sum(r[key]for r in g['official']['results'])
        return Fraction(g['helm'][key],denominator)if denominator else Fraction()
    for scenario in ['melee','Diamond Port','Retcon','Walls','SpinBot']:
        means[scenario]={}
        for arm in ['control','candidate']:
            selected=[g for g in games if g['scenario']==scenario and g['arm']==arm]
            exact[scenario,arm]=sum((share(g,'totalScore')for g in selected),Fraction())/len(selected)
            survival[scenario,arm]=sum((share(g,'survival')for g in selected),Fraction())/len(selected)
            means[scenario][arm]=dict(meanShare=float(exact[scenario,arm]),meanShareExact=str(exact[scenario,arm]),survivalShare=float(survival[scenario,arm]),scores=[g['helm']['totalScore']for g in selected],firstPlaces=sum(g['helm']['firstPlaces']for g in selected))
        means[scenario]['deltaPP']=float(100*(exact[scenario,'candidate']-exact[scenario,'control']))
    x=[]
    for i in range(32):
        first,second=sorted([g for g in games if g['pair']==i],key=lambda g:g['slot']);v=share(first,'totalScore')-share(second,'totalScore');x.append(v)
        assert(first['arm']=='candidate')==(i in assignment)
        pairs.append(dict(pair=i,firstArm=first['arm'],slotContrastExact=str(v),candidateEffectPP=float(100*v*(1 if i in assignment else -1))))
    test=exact_tail(x,assignment)
    exposure=sum(int(v)for g in games if g['arm']=='candidate'for v in g['movementDifferences'])
    gates=dict(meleeShare=exact['melee','candidate']>=Fraction(1,5),meleeGain=exact['melee','candidate']-exact['melee','control']>=Fraction(1,50),survival=survival['melee','candidate']-survival['melee','control']>=-Fraction(3,100),exposure=exposure>=100,randomization=Fraction(test['pExact'])<=Fraction(1,20),**{n:exact[n,'candidate']-exact[n,'control']>=-Fraction(1,20)for n in ['Diamond Port','Retcon','Walls','SpinBot']})
    return dict(scenarios=means,pairs=pairs,randomization=test,movementDifferences=exposure,gates=gates,decision='accept-measured-melee-improvement'if all(gates.values())else'reject-improvement-claim')

def play(g,f,out,monitor):
    row=dict(**g,status='setup-failed');run=out/'games'/g['id']
    try:
        run.mkdir(parents=True);dirs,ids=stage(g,f,out,run);(run/'home').mkdir();(run/'recordings').mkdir();java=Path(f['java'])
        env={k:os.environ[k]for k in ('TMPDIR','LANG','LC_ALL','TZ')if k in os.environ}
        env.update(HOME=str(run/'home'),JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+':/usr/bin:/bin:/opt/homebrew/bin',ROBOCODE_PYTHON_BIN=f['python'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',ROBOCODE_ENV_FILE=str(run/'absent.env'),ROBOCODE_GUN_ENV_FILE=str(run/'absent-guns.env'),RUMBLE_JAVA_CACHE=str(Path(f['source'])/'.benchmark-cache/retcon-api'),KEEL_HELM_METRICS=str(run/'helm-metrics.txt'),KEEL_HELM_SEED=str(g['seed']),KEEL_SEED=str(g['seed']),PRISM_SEED=str(g['seed']))
        config=dict(run=str(run),java=f['java'],serverJar=f['serverJar'],runner=str(Path(f['source'])/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),adapter=str(ROOT/'tournament/scripts/PairedBattle.java'),rounds=g['rounds'],bots=dirs,identities=dict(ids),positions=g['positions']);core.save(run/'config.json',config)
        cmd=[f['python'],str(ROOT/'tournament/tools/paired_transport.py'),str(run/'config.json')];core.save(run/'invocation.json',dict(command=cmd,environment=env));row['status']='attempted';print('START',g['id'],g['scenario'],flush=True)
        def check():
            if monitor.error:raise ValueError(monitor.error)
        core.execute(cmd,env,run,monitor.register,failure_check=check)
        official=json.loads((run/'battle.json').read_text());assert official['numberOfRounds']==g['rounds']and sorted((b['name'],b['version'])for b in official['results'])==sorted(ids)
        raw=list((run/'recordings').glob('*.gz'));assert len(raw)==1
        audit=core.audit(raw[0],official,g,ids);assert audit['metrics']['skippedTurns']==0
        telemetry=core.telemetry(run/'helm-metrics.txt',g['rounds']);transport=inspect_transport(run/'transport.json',g,ids);start=starts(raw[0],g)
        core.save(run/'starts.json',start);own=next(b for b in official['results']if b['name']=='KeelHelm')
        differences=re.findall(r'riskDifferences=(\d+)',(run/'helm-metrics.txt').read_text());assert len(differences)==g['rounds']if g['arm']=='candidate'else not differences
        row.update(movementDifferences=differences,status='complete',official=official,helm=own,audit=audit,telemetry=telemetry,transportAudit=transport,startAudit=start,recording=str(raw[0].relative_to(out)))
    except BaseException as error:row['error']=repr(error)
    if run.exists():core.save(run/'result.json',row)
    print('END',g['id'],row['status'],row.get('error'),flush=True);return row

def prepare(a):
    out=a.evidence.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not(out/'freeze.json').exists()
    project=ROOT/'benchmarks/keelhelm-risk/candidate-project'
    assert core.sha(ROOT/'experiments/KeelHelm/KeelHelm.jar')==BASE
    shutil.copytree(ROOT/'experiments/KeelHelm',out/'control-source',ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
    # Draw once, exclusive-create, before deriving any treatment-labelled schedule.
    draw=out/'assignment.json'
    if not draw.exists():
        rank=secrets.randbelow(math.comb(32,16))
        with draw.open('x')as h:json.dump(dict(mechanism='one secrets.randbelow(C(32,16)); lexicographic unranking',rank=rank,space=math.comb(32,16),candidateFirst=unrank(32,16,rank),createdUTC=datetime.now(timezone.utc).isoformat(),sourceSHA256=core.sha(Path(__file__))),h,indent=2)
    assignment=json.loads(draw.read_text());assert assignment['candidateFirst']==unrank(32,16,assignment['rank'])
    rows=make_schedule(assignment['candidateFirst']);core.save(out/'schedule.json',rows)
    with zipfile.ZipFile(a.source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar')as z:(out/'server.jar').write_bytes(z.read('robocode-tankroyale-server.jar'))
    extras=core.files(ROOT/'benchmarks/keelhelm-risk/validation')+[Path(__file__),ROOT/'tournament/tools/keelhelm_randomization.py',ROOT/'tournament/tools/test_keelhelm_randomization.py',ROOT/'tournament/tools/test_keelhelm_risk.py',ROOT/'tournament/tools/keelhelm_calibration.py',ROOT/'tournament/tools/paired_transport.py',ROOT/'tournament/scripts/PairedBattle.java',ROOT/'docs/improvements/keelhelm-risk-protocol.md',out/'server.jar',draw,out/'schedule.json']
    f=core.collect(a.source.resolve(),out,a.pins.resolve(),a.java.absolute(),a.python.absolute(),control_jar_sha=BASE,schedule_rows=rows,extra_paths=extras,candidate_project=project)
    f['serverJar']=str(out/'server.jar');f['assignment']=assignment;f['benchmark']=dict(name='isolated-risk',games=88,rounds=520,concurrency=1);core.save(out/'freeze.json',f)
    for g in rows:
        run=out/'preflight'/g['id'];run.mkdir(parents=True);dirs,ids=stage(g,f,out,run);assert len(set(dirs))==len(ids)==len(g['order'])
    print('PREPARED',core.sha(out/'freeze.json'),flush=True)

def run(a):
    out=a.evidence.resolve();assert a.approval_sha==core.sha(out/'freeze.json');f=json.loads((out/'freeze.json').read_text())
    assignment=f['assignment']['candidateFirst'];assert f['schedule']==make_schedule(assignment)and f['source']==str(a.source.resolve())and f['evidence']==str(out);assert not(out/'results.json').exists()
    report=dict(status='running',freezeSHA256=a.approval_sha,plannedGames=88,plannedRounds=520,games=[],waves=[]);core.save(out/'results.json',report)
    try:
        for g in f['schedule']:
            core.verify_inputs(f);ad=resources.read_resources();ad['diskFreeBytes']=shutil.disk_usage(out).free;core.save(out/(g['id']+'-admission.json'),ad);assert ad['availablePercent']>=35 and ad['diskFreeBytes']>=2*1024**3
            monitor=resources.Monitor(out/(g['id']+'-resources.jsonl'));monitor.start()
            try:row=play(g,f,out,monitor)
            finally:monitor.close()
            report['games'].append(row);report['waves'].append(dict(game=g['id'],samples=monitor.samples,resourceError=monitor.error));core.save(out/'results.json',report)
            if row['status']!='complete'or monitor.error:raise ValueError(dict(slotError=row.get('error'),resourceError=monitor.error))
        report.update(status='complete',completedGames=88,completedRounds=520,summary=summarize(report['games'],assignment))
    except BaseException as error:report.update(status='invalid',error=repr(error));raise
    finally:core.save(out/'results.json',report)
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--approval-sha')
    for n in ['pins','java','python']:p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:run(a)
