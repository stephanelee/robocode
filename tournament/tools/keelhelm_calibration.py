"""Issue132: prospective identical-code A/A calibration; no gameplay modification."""
import argparse,collections,gzip,json,os,shutil,sys,hashlib
from pathlib import Path
from fractions import Fraction
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import keelhelm_melee as core
import keelhelm_resources as resources
BASE='19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b'
NAMES=['KeelHelm','Diamond Port','Keel','Prism']

def schedule():
    rows=[];pair=0
    for layout in range(4):
        for scenario,rounds in [('melee',10),('Diamond Port',5)]:
            for mode in ['random','fixed']:
                order=NAMES[layout:]+NAMES[:layout]if scenario=='melee'else(['KeelHelm','Diamond Port']if layout%2==0 else['Diamond Port','KeelHelm'])
                for arm in (['A','B']if pair%2==0 else['B','A']):
                    rows.append(dict(id=f'{pair+1:02d}-{mode}-{arm}',pair=pair,layout=layout,mode=mode,arm=arm,scenario=scenario,rounds=rounds,seed=2026112400+pair,order=order))
                pair+=1
    assert len(rows)==32 and sum(g['rounds']for g in rows)==240
    return rows

def positions(game):
    if game['mode']=='random':return None
    i=game['layout']
    if game['scenario']=='melee':
        points=[[200,200,45],[800,200,135],[800,800,225],[200,800,315]]
        return {name:points[(j+i)%4]for j,name in enumerate(NAMES)}
    points=[[[160,150,20],[640,450,200]],[[160,450,340],[640,150,160]],[[200,300,0],[600,300,180]],[[400,120,90],[400,480,270]]][i]
    return dict(zip(['KeelHelm','Diamond Port'],points))

def stage(g,f,out,run):
    staged=dict(g,arm='candidate'if g['arm']=='B'else'control')
    dirs,ids=core.stage_bots(staged,Path(f['source']),out,f,run)
    for directory,(name,version)in zip(dirs,ids):
        folder=Path(directory);launch=folder/(folder.name+'.sh');assert launch.exists()
        launch.rename(folder/'.pairing-original.sh')
        launch.write_text('#!/bin/sh\nset -eu\nexport SERVER_URL="${SERVER_URL%/}/identity/'+quote(name,safe='')+'"\nexec "$(dirname "$0")/.pairing-original.sh"\n');launch.chmod(0o755)
    return dirs,ids

def inspect_transport(path,game,ids):
    data=json.loads(path.read_text());assert data.get('complete')and not data.get('error')and not data['cleanupErrors']and not data['relayErrors']
    assert data['runnerReturnCode']==0
    bots=[r for r in data['connections']if 'name'in r];assert sorted((r['name'],json.loads(r['handshakeAfter'])['version'])for r in bots)==sorted(ids)
    for r in data['connections']:
        assert not r.get('error')
        assert r.get('upInputCount',0)==r.get('upOutputCount',0)
        for a,b in [('downInput','downOutput'),('upAfterHandshakeInput','upAfterHandshakeOutput')]:
            assert r[a]==r[b]and r.get(a+'Count',0)==r.get(b+'Count',0)
        if game['mode']=='random'or'name'not in r:assert r['upInput']==r['upOutput']and r.get('upInputCount',0)==r.get('upOutputCount',0)and r['rewrites']==0
    for r in bots:
        before=json.loads(r['handshakeBefore']);after=json.loads(r['handshakeAfter'])
        assert before['sessionId']==after['sessionId']==r['serverSessionId']
        if game['mode']=='fixed':
            assert r['rewrites']==1
            assert after.pop('initialPosition')==dict(zip(('x','y','direction'),positions(game)[r['name']]))
            assert {k:v for k,v in before.items()if k!='initialPosition'}==after
        else:
            p=before.get('initialPosition');assert p is None or all(v is None for v in p.values())
    if game['mode']=='fixed':assert [r['name']for r in sorted(bots,key=lambda r:r['serverSessionId'])]==sorted(n for n,_ in ids)
    return dict(connections=len(data['connections']),botHandshakes=len(bots),handshakeSHA256=core.sha(path))

def starts(recording,game):
    names={};rounds={};trajectories=collections.defaultdict(list)
    with gzip.open(recording,'rt')as stream:
        for line in stream:
            e=json.loads(line)
            if e['type']=='GameStartedEventForObserver':
                names={b['id']:b['name']for b in e['participants']}
                if game['mode']=='fixed':assert [names[k]for k in sorted(names)]==sorted(game['order'])
            elif e['type']=='TickEventForObserver':
                r,t=e['roundNumber'],e['turnNumber'];state={names[b['id']]:{k:b[k]for k in ('x','y','direction','gunDirection','radarDirection','speed','energy','gunHeat','enemyCount')}for b in e['botStates']}
                if t==1:
                    rounds[r]={n:[s['x'],s['y'],s['direction']]for n,s in state.items()}
                    if game['mode']=='fixed':
                        expected=positions(game);assert set(state)==set(expected)
                        for n,p in expected.items():assert all(abs(a-b)<1e-7 for a,b in zip(rounds[r][n],p)),(r,n,rounds[r][n],p)
                trajectories[r].append(hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    assert sorted(rounds)==list(range(1,game['rounds']+1))
    return dict(ids=names,initialPoses=rounds,trajectories=dict(trajectories))

def summarize(games):
    means={};pairs=[]
    for mode in ['random','fixed']:
        for scenario in ['melee','Diamond Port']:
            means[mode,scenario]={}
            for arm in ['A','B']:
                selected=[g for g in games if g['mode']==mode and g['scenario']==scenario and g['arm']==arm];assert len(selected)==4
                values=[Fraction(g['helm']['totalScore'],sum(b['totalScore']for b in g['official']['results']))for g in selected]
                means[mode,scenario][arm]=sum(values,Fraction())/4
    for index in range(16):
        a=next(g for g in games if g['pair']==index and g['arm']=='A');b=next(g for g in games if g['pair']==index and g['arm']=='B')
        delta=Fraction(a['helm']['totalScore'],sum(r['totalScore']for r in a['official']['results']))-Fraction(b['helm']['totalScore'],sum(r['totalScore']for r in b['official']['results']))
        divergence={}
        for r in a['startAudit']['trajectories']:
            x=a['startAudit']['trajectories'][r];y=b['startAudit']['trajectories'][r]
            first=next((i+1 for i,(u,v)in enumerate(zip(x,y))if u!=v),None)
            if first is None and len(x)!=len(y):first=min(len(x),len(y))+1
            divergence[r]=first
        pairs.append(dict(pair=index,mode=a['mode'],scenario=a['scenario'],shareDeltaPP=float(100*delta),absoluteDeltaPP=float(100*abs(delta)),shareDeltaPPExact=str(100*delta),scoreDelta=a['helm']['totalScore']-b['helm']['totalScore'],winDelta=a['helm']['firstPlaces']-b['helm']['firstPlaces'],firstStateDivergence=divergence))
    output=[]
    for (mode,scenario),arms in means.items():
        d=arms['A']-arms['B'];limit=Fraction(2 if scenario=='melee'else 5,100)
        output.append(dict(mode=mode,scenario=scenario,A=float(arms['A']),B=float(arms['B']),deltaPP=float(100*d),deltaPPExact=str(100*d),withinAllowance=abs(d)<=limit))
    gates={r['scenario']:r['withinAllowance']for r in output if r['mode']=='fixed'}
    return dict(means=output,pairs=pairs,gates=gates,decision='continue-to-new-movement-protocol'if all(gates.values())else'stop-measurement-not-ready')

def prepare(a):
    out=a.evidence.resolve();out.mkdir(parents=True,exist_ok=False);project=ROOT/'experiments/KeelHelm';assert core.sha(project/'KeelHelm.jar')==BASE
    shutil.copytree(project,out/'control-source',ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
    # Extract the exact release's server, never rebuild or patch it.
    import zipfile
    z=zipfile.ZipFile(a.source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar');server=z.read('robocode-tankroyale-server.jar');(out/'server.jar').write_bytes(server)
    extras=[Path(__file__),ROOT/'tournament/tools/paired_transport.py',ROOT/'tournament/tools/test_paired_transport.py',ROOT/'tournament/tools/test_keelhelm_calibration.py',ROOT/'tournament/scripts/PairedBattle.java',ROOT/'docs/improvements/keelhelm-aa-protocol.md',out/'server.jar']
    f=core.collect(a.source.resolve(),out,a.pins.resolve(),a.java.absolute(),a.python.absolute(),control_jar_sha=BASE,schedule_rows=schedule(),extra_paths=extras,candidate_project=project)
    f['serverJar']=str(out/'server.jar');f['benchmark']=dict(name='aa-calibration-v1',games=32,rounds=240,concurrency=1,jarSHA256=BASE);core.save(out/'freeze.json',f)
    for g in f['schedule']:
        run=out/'preflight'/g['id'];run.mkdir(parents=True);dirs,ids=stage(g,f,out,run);assert len(set(dirs))==len(ids)==len(g['order'])
    print('PREPARED',core.sha(out/'freeze.json'),flush=True)

def play(g,f,out,monitor):
    row=dict(**g,status='setup-failed');run=out/'games'/g['id']
    try:
        run.mkdir(parents=True);dirs,ids=stage(g,f,out,run);(run/'home').mkdir();(run/'recordings').mkdir();java=Path(f['java'])
        env={k:os.environ[k]for k in ('TMPDIR','LANG','LC_ALL','TZ')if k in os.environ}
        env.update(HOME=str(run/'home'),JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+':/usr/bin:/bin:/opt/homebrew/bin',ROBOCODE_PYTHON_BIN=f['python'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',ROBOCODE_ENV_FILE=str(run/'absent.env'),ROBOCODE_GUN_ENV_FILE=str(run/'absent-guns.env'),RUMBLE_JAVA_CACHE=str(Path(f['source'])/'.benchmark-cache/retcon-api'),KEEL_HELM_METRICS=str(run/'helm-metrics.txt'),KEEL_HELM_SEED=str(g['seed']),KEEL_SEED=str(g['seed']),PRISM_SEED=str(g['seed']))
        config=dict(run=str(run),java=f['java'],serverJar=f['serverJar'],runner=str(Path(f['source'])/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),adapter=str(ROOT/'tournament/scripts/PairedBattle.java'),rounds=g['rounds'],bots=dirs,identities=dict(ids),positions=positions(g));core.save(run/'config.json',config)
        cmd=[f['python'],str(ROOT/'tournament/tools/paired_transport.py'),str(run/'config.json')];core.save(run/'invocation.json',dict(command=cmd,environment=env));row['status']='attempted';print('START',g['id'],g['scenario'],flush=True)
        def check():
            if monitor.error:raise ValueError(monitor.error)
        core.execute(cmd,env,run,monitor.register,failure_check=check)
        official=json.loads((run/'battle.json').read_text());assert official['numberOfRounds']==g['rounds']and sorted((b['name'],b['version'])for b in official['results'])==sorted(ids)
        raw=list((run/'recordings').glob('*.gz'));assert len(raw)==1
        audit=core.audit(raw[0],official,g,ids);assert audit['metrics']['skippedTurns']==0
        telemetry=core.telemetry(run/'helm-metrics.txt',g['rounds']);transport=inspect_transport(run/'transport.json',g,ids);start=starts(raw[0],g)
        core.save(run/'starts.json',start);own=next(b for b in official['results']if b['name']=='KeelHelm')
        row.update(status='complete',official=official,helm=own,audit=audit,telemetry=telemetry,transportAudit=transport,startAudit=start,recording=str(raw[0].relative_to(out)))
    except BaseException as error:row['error']=repr(error)
    if run.exists():core.save(run/'result.json',row)
    print('END',g['id'],row['status'],row.get('error'),flush=True);return row

def run(a):
    out=a.evidence.resolve();assert a.approval_sha==core.sha(out/'freeze.json');f=json.loads((out/'freeze.json').read_text());assert f['schedule']==schedule()and f['source']==str(a.source.resolve())and f['evidence']==str(out);assert not(out/'results.json').exists()
    report=dict(status='running',freezeSHA256=a.approval_sha,plannedGames=32,plannedRounds=240,games=[],waves=[]);core.save(out/'results.json',report)
    try:
        for g in f['schedule']:
            core.verify_inputs(f);ad=resources.read_resources();ad['diskFreeBytes']=shutil.disk_usage(out).free;core.save(out/(g['id']+'-admission.json'),ad);assert ad['availablePercent']>=35 and ad['diskFreeBytes']>=2*1024**3
            monitor=resources.Monitor(out/(g['id']+'-resources.jsonl'));monitor.start()
            try:row=play(g,f,out,monitor)
            finally:monitor.close()
            report['games'].append(row);report['waves'].append(dict(game=g['id'],samples=monitor.samples,resourceError=monitor.error));core.save(out/'results.json',report)
            if row['status']!='complete'or monitor.error:raise ValueError(monitor.error or row.get('error'))
        report.update(status='complete',completedGames=32,completedRounds=240,summary=summarize(report['games']))
    except BaseException as error:report.update(status='invalid',error=repr(error));raise
    finally:core.save(out/'results.json',report)
    print(json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--approval-sha')
    for n in ['pins','java','python']:p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:run(a)
