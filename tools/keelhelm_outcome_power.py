"""Issue131: frozen confirmed-outcome power comparison, 40 cold serial games / 520 rounds."""
import argparse,gzip,json,os,re,shutil
from fractions import Fraction
from pathlib import Path
import keelhelm_melee as core
import keelhelm_resources as resources
import keelhelm_followups as legacy

BASE='19f779d4c4df97e9510f22bc561c0b19af0c07eaa1f1f3fd1d93f70992f7e55b'

def schedule():
    pairs=[];names=['KeelHelm','Diamond Port','Keel','Prism']
    for slot in range(8):
        i=slot%4;pairs.append(('melee',20,names[i:]+names[:i]))
    for name in ('Diamond Port','Retcon'):
        for slot in range(4):pairs.append((name,10,['KeelHelm',name] if slot%2==0 else [name,'KeelHelm']))
    for name in ('Walls','SpinBot'):
        for slot in range(2):pairs.append((name,5,['KeelHelm',name] if slot%2==0 else [name,'KeelHelm']))
    games=[]
    for i,(scenario,rounds,order) in enumerate(pairs):
        for arm in (['control','candidate'] if i%2==0 else ['candidate','control']):
            games.append(dict(id=f'{i+1:02d}-{scenario.lower().replace(" ","-")}-{arm}',arm=arm,scenario=scenario,rounds=rounds,seed=2026112300+i,order=order))
    assert len(games)==40 and sum(g['rounds'] for g in games)==520
    return games

def norm(angle):return (angle+180)%360-180

def exposure(recording,telemetry,rounds):
    import collections,math
    identities={};physical={};outcomes=collections.defaultdict(list);states={};own=None
    with gzip.open(recording,'rt') as stream:
        for line in stream:
            e=json.loads(line)
            if e['type']=='GameStartedEventForObserver':
                identities={p['id']:p['name']for p in e['participants']};own=next(i for i,n in identities.items()if n=='KeelHelm')
            elif e['type']=='TickEventForObserver':
                r,t=e['roundNumber'],e['turnNumber']
                for bot in e['botStates']:states[r,t,bot['id']]=bot
                for event in e['events']:
                    kind=event['type']
                    if kind=='BulletFiredEvent' and event['bullet']['ownerId']==own:
                        bullet=event['bullet'];key=(r,t,bullet['bulletId']);assert key not in physical;physical[key]=bullet
                    if kind in ('BulletHitBotEvent','BulletHitWallEvent','BulletHitBulletEvent'):
                        seen=set()
                        for label in ('bullet','hitBullet'):
                            bullet=event.get(label)
                            if bullet and bullet['ownerId']==own and bullet['bulletId']not in seen:
                                seen.add(bullet['bulletId']);outcomes[r,bullet['bulletId']].append(dict(turn=t,mask={'BulletHitBotEvent':1,'BulletHitWallEvent':2,'BulletHitBulletEvent':4}[kind],damage=event.get('damage',0)if kind=='BulletHitBotEvent'else 0))
    windows={};snapshots={0:{}};sequence=0;named={};total=0;allResolved=0
    def baseline(distance,energy,enemy):
        p=2.2 if distance<250 else 1.6 if distance<500 else 1
        if energy<20:p=min(p,.5+energy/20)
        p=min(p,enemy/4+.05 if enemy<=4 else (enemy+2)/6+.05)
        return max(.1,min(3,p))
    def windowstate():return {key:(len(values),sum(values))for key,values in windows.items()}
    for r in range(1,rounds+1):
        lines=[s for s in telemetry if s.startswith(f'{r} powertrace ')];assert len(lines)==1
        stat=next(s for s in telemetry if s.startswith(f'{r} stats '));assert re.search(r'\bpowerTraceOverflow=0(?:\s|$)',stat)
        def integer(key):return int(re.search(r'\b'+key+r'=(\d+)',stat)[1])
        ledger={};changed={};confirmed=resolved=successful=evicted=0
        text=lines[0].split(' ',2)[2]
        for record in ([]if text=='none'else text.split(';')):
            p=record.split(',');seq=int(p[0]);assert seq==sequence+1;sequence=seq
            if p[1]=='F':
                assert len(p)==17
                context,request,turn,bid,target,bin=map(int,p[2:8]);actual,base,selected,distance,energy,enemy=map(float,p[8:14]);age,n,h=map(int,p[14:17]);key=(target,bin)
                assert all(math.isfinite(v)for v in (actual,base,selected,distance,energy,enemy))
                assert target in identities and target!=own and bin==(0 if distance<250 else 1 if distance<500 else 2)
                assert (n,h)==snapshots[context].get(key,(0,0)) and request==turn-1 and 0<=age<=12
                bullet=physical[r,turn,bid];assert abs(bullet['power']-actual)<1e-6 and abs(actual-selected)<1e-6
                state=states[r,request,own];assert state['enemyCount']>1 and abs(state['energy']-energy)<1e-5
                assert abs(base-baseline(distance,energy,enemy))<1e-6
                proposed=base
                if distance<500 and 0<=age<=4 and energy>=45 and enemy>=8 and n==32 and 10*(h+2)>=3*(n+8):
                    raised=min(2.6,base+.4,enemy/4+.05 if enemy<=4 else (enemy+2)/6+.05)
                    if raised-base>=.2:proposed=raised
                assert abs(selected-proposed)<1e-6
                assert bid not in ledger;ledger[bid]=(target,bin);confirmed+=1
                if actual-base>.1:changed[target]=changed.get(target,0)+1
            elif p[1]=='R':
                assert len(p)==11
                turn,bid,target,bin,callback,mask=map(int,p[2:8]);damage=float(p[8]);hit,end=map(int,p[9:11]);assert ledger.pop(bid)==(target,bin)
                assert math.isfinite(damage)and damage>=0 and hit==int(damage>0)and end in (0,1)
                available=[e for e in outcomes[r,bid]if e['turn']==callback];availableMask=0
                for e in available:availableMask|=e['mask']
                assert mask>0 and mask&~availableMask==0 and damage<=sum(e['damage']for e in available)+1e-6
                assert (end==0 and callback<turn)or(end==1 and hit==1 and callback<=turn)
                values=windows.setdefault((target,bin),collections.deque(maxlen=32));values.append(hit)
                resolved+=1;successful+=hit;allResolved+=1
            elif p[1]=='E':
                assert len(p)==6
                turn,bid,target,bin=map(int,p[2:6]);assert ledger.pop(bid)==(target,bin);evicted+=1
            else:raise ValueError('Unknown outcome trace type')
            snapshots[sequence]=windowstate()
        assert confirmed==integer('outcomeConfirmed')and resolved==integer('outcomeResolved')and successful==integer('outcomeSuccesses')
        assert evicted+len(ledger)==integer('outcomeCensored')and confirmed==resolved+evicted+len(ledger)
        declared=re.search(r'\bpowerTargets=([^ ]+)',stat)[1]
        expected={}if declared=='none'else{int(k):int(v)for k,v in(x.split(':')for x in declared.split(','))}
        assert expected==changed and sum(changed.values())==integer('powerConfirmed')
        for target,count in changed.items():named[identities[target]]=named.get(identities[target],0)+count;total+=count
    return dict(confirmedChanged=total,namedTargets=named,resolvedFeedback=allResolved)

def summary(games):
    report=legacy.summary(games)
    survival={}
    for arm in ('control','candidate'):
        values=[Fraction(g['helm']['survival'],sum(b['survival']for b in g['official']['results']))for g in games if g['scenario']=='melee'and g['arm']==arm]
        survival[arm]=sum(values,Fraction())/len(values)
    total=0;names={}
    for g in games:
        if g['arm']=='candidate'and g['scenario']=='melee':
            total+=g['exposure']['confirmedChanged']
            for name,n in g['exposure']['namedTargets'].items():names[name]=names.get(name,0)+n
    report['survival']={a:dict(meanShare=float(v),meanShareExact=str(v))for a,v in survival.items()}
    report['exposure']=dict(confirmedChanged=total,namedTargets=names);report['policyChangeExposure']=total
    report['gates'].update(survival=survival['candidate']-survival['control']>=-Fraction(3,100),confirmedChanged=total>=100,multipleNamedTargets=len([n for n,v in names.items()if v>0])>=2)
    report['decision']='accept'if all(report['gates'].values())else'reject'
    return report

def prepare(a):
    out=a.evidence.resolve();out.mkdir(parents=True,exist_ok=False)
    candidate=core.ROOT/'benchmarks/keelhelm-outcome-power/candidate-project';control=core.ROOT/'experiments/KeelHelm';assert core.sha(control/'KeelHelm.jar')==BASE
    for label,src in [('control-source',control),('candidate-source',candidate)]:shutil.copytree(src,out/label,ignore=shutil.ignore_patterns('build','.gradle','__pycache__'))
    extras=[Path(__file__),Path(legacy.__file__),core.ROOT/'tools/keelhelm_resources.py',core.ROOT/'tools/test_keelhelm_outcome_power.py',core.ROOT/'docs/improvements/keelhelm-outcome-power-protocol.md']
    f=core.collect(a.source.resolve(),out,a.pins.resolve(),a.java.absolute(),a.python.absolute(),control_jar_sha=BASE,schedule_rows=schedule(),extra_paths=extras,candidate_project=candidate)
    f['benchmark']=dict(name='confirmed-outcome-power-v1',games=40,rounds=520,concurrency=1,controlJarSHA256=BASE,candidateJarSHA256=core.sha(candidate/'KeelHelm.jar'))
    core.save(out/'freeze.json',f)
    for g in f['schedule']:
        run=out/'preflight'/g['id'];run.mkdir(parents=True);dirs,ids=core.stage_bots(g,a.source.resolve(),out,f,run);assert len(set(dirs))==len(ids)==len(g['order'])
    print('PREPARED',core.sha(out/'freeze.json'),flush=True)

def play(g,f,out,monitor):
    row=dict(**g,status='setup-failed');run=out/'games'/g['id']
    try:
        run.mkdir(parents=True);source=Path(f['source']);java=Path(f['java'])
        dirs,ids=core.stage_bots(g,source,out,f,run);(run/'home').mkdir();(run/'recordings').mkdir()
        env={k:os.environ[k]for k in ('TMPDIR','LANG','LC_ALL','TZ')if k in os.environ}
        env.update(HOME=str(run/'home'),JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+':/usr/bin:/bin:/opt/homebrew/bin',ROBOCODE_PYTHON_BIN=f['python'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',ROBOCODE_ENV_FILE=str(run/'absent.env'),ROBOCODE_GUN_ENV_FILE=str(run/'absent-guns.env'),RUMBLE_JAVA_CACHE=str(source/'.benchmark-cache/retcon-api'),KEEL_HELM_METRICS=str(run/'helm-metrics.txt'),KEEL_HELM_MELEE_FEATURES='0',KEEL_HELM_OUTCOME_POWER='true'if g['arm']=='candidate'else'false',KEEL_HELM_SEED=str(g['seed']),KEEL_SEED=str(g['seed']),PRISM_SEED=str(g['seed']))
        cmd=[f['java'],'-Dtournament.recordingDir='+str(run/'recordings'),'-cp',str(source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),str(core.ROOT/'tournament/scripts/TournamentBattle.java'),str(g['rounds']),'900',str(run/'battle.json'),*dirs]
        core.save(run/'invocation.json',dict(command=cmd,environment=env));row['status']='attempted';print('START',g['id'],flush=True)
        def check():
            if monitor.error:raise ValueError(monitor.error)
        core.execute(cmd,env,run,monitor.register,failure_check=check)
        official=json.loads((run/'battle.json').read_text());assert official['numberOfRounds']==g['rounds']
        assert sorted((b['name'],b['version'])for b in official['results'])==sorted(ids)
        recordings=list((run/'recordings').glob('*.gz'));assert len(recordings)==1
        audit=core.audit(recordings[0],official,g,ids);assert audit['metrics']['skippedTurns']==0
        telemetry=core.telemetry(run/'helm-metrics.txt',g['rounds'])
        action=exposure(recordings[0],telemetry,g['rounds'])if g['arm']=='candidate'else dict(confirmedChanged=0,namedTargets={})
        own=next(b for b in official['results']if b['name']=='KeelHelm')
        row.update(status='complete',official=official,helm=own,audit=audit,telemetry=telemetry,exposure=action,recording=str(recordings[0].relative_to(out)))
    except BaseException as e:row['error']=repr(e)
    if run.exists():core.save(run/'result.json',row)
    print('END',g['id'],row['status'],row.get('error'),flush=True);return row

def run(a):
    out=a.evidence.resolve();assert a.approval_sha==core.sha(out/'freeze.json')
    f=json.loads((out/'freeze.json').read_text());assert f['source']==str(a.source.resolve())and f['evidence']==str(out)
    assert f['schedule']==schedule()and f['benchmark']['concurrency']==1
    assert not(out/'results.json').exists(),'No resume/replacement'
    report=dict(status='running',freezeSHA256=a.approval_sha,plannedGames=40,plannedRounds=520,games=[],waves=[]);core.save(out/'results.json',report)
    try:
        for g in f['schedule']:
            core.verify_inputs(f);ad=resources.read_resources();ad['diskFreeBytes']=shutil.disk_usage(out).free;core.save(out/(g['id']+'-admission.json'),ad)
            if ad['availablePercent']<35 or ad['diskFreeBytes']<2*1024**3:raise ValueError('Admission failed; no game launched')
            monitor=resources.Monitor(out/(g['id']+'-resources.jsonl'));monitor.start()
            try:row=play(g,f,out,monitor)
            finally:monitor.close()
            report['games'].append(row);report['waves'].append(dict(game=g['id'],samples=monitor.samples,resourceError=monitor.error));core.save(out/'results.json',report)
            if row['status']!='complete'or monitor.error:raise ValueError(monitor.error or row.get('error'))
        report.update(status='complete',completedGames=40,completedRounds=520,summary=summary(report['games']))
    except BaseException as e:report.update(status='invalid',error=repr(e));raise
    finally:core.save(out/'results.json',report)
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--approval-sha')
    for n in ['pins','java','python']:p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:run(a)
