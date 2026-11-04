"""One registered descriptive Keel–Prism comparison. No retries; stop first invalid."""
import gzip,hashlib,json,os,platform,shutil,statistics,subprocess,sys
from pathlib import Path
import benchmark
ROOT=benchmark.ROOT
REG=ROOT/'benchmarks/keel-prism-v1'
OUT=ROOT/'build/keel-prism-remaining'
JAVA=Path(r'C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot')
RUNNER=ROOT/'build/completion-test-cache/robocode-tankroyale-runner-1.4.0.jar'
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(path,game):
    recording,=path.glob('*.battle.gz'); endpoints=[]; totals={n:dict(shots=0,hits=0,shotEnergy=0.,damage=0.,minEnergy=None) for n in ('Keel','Prism')}; per={}; ids={}; ticks={}
    with gzip.open(recording,'rt',encoding='utf-8') as f:
      for line in f:
        e=json.loads(line); kind=e['type']
        if kind=='GameStartedEventForObserver':
          s=e['gameSetup']; assert s['arenaWidth']==800 and s['arenaHeight']==600 and s['turnTimeout']==100000 and s['numberOfRounds']==10
          ids={x['id']:x['name'] for x in e['participants']}; assert set(ids.values())=={'Keel','Prism'}
        elif kind=='TickEventForObserver':
          r=e['roundNumber'];ticks.setdefault(r,[]).append(e['turnNumber']); cells=per.setdefault(r,{n:dict(shots=0,hits=0,shotEnergy=0.,damage=0.,lastObservedEnergy=None) for n in totals})
          for b in e['botStates']:
            n=ids[b['id']];cells[n]['lastObservedEnergy']=b['energy'];old=totals[n]['minEnergy'];totals[n]['minEnergy']=b['energy'] if old is None else min(old,b['energy'])
          for event in e['events']:
            if event['type']=='BulletFiredEvent':
              n=ids[event['bullet']['ownerId']]
              for c in (totals[n],cells[n]):c['shots']+=1;c['shotEnergy']+=event['bullet']['power']
            elif event['type']=='BulletHitBotEvent':
              n=ids[event['bullet']['ownerId']]
              for c in (totals[n],cells[n]):c['hits']+=1;c['damage']+=event['damage']
        elif kind=='RoundEndedEventForObserver':endpoints.append(e)
        elif kind=='GameEndedEventForObserver':end=e
    assert end['numberOfRounds']==10 and [e['roundNumber'] for e in endpoints]==list(range(1,11))
    previous={};rounds=[]
    for e in endpoints:
      r=e['roundNumber']; seq=ticks[r];assert seq==list(range(1,e['turnNumber']+1))
      for x in e['results']:
        n=x['name'];old=previous.get(n,{});per[r][n].update(score=x['totalScore']-old.get('totalScore',0),firstPlaces=x['firstPlaces']-old.get('firstPlaces',0),bulletDamage=x['bulletDamage']-old.get('bulletDamage',0));previous[n]=x
      rounds.append(dict(round=r,turns=len(seq),bots=per[r]))
    markers=(path/'markers.txt').read_text().splitlines()
    for tag in ('start','end'):assert [int(x.split()[0]) for x in markers if x.endswith(' '+tag)]==list(range(1,11))
    tel=[json.loads(x) for x in (path/'prism.jsonl').read_text().splitlines()];finals=[x for x in tel if x['event']=='game'];assert len(finals)==1 and finals[0]['counters']['rounds']==10 and all(x['seed']==game['seed'] for x in tel)
    totals['Keel']['privateSkips']=sum(' skip ' in x for x in markers);totals['Prism']['privateSkips']=finals[0]['counters'].get('turns.skipped',0)
    results=end['results'];assert {(x['name'],x['version']) for x in results}=={('Keel','0.1.0'),('Prism','0.1.0')}
    for x in results:totals[x['name']].update(score=x['totalScore'],firstPlaces=x['firstPlaces'],bulletDamage=x['bulletDamage'])
    return dict(recordingSHA256=digest(recording),results=results,bots=totals,perRound=rounds,prismRoundSnapshots=sum(x['event']=='round' for x in tel))
def main():
    plan=json.loads((REG/'plan.json').read_bytes())
    assert [g['id'] for g in plan['games']]==[f'kp-utf8-{i:02}' for i in range(2,7)] and all(g['rounds']==10 and g['seed']==2026270200+i and g['first']==('Keel' if i%2 else 'Prism') for i,g in enumerate(plan['games'],2))
    assert digest(RUNNER)==plan['runnerSHA256']
    for n,h in plan['jars'].items():assert digest(ROOT/f'experiments/{n}/{n}.jar')==h
    env={k:v for k,v in os.environ.items() if not k.startswith(('AISURVIVOR_','KEEL_','PRISM_','ROBOCODE_'))};env['JAVA_TOOL_OPTIONS']='-Dfile.encoding=UTF-8';env.pop('_JAVA_OPTIONS',None);env['JAVA_HOME']=str(JAVA);env['PATH']=str(JAVA/'bin')+os.pathsep+env['PATH']
    profile=dict(jvmOptions=env['JAVA_TOOL_OPTIONS'],javaVersion=subprocess.run([str(JAVA/'bin/java.exe'),'-version'],capture_output=True,text=True).stderr,javaFiles={x:digest(JAVA/x) for x in ('bin/java.exe','lib/modules','release')},assets={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),REG/'KeelPrismBattle.java']},os=platform.platform(),python=sys.version,runnerSHA256=digest(RUNNER),sourceCommit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),planSHA256=digest(REG/'plan.json'))
    if sys.argv[1]=='profile':
      save(REG/'runtime.json',profile);print(json.dumps(profile,indent=2));return
    frozen=json.loads((REG/'runtime.json').read_bytes());assert all(profile[k]==frozen[k] for k in ('jvmOptions','javaVersion','javaFiles','assets','os','python','runnerSHA256','planSHA256'))
    OUT.mkdir(parents=True,exist_ok=False);save(OUT/'runtime.json',profile)
    ref='refs/tags/keel-prism-remaining50-v1-77b8a5c3114641c1-8c045f80d37a3076'
    subprocess.run(['gh','api','-X','POST','repos/stephanelee/robocode/git/refs','-f','ref='+ref,'-f','sha='+profile['sourceCommit']],check=True,capture_output=True,text=True)
    save(OUT/'receipt.json',dict(ref=ref,head=profile['sourceCommit'],profileSHA256=digest(REG/'runtime.json')))
    report=dict(plan=plan,execution=profile['sourceCommit'],games=[],decision='descriptive only')
    for game in plan['games']:
      data=OUT/game['id'];data.mkdir();stage=ROOT/'build/keel-prism-staging'/game['id'];stage.mkdir(parents=True,exist_ok=False)
      for n in plan['jars']:
        shutil.copytree(ROOT/f'experiments/{n}/bot',stage/n);shutil.copy2(ROOT/f'experiments/{n}/{n}.jar',stage/n/f'{n}.jar');assert digest(stage/n/f'{n}.jar')==plan['jars'][n]
      local=env|dict(KEEL_METRICS=str(data/'markers.txt'),KEEL_SEED=str(game['seed']),PRISM_TELEMETRY_PATH=str(data/'prism.jsonl'),PRISM_SEED=str(game['seed']))
      other='Prism' if game['first']=='Keel' else 'Keel';cmd=[str(JAVA/'bin/java.exe'),'-Dfile.encoding=UTF-8','-cp',str(RUNNER),str(REG/'KeelPrismBattle.java'),'10',str(data),str(stage/game['first']),str(stage/other)]
      save(data/'attempt.json',dict(game=game,command=cmd,execution=profile['sourceCommit'],jars=plan['jars'],privatePaths={k:v for k,v in local.items() if k.startswith(('KEEL_','PRISM_'))}))
      try:benchmark.run_process(cmd,stage,local,data/'runner.log',3100);row=dict(**game,status='valid',**analyze(data,game))
      except Exception as exc:
        report['games'].append(dict(**game,status='invalid',error=repr(exc)));report['decision']='invalid; stopped without replacement';save(OUT/'report.json',report);raise
      report['games'].append(row);save(OUT/'report.json',report);print(game['id'],row['bots'],flush=True)
    report['meanKeelShare']=statistics.mean(x['bots']['Keel']['score']/sum(b['score'] for b in x['bots'].values()) for x in report['games']);save(OUT/'report.json',report)
if __name__=='__main__':main()