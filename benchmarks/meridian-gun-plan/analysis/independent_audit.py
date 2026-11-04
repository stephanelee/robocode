"""Independent observer/result audit; no repository reducers imported or battles run."""
import pathlib,json,gzip,hashlib,math,statistics,struct,subprocess,datetime,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[3])
parser.add_argument('--artifact-dir',type=pathlib.Path,help='Directory containing original artifact.zip and extracted/')
parser.add_argument('--output',type=pathlib.Path,help='Optionally save independently recomputed game/per-round JSON')
args=parser.parse_args()
ROOT=args.repository.resolve()
ART=(args.artifact_dir or ROOT/'build/acquisition-41766660321').resolve(); EX=ART/'extracted/local-comparison/execution'
REG=ROOT/'benchmarks/meridian-gun-plan'; SMALL=ART/'extracted/build/meridian'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def equal(a,b,at=''):
 if isinstance(a,dict):
  assert isinstance(b,dict),(at,type(b))
  for k,v in a.items():equal(v,b[k],at+'.'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),(at,len(a),len(b))
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,at+str(i))
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),(at,a,b)
 else:assert a==b,(at,a,b)
assert sha(ART/'artifact.zip')=='0e63d31532e2cd03657d14352f69c80b111a148ae3543457b4542a66e070facd'
profile=read(EX/'profile.json');allowed=read(ROOT/'benchmarks/meridian-aim/profile-20261026.json')
allowed['assets']['experiments/Meridian/src/Meridian.java']='5268264c49dd1150eb862b4cd59814a9e589abb3e5f284c56bf07dd07989d0d1'
allowed['assets']['experiments/Meridian/Meridian.jar']='7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe'
assert profile==allowed
assert (EX/'package-contents.json').read_bytes()==(REG/'package-contents.json').read_bytes()
host=read(EX/'host.json');receipt=read(EX/'activation.json')
assert host==receipt['hostIdentity'] and sha(SMALL/'gun-plan-host.json')==receipt['hostDocumentSHA256']
assert host==read(SMALL/'gun-plan-host.json')
assert host['runID']=='41766660321' and host['runAttempt']=='1'
assert host['runtimeProfileSHA256']==sha(EX/'profile.json')
assert host['packageContentsSHA256']==sha(EX/'package-contents.json')
assert host['toolingManifestSHA256']=='529acb1d84ce1f0634eed4a4608244bad47a8b070f944617a404a757c24eab52'
assert receipt['payloadSHA']=='04fc2c87b6f961846e755a98e93213d320af9374'
assert receipt['receiptRef']=='refs/tags/experiment-receipts/meridian-gp60-issue90'
plan=read(REG/'games.json');decision=read(EX/'decision.json')
assert len(plan)==12 and len(decision['games'])==12 and decision['status']=='complete'
assert sorted(p.name for p in (EX/'raw').iterdir())==sorted(g['id'] for g in plan)
rows=[]; setup=dict(gameType='1v1',arenaWidth=800,arenaHeight=600,minNumberOfParticipants=2,maxNumberOfParticipants=2,numberOfRounds=5,gunCoolingRate=.1,maxInactivityTurns=450,turnTimeout=100000,readyTimeout=10000000,defaultTurnsPerSecond=-1)
countfields=['shots','hits','enemyShots','hitsTaken','damageDealt','damageTaken','shotEnergy','wallContacts','botContacts','commonOpportunities','noFireOpportunities']
for game,expected in zip(plan,decision['games']):
 data=EX/'raw'/game['id'];stage=EX/'stages'/game['id'];attempt=read(data/'attempt.json')
 assert attempt['game']==game and attempt['execution']==receipt['payloadSHA']
 assert attempt['runtimeProfileSHA256']==host['runtimeProfileSHA256']
 assert attempt['command'][4]==str(game['port']) and attempt['command'][5]=='5'
 assert attempt['environment']['MERIDIAN_SEED']==str(game['seed'])
 assert attempt['environment']['HOME'].endswith('/'+game['id']+'/home')
 for rel,h in attempt['stagedSHA256'].items(): assert sha(stage/rel)==h,(game['id'],rel)
 target='7667ef48974e67fe1e42ecee15005905bfe915c4c4afe7cba8e52c146ec2cafe' if game['arm']=='candidate' else '5875565a13b5759940ccb080c0c7f237caec0bae525a68e68643a35890a1a8a9'
 assert sha(stage/'Meridian/Meridian.jar')==target
 if game['enemy']=='Diamond Port':
  lock=read(ROOT/'tools/opponents/strong-melee.lock.json')
  required=[r for r in lock['files'] if r['path']=='LICENSE' or r['path'].startswith(('bots/bot_core/','bots/ports/diamond-port/'))]
  assert len(required)==104
  for item in required:
   path=item['path'];rel='opponents/LICENSE.shared-Apache-2.0' if path=='LICENSE' else 'opponents/'+path.removeprefix('bots/ports/').removeprefix('bots/')
   assert sha(stage/rel)==item['sha256'],rel
 elif game['enemy']=='Retcon':
  for item in read(ROOT/'tools/opponents/retcon.lock.json')['files']:
   raw=(stage/'Retcon'/item['path']).read_bytes(); assert len(raw)==item['size']
   assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['gitBlobSHA']
 else:assert sha(stage/'AISurvivor/AISurvivor.jar')==profile['assets']['AISurvivor/AISurvivor.jar']
 files=list(data.glob('*.battle.gz')); assert len(files)==1
 with gzip.open(files[0],'rt') as f: events=[json.loads(l) for l in f]
 json.dumps(events,allow_nan=False)
 assert events[0]['type']=='GameStartedEventForObserver' and events[-1]['type']=='GameEndedEventForObserver'
 for k,v in setup.items(): assert events[0]['gameSetup'][k]==v,k
 final=events[-1]; assert final['numberOfRounds']==5
 official=final['results']; assert sorted((r['name'],r['version']) for r in official)==sorted([('Meridian','0.3.0'),(game['enemy'],game['version'])])
 own=next(r for r in official if r['name']=='Meridian');uid=own['id']
 marker=(data/'markers.txt').read_text().splitlines();assert not any('skip' in x for x in marker)
 for tag in ['start','end']:assert [int(x.split()[0]) for x in marker if x.endswith(' '+tag)]==[1,2,3,4,5]
 assert not any('Skipped' in e['type'] for e in events)
 ends=[e for e in events if e['type']=='RoundEndedEventForObserver'];assert [e['roundNumber'] for e in ends]==[1,2,3,4,5]
 roundrows=[];prev={r['id']:(0,0) for r in official}
 for r in range(1,6):
  ticks=[e for e in events if e['type']=='TickEventForObserver' and e['roundNumber']==r]
  assert [e['turnNumber'] for e in ticks]==list(range(1,ends[r-1]['turnNumber']+1))
  rr={k:0 for k in countfields};rr.update(round=r,turns=len(ticks));longest=streak=0
  for i,t in enumerate(ticks):
   for e in t['events']:
    k=e['type'];assert 'Skipped' not in k;b=e.get('bullet',{})
    if k=='BulletFiredEvent':
     if b['ownerId']==uid:rr['shots']+=1;rr['shotEnergy']+=b['power']
     else:rr['enemyShots']+=1
    if k=='BulletHitBotEvent':
     if b['ownerId']==uid:rr['hits']+=1;rr['damageDealt']+=e['damage']
     elif e['victimId']==uid:rr['hitsTaken']+=1;rr['damageTaken']+=e['damage']
    if k=='BotHitWallEvent' and e['victimId']==uid:rr['wallContacts']+=1
    if k=='BotHitBotEvent' and e['botId']==uid:rr['botContacts']+=1
   mine=[b for b in t['botStates'] if b['id']==uid]
   scans=[e for e in t['events'] if e['type']=='ScannedBotEvent' and e['scannedByBotId']==uid]
   eligible=False
   if i<len(ticks)-1 and mine and scans:
    m,s=mine[0],scans[-1]; distance=math.hypot(s['x']-m['x'],s['y']-m['y'])
    nominal=min(2.8 if distance<300 else 1.65,m['energy']/12,max(.1,(s['energy']+2)/6))
    nominal=max(.1,min(3,nominal))
    eligible=m['gunHeat']==0 and m['energy']>nominal and (m['energy']>1.2 or s['energy']<=.1)
   fired=i<len(ticks)-1 and any(e['type']=='BulletFiredEvent' and e['bullet']['ownerId']==uid for e in ticks[i+1]['events'])
   rr['commonOpportunities']+=eligible;rr['noFireOpportunities']+=eligible and not fired
   streak=streak+1 if eligible and not fired else 0;longest=max(longest,streak)
  rr['longestCommonOpportunityNoFireRun']=longest
  res=ends[r-1]['results'];delta={x['id']:x['totalScore']-prev[x['id']][0] for x in res}
  ownres=next(x for x in res if x['id']==uid)
  rr.update(score=delta[uid],totalScore=sum(delta.values()),share=delta[uid]/sum(delta.values()),wins=ownres['firstPlaces']-prev[uid][1])
  prev={x['id']:(x['totalScore'],x['firstPlaces']) for x in res}
  equal(rr,expected['perRound'][r-1],game['id']+':round'+str(r));roundrows.append(rr)
 totals={k:sum(r[k] for r in roundrows) for k in countfields}
 totals.update(longestCommonOpportunityNoFireRun=max(r['longestCommonOpportunityNoFireRun'] for r in roundrows),wins=own['firstPlaces'],share=own['totalScore']/sum(x['totalScore'] for x in official),skips=0)
 totals['hitRate']=totals['hits']/totals['shots'];totals['enemyHitRate']=totals['hitsTaken']/totals['enemyShots']
 totals['shotsPer1000ObserverTurns']=1000*totals['shots']/sum(r['turns'] for r in roundrows)
 totals['enemyShotsPer1000ObserverTurns']=1000*totals['enemyShots']/sum(r['turns'] for r in roundrows)
 totals['damagePer100Shots']=100*totals['damageDealt']/totals['shots'];totals['damagePer100EnemyShots']=100*totals['damageTaken']/totals['enemyShots']
 equal(totals,expected,game['id']);assert own['firstPlaces']==sum(r['wins'] for r in roundrows)
 assert expected['recordingSHA256']==sha(files[0]);assert expected['status']=='valid'
 if game['enemy']=='AISurvivor':
  ais=[json.loads(l) for l in (data/'ais.jsonl').read_text().splitlines()]
  assert [x['round'] for x in ais]==[1,2,3,4,5]
  assert all(x['mode']=='train' and x['seed']==game['seed'] and x['checkpointLoaded'] is False and x['totalRounds']==x['round'] and x['skippedTurns']==0 for x in ais)
  model=(data/'ais-model.dat').read_bytes();h=struct.unpack('>7i',model[:28]);assert h[:6]==(0x41495356,1,14,32,6,5) and h[6]>=0
  assert len(model)==28+8+2*(14*32+32+32*6+6)*8+5*9*47*8
  numbers=struct.unpack('>'+str((len(model)-28)//8)+'d',model[28:]);assert all(math.isfinite(n) for n in numbers) and .15<=numbers[0]<=1
  assert all(n>=0 for n in numbers[-5*9*47:])
 totals.update(id=game['id'],arm=game['arm'],enemy=game['enemy'],score=own['totalScore'],opponentScore=sum(x['totalScore'] for x in official)-own['totalScore'],perRound=roundrows)
 rows.append(totals)
output=dict(status='PASS',games=rows,physicalRounds=60,invalidRounds=0,independent=True,artifactSHA256=sha(ART/'artifact.zip'))
if args.output:
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(output,indent=2)+'\n')
for row in rows:print(row['id'],f"share={100*row['share']:.6f}% wins={row['wins']}/5 shots/hits={row['shots']}/{row['hits']} damage={row['damageDealt']:.6f}/{row['damageTaken']:.6f} longest={row['longestCommonOpportunityNoFireRun']}")
print('PASS all12/60 independent metrics+provenance')
for enemy in ('Diamond Port','Retcon','AISurvivor'):
 groups={arm:[r for r in rows if r['enemy']==enemy and r['arm']==arm] for arm in ('baseline','candidate')}
 for arm,group in groups.items():
  for key,expected_stats in decision['comparison'][enemy][arm].items():
   values=[r['wins']/5 if key=='winRate' else r[key] for r in group]
   equal(dict(mean=statistics.mean(values),sampleSD=statistics.stdev(values),values=values),expected_stats,enemy+arm+key)
 for key,val in decision['comparison'][enemy]['candidateMinusBaseline'].items():
  means={a:statistics.mean(r['wins']/5 if key=='winRate' else r[key] for r in g) for a,g in groups.items()}
  equal(means['candidate']-means['baseline'],val,enemy+key)
print('PASS equal-game means/sampleSD/allcomparisons independently reconciled')
