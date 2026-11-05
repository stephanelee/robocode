"""Supplemental public observer/own-marker arithmetic. No imports from frozen reducer/harness."""
import argparse,collections,gzip,hashlib,io,json,math,re,statistics,zipfile
from fractions import Fraction
from pathlib import Path
COUNTS=('shots','hits','shotEnergy','damageDealt','enemyShots','enemyHits','enemyShotEnergy','damageTaken','wallContacts','botContacts','bulletCollisions')
EXPOSURE=('rawRamEvents','eligibleRamCorrectionEvents','eligibleRamCorrectedScans','ramChangedPowerScans','ramChangedGateScans')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def game(z,prefix,g):
 names=[n for n in z.namelist() if n.startswith(prefix+'games/'+g['id']+'/') and n.endswith('.battle.gz')];require(len(names)==1,'recording cardinality '+g['id']);name=names[0]
 counters=dict.fromkeys(COUNTS,0);per={};ends={};turns={};mix=[collections.Counter(),collections.Counter()];final=None;own=None;h=hashlib.sha256()
 with z.open(name) as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 with z.open(name) as f,gzip.GzipFile(fileobj=f) as gz,io.TextIOWrapper(gz,encoding='utf8') as lines:
  first=json.loads(next(lines));require(first['type']=='GameStartedEventForObserver','start');own=next(p['id'] for p in first['participants'] if p['name']==g['participantName']);other=next(p['id'] for p in first['participants'] if p['id']!=own)
  for line in lines:
   e=json.loads(line);require(final is None,'after final');kind=e['type']
   if kind=='TickEventForObserver':
    r=e['roundNumber'];t=e['turnNumber'];require(t==turns.get(r,0)+1 and r not in ends,'tick sequence');turns[r]=t;per.setdefault(r,dict.fromkeys(COUNTS,0))
    def add(key,n=1):
     require(type(n) in (int,float) and math.isfinite(n) and n>=0,'count');counters[key]+=n;per[r][key]+=n
    for ev in e['events']:
     k=ev['type']
     if k=='BulletFiredEvent':
      mine=ev['bullet']['ownerId']==own;add('shots' if mine else 'enemyShots');add('shotEnergy' if mine else 'enemyShotEnergy',ev['bullet']['power']);mix[0 if mine else 1][str(ev['bullet']['power'])]+=1
     elif k=='BulletHitBotEvent':
      mine=ev['bullet']['ownerId']==own;add('hits' if mine else 'enemyHits');add('damageDealt' if mine else 'damageTaken',ev['damage'])
     elif k=='BulletHitBulletEvent' and own in (ev['bullet']['ownerId'],ev['hitBullet']['ownerId']):add('bulletCollisions')
     elif k=='BotHitWallEvent' and ev['victimId']==own:add('wallContacts')
     elif k=='BotHitBotEvent' and ev['botId']==own:add('botContacts')
   elif kind=='RoundEndedEventForObserver':
    r=e['roundNumber'];require(r==len(ends)+1 and turns[r]==e['turnNumber'],'round endpoint');ends[r]=e
   elif kind=='GameEndedEventForObserver':final=e
   else:raise ValueError('unknown observer message '+kind)
 require(list(ends)==list(range(1,g['rounds']+1)) and final is not None,'full rounds');require(final['results']==ends[g['rounds']]['results'],'final endpoint')
 marks=z.read(prefix+'games/'+g['id']+'/markers.txt').decode().splitlines();m={r:{'start':[],'stats':[],'end':[],'skip':[]} for r in ends}
 for i,line in enumerate(marks):
  found=re.fullmatch(r'([1-9][0-9]*) (start|stats(?: .*)?|end|skip(?: .*)?)',line);require(found is not None,'marker syntax');r=int(found[1]);k=found[2].split()[0];m[r][k].append((i,line))
 previousScore=0;previousTotal=0;previousWins=0;maxms=0;skips=0;expo=dict.fromkeys(EXPOSURE,0) if g['bot']=='candidate' else None;rows=[]
 for r,e in ends.items():
  row=m[r];require(all(len(row[k])==1 for k in ('start','stats','end')),'marker cardinality');require(row['start'][0][0]<row['stats'][0][0]<row['end'][0][0],'marker order');require(r==1 or m[r-1]['end'][0][0]<row['start'][0][0],'round marker interleave');s=row['stats'][0][1];v=re.findall(r'(?:^| )maxMs=([0-9]+)(?: |$)',s);require(len(v)==1,'timing');ms=int(v[0]);maxms=max(maxms,ms);skips+=len(row['skip']);x=None
  if expo is not None:
   x={}
   for key in EXPOSURE:
    val=re.findall(r'(?:^| )'+key+r'=([0-9]+)(?: |$)',s);require(len(val)==1,'exposure');x[key]=int(val[0]);require(x[key]<=1000000,'counter cap');expo[key]+=x[key]
   require(x['ramChangedGateScans']<=x['ramChangedPowerScans']<=x['eligibleRamCorrectedScans']<=x['eligibleRamCorrectionEvents']<=x['rawRamEvents']<=per[r]['botContacts'],'counter chain')
  scores={x['id']:x['totalScore'] for x in e['results']};wins={x['id']:x['firstPlaces'] for x in e['results']};a=scores[own]-previousScore;b=sum(scores.values())-previousTotal;w=wins[own]-previousWins;require(0<=a<=b and b>0 and w in (0,1),'round scores');rows.append(dict(round=r,turns=turns[r],ownScore=a,totalScore=b,win=w,share=a/b,maxMs=ms,exposure=x,**per[r]));previousScore=scores[own];previousTotal=sum(scores.values());previousWins=wins[own]
 return dict(game=g,recordingSHA256=h.hexdigest(),ownScore=previousScore,totalScore=previousTotal,wins=previousWins,share=previousScore/previousTotal,turns=sum(turns.values()),skips=skips,maxMs=maxms,exposure=expo,perRound=rows,powerMix=dict(mix[0]),enemyPowerMix=dict(mix[1]),hitRate=counters['hits']/counters['shots'] if counters['shots'] else None,enemyHitRate=counters['enemyHits']/counters['enemyShots'],firingPer1000Turns=counters['shots']*1000/sum(turns.values()),**counters)
def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--prefix',required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.registry.read_text())['screen'];require(len(plan)==14 and sum(g['rounds'] for g in plan)==120,'fixed population')
 with zipfile.ZipFile(a.zip) as z:rows=[game(z,a.prefix,g) for g in plan]
 groups={}
 for arm in ('candidate','baseline'):
  for opponent in ('diamond','retcon','walls','ais'):
   r=[x for x in rows if x['game']['bot']==arm and x['game']['opponent']==opponent];vals=[Fraction(x['ownScore'],x['totalScore']) for x in r];mean=sum(vals,Fraction())/len(vals);groups[arm+'/'+opponent]=dict(mean=float(mean),fraction=[mean.numerator,mean.denominator],wins=sum(x['wins'] for x in r),aboveHalf=sum(v>Fraction(1,2) for v in vals),sampleSD=statistics.stdev(map(float,vals)) if len(vals)>1 else None)
 def value(k):return Fraction(*groups[k]['fraction'])
 d=value('candidate/diamond');dd=d-value('baseline/diamond');r=value('candidate/retcon');dr=r-value('baseline/retcon');exposure=sum(x['exposure']['eligibleRamCorrectedScans'] for x in rows if x['game']['bot']=='candidate' and x['game']['opponent']=='retcon');checks=dict(zeroSkips=all(x['skips']==0 for x in rows),timing=all(x['maxMs']<75 for x in rows),retconExposure=exposure>0,diamondAboveHalf=d>Fraction(1,2),diamondDelta=dd>=Fraction(-3,100),diamondWins=groups['candidate/diamond']['wins']>=16,diamondGames=groups['candidate/diamond']['aboveHalf']>=2,retconShare=r>=Fraction(9,10),retconDelta=dr>=Fraction(1,100),retconWins=groups['candidate/retcon']['wins']>=16)
 for opp in ('walls','ais'):checks[opp+'Delta']=value('candidate/'+opp)-value('baseline/'+opp)>=Fraction(-5,100);checks[opp+'Wins']=groups['candidate/'+opp]['wins']>=4
 out=dict(scope='Supplemental independent raw public-event/own-marker arithmetic only; primary custody/authority audit still mandatory',sourceSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),games=rows,groups=groups,diamondDelta=float(dd),retconDelta=float(dr),candidateRetconEligibleScans=exposure,checks=checks,allGates=all(checks.values()));a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
