"""Independent own-trace cross-check; no model/opponent inputs or launches."""
import argparse,json,re,math,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
def vals(s):
 d={}
 for m in re.finditer(r'\b([A-Za-z]+)=([-+0-9.eE]+)(?=[ ,]|$)',s):d.setdefault(m[1],float(m[2]))
 return d
def angular(a,b):return abs(math.atan2(math.sin(a-b),math.cos(a-b)))
games=[];total=collections.Counter();guns=collections.Counter();cohorts=0
for game in range(1,6):
 states={};commands={};prepared={};seen=set();c=collections.Counter();gg=collections.Counter();rating={}
 with (args.root/f'diamond-port-g{game:02d}'/'FACET_trace.jsonl').open() as file:
  for n,line in enumerate(file,1):
   assert n<=100000
   row=json.loads(line);kind=row['kind'];text=row['value'];key=row['round'],row['turn']
   if kind=='own.state':states[key]=vals(text)
   elif kind=='fire.prepared':prepared[key]=vals(text)
   elif kind=='command.submitted':
    commands[key]=text;c['decisions']+=1
    if 'fire=true' in text:c['fireCommands']+=1
    s=states.get(key);old=prepared.get((key[0],key[1]-1))
    if s and s['heat']==0 and s['energy']>.3:
     c['coolFunded']+=1
     if old and s['energy']>old['power'] and angular(s['gun'],old['angle'])<=math.radians(.5)+1e-9:
      c['coarseAligned']+=1
      if 'fire=false' in text:c['coarseAlignedNoSubmit']+=1
   elif kind=='fire.confirmed':
    v=vals(text);request=key[0],int(v['request']);s=states[request];assert 'fire=true' in commands[request]
    identity=key[0],int(v['id']);assert identity not in seen;seen.add(identity)
    assert angular(math.radians(v['heading']),s['gun'])<=1e-7
    pos=re.search(r'position=([^, ]+),([^ ]+)',text);assert math.hypot(float(pos[1])-s['x'],float(pos[2])-s['y'])<=1e-6
    assert key[1]==int(v['request'])+1==int(v['creation'])==int(v['firstMotion'])
    assert 'pending=true' in text
    gun=re.search(r'\bgun=(analytic|long|recent)\b',text)[1];gg[gun]+=1;c['reconciled']+=1
   elif kind=='gun.rating':
    gun=re.search(r'\bgun=(analytic|long|recent)\b',text)[1];v=vals(text);rating.setdefault(key[0],{})[gun]=int(v['cohort']);assert 0<=v['raw']<=1 and 0<=v['adjusted']<=10
 assert n==100000;c['lines']=n
 rc=0
 for views in rating.values():
  if len(views)==3:assert len(set(views.values()))==1;rc+=1
 cohorts+=rc;total.update(c);guns.update(gg);games.append(dict(game=game,counts=dict(c),guns=dict(gg),commonRatingRoundSnapshots=rc))
result=dict(totals=dict(total),guns=dict(guns),commonRatingRoundSnapshots=cohorts,games=games)
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['totals']));print(dict(guns),cohorts)
