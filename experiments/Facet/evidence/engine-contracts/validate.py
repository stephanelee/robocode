import json,math,hashlib,subprocess,platform
from pathlib import Path
P=Path(__file__).resolve().parent

def speed(v,target):
 if v<0:return -speed(-v,-target)
 if v>0:
  if target>0:return min(8,v+min(1,target-v)) if target>=v else max(-8,v+max(-2,target-v))
  if target==0:return 0
  return v/2-1
 return math.copysign(min(1,abs(target)),target) if target else 0

def wrap(x):return (x+180)%360-180

def clamp(x,low,high):return min(high,max(low,x))

def predict(a):
 v=speed(a['speed'],a['targetSpeed']);heading=math.radians(a['body']);x=a['x']+v*math.cos(heading);y=a['y']+v*math.sin(heading)
 wall=x<18 or y<18 or x>a['width']-18 or y>a['height']-18
 body=(a['body']+clamp(a['bodyRate'],-10+.75*abs(v),10-.75*abs(v)))%360
 gun=(a['gun']+clamp(a['gunRate'],-20,20))%360;radar=(a['radar']+clamp(a['radarRate'],-45,45))%360
 oldx,oldy=a['x'],a['y'];newx,newy=x,y
 if x<18 or x>a['width']-18:
  newx=clamp(x,18,a['width']-18)
  if x!=oldx:newy=oldy+(y-oldy)*(newx-oldx)/(x-oldx)
 if y<18 or y>a['height']-18:
  newy=clamp(y,18,a['height']-18)
  if y!=oldy:newx=oldx+(x-oldx)*(newy-oldy)/(y-oldy)
 return {'x':newx,'y':newy,'speed':0 if wall else v,'body':body,'gun':gun,'radar':radar,'wall':wall}

def main():
 report={'purpose':'own-state contract diagnostics only','status':'running','comparisons':0,'maxError':{},'failures':[],'coverage':{},'limitations':['First attempt failed before game start; preserved in attempt-1/.','Raw API infinite remaining-distance values use Java Infinity token, which is nonstandard JSON; strict normalized fixtures encode this as a string.','Runner detects completion marker at turn400 asynchronously; actual last-turn and row counts below include commands delivered before stop takes effect.','Probe fire is power0.1 and isolated; no competitive interpretation.','No rejected fire request was exercised. No enemy hit/scan-energy chronology acquisition in this diagnostic.','No bot-bot collision geometry fixture was acquired.']}
 fixtures=[]
 for label in ['Alpha','Beta']:
  rows=[json.loads(l) for l in (P/(label+'.jsonl')).read_text().splitlines()]
  decisions=[r for r in rows if r['type']=='decision']; byturn={r['turn']:r for r in decisions};report[label]={'decisionRows':len(decisions),'lastTurn':max(byturn),'skips':sum(r['type']=='skip' for r in rows),'wallCallbacks':sum(r['type']=='wall' for r in rows)}
  for a,b in zip(decisions,decisions[1:]):
   if b['turn']!=a['turn']+1:continue
   expected=predict(a);errors={key:abs(wrap(b[key]-expected[key])) if key in ['body','gun','radar'] else abs(b[key]-expected[key]) for key in ['x','y','speed','body','gun','radar']}
   for key,value in errors.items():report['maxError'][key]=max(report['maxError'].get(key,0),value)
   if any(v>1e-6 for v in errors.values()):report['failures'].append({'label':label,'turn':a['turn'],'phase':a['phase'],'errors':errors})
   report['comparisons']+=1
   if label=='Alpha':
    report['coverage'][a['phase']]=report['coverage'].get(a['phase'],0)+1
    if a['phase']!='idle':
     before={k:a[k] for k in ['turn','x','y','speed','body','gun','radar','heat','width','height']};after={k:b[k] for k in ['turn','x','y','speed','body','gun','radar','heat']};command={k:a[k] for k in ['targetSpeed','bodyRate','gunRate','radarRate','fireRequested','fireAccepted','distanceRemaining']}
     if not math.isfinite(command['distanceRemaining']):command['distanceRemaining']='Infinity'
     fixtures.append({'phase':a['phase'],'before':before,'command':command,'after':after,'expectedWall':expected['wall']})
  if label=='Alpha':
   fired=[r for r in rows if r['type']=='fired'];assert len(fired)==1
   f=fired[0];a=byturn[f['eventTurn']-1];bs=next(r for r in rows if r['type']=='bulletState' and r['turn']==f['eventTurn'] and r['bullet']==f['bullet']);angle=math.radians(a['gun']);bulletSpeed=20-3*f['power']
   errors={'originX':abs(f['bulletX']-a['x']),'originY':abs(f['bulletY']-a['y']),'heading':abs(wrap(f['bulletDirection']-a['gun'])),'firstMotionX':abs(bs['bulletX']-(a['x']+bulletSpeed*math.cos(angle))),'firstMotionY':abs(bs['bulletY']-(a['y']+bulletSpeed*math.sin(angle))),'fireHeat':abs(f['heat']-(1+f['power']/5))}
   assert a['fireRequested'] and a['fireAccepted'];assert all(e<1e-6 for e in errors.values()),errors
   report['fire']={'decisionTurn':a['turn'],'requestedFireTurn':a['turn'],'confirmedFireTurn':f['eventTurn'],'firstBulletMotionTurn':bs['turn'],'acceptedRequests':1,'confirmedBullets':1,'errors':errors,'origin':[f['bulletX'],f['bulletY']],'headingDegrees':f['bulletDirection'],'heatAtCreation':f['heat']}
   report['wallContactObserved']=any(r['type']=='wall' for r in rows)
   report['cornerContactObserved']=any(a['phase']=='corner-ray' and (abs(a['x']-782)<1e-6 and abs(a['y']-582)<1e-6) for a in decisions)
 report['status']='passed' if not report['failures'] else 'failed';report['fixturesSHA256']=hashlib.sha256((json.dumps(fixtures,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
 (P/'motion-fixtures.json').write_text(json.dumps(fixtures,indent=2,allow_nan=False)+'\n')
 report['runtime']=subprocess.run(['/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/bin/java','-version'],capture_output=True,text=True).stderr;report['os']=platform.platform();report['hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'Alpha.jsonl',P/'Beta.jsonl',P/'ContractProbe.java',P/'ContractRunner.java',P/'official-server.jar',P/'validate.py',P/'protocol.md']}
 (P/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2,allow_nan=False))
 if report['failures']:raise SystemExit(1)
if __name__=='__main__':main()
