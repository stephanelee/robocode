"""Stream and independently audit retained issue-93 rows; never execute the diagnostic or new cases."""
import argparse,collections,gzip,hashlib,json,math,pathlib,struct,zipfile
P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--repository',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[3])
P.add_argument('--raw',type=pathlib.Path,help='Reassembled original attempt directory; defaults to repository/build/meridian-wave-support-attempt')
P.add_argument('--output',type=pathlib.Path)
P.add_argument('--verify-local-runtime',action='store_true',help='Also rehash original absolute JDK/Python/classpath and build-output files; requires original host')
A=P.parse_args();ROOT=A.repository.resolve();RAW=(A.raw or ROOT/'build/meridian-wave-support-attempt').resolve();REG=ROOT/'benchmarks/meridian-wave-support'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def lines(name):
 with gzip.open(RAW/name,'rt') as f:
  for line in f:yield json.loads(line)
def close(a,b,tag='',tol=1e-8):
 if isinstance(a,dict):
  for k,v in a.items():close(v,b[k],tag+'.'+k,tol)
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),(tag,len(a),len(b))
  for i,(x,y) in enumerate(zip(a,b)):close(x,y,tag+str(i),tol)
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=tol),(tag,a,b)
 else:assert a==b,(tag,a,b)
def legal(p):return 18<=p['x']<=782 and 18<=p['y']<=582
def sn(h):return math.sin(math.radians(h))
def cs(h):return math.cos(math.radians(h))
def clip(v,a,b):return max(a,min(v,b))
def history(s,t):
 p=s['own'];return dict(x=p['x']-(13-t)*p['v']*cs(p['h']),y=p['y']-(13-t)*p['v']*sn(p['h']),h=p['h'],v=p['v'])
def speed(v,target):
 if v<0:return -speed(-v,-target)
 if v>0:
  if target>0:return min(v+min(target-v,1),8) if target>=v else max(v+max(target-v,-2),-8)
  return 0.0 if target==0 else v/2-1
 return clip(target,-1,1)
def constraint(ox,oy,x,y):
 nx,ny=x,y
 if x<18 or x>782:
  nx=clip(x,18,782)
  if x!=ox:ny=oy+(y-oy)*(nx-ox)/(x-ox)
 if y<18 or y>582:
  ny=clip(y,18,582)
  if y!=oy:nx=ox+(x-ox)*(ny-oy)/(y-oy)
 return nx,ny
def step(p,c,passes):
 v=speed(p['v'],c['speed']);x=p['x']+cs(p['h'])*v;y=p['y']+sn(p['h'])*v
 nx,ny=constraint(p['x'],p['y'],x,y);newv=0.0 if nx!=x or ny!=y else v
 if passes==2:nx,ny=constraint(p['x'],p['y'],nx,ny)
 return dict(x=nx,y=ny,h=p['h']+clip(c['turn'],-10+.75*abs(v),10-.75*abs(v)),v=newv),(x,y),constraint(p['x'],p['y'],x,y)
def orbit(p,s,sign):
 distance=math.hypot(p['x']-s['sx'],p['y']-s['sy'])
 angle=math.degrees(math.atan2(s['sy']-p['y'],s['sx']-p['x']))+sign*(90+clip((380-distance)*.1,-32,32))
 for _ in range(72):
  x=p['x']+cs(angle)*110;y=p['y']+sn(angle)*110
  if 24<=x<=776 and 24<=y<=576:break
  angle+=sign*5
 d=math.remainder(angle-p['h'],360)
 return dict(turn=math.remainder(d+180,360),speed=-8.0) if abs(d)>90 else dict(turn=d,speed=8.0)
def pinned_intersection(x1,y1,x2,y2,cx,cy):
 if (x1-cx)**2+(y1-cy)**2<=324 or (x2-cx)**2+(y2-cy)**2<=324:return True
 dx,dy=x2-x1,y2-y1; dot=((cx-x1)*dx+(cy-y1)*dy)/(dx*dx+dy*dy)
 px,py=x1+dot*dx,y1+dot*dy
 if not abs((px-x1)*dy-(py-y1)*dx)<1e-6:return False
 on=(x1<=px<=x2 if dx>0 else x2<=px<=x1) if abs(dx)>=abs(dy) else (y1<=py<=y2 if dy>0 else y2<=py<=y1)
 return on and (px-cx)**2+(py-cy)**2<=324
def oracle(s,t,h,p,retained):
 v=20-3*s['power'];points=dict(x1=s['sx']+cs(h)*(t+1)*v,y1=s['sy']+sn(h)*(t+1)*v,x2=s['sx']+cs(h)*(t+2)*v,y2=s['sy']+sn(h)*(t+2)*v)
 close(points,retained,'bullet endpoints')
 # Classify using retained endpoint doubles: isolate classifier fidelity from cross-language trig rounding.
 x1,y1,x2,y2=[retained[k] for k in ['x1','y1','x2','y2']]
 dx,dy=x2-x1,y2-y1;u=clip(((p['x']-x1)*dx+(p['y']-y1)*dy)/(dx*dx+dy*dy),0,1)
 dist=(x1+u*dx-p['x'])**2+(y1+u*dy-p['y'])**2
 removed=x1<=0 or y1<=0 or x1>=800 or y1>=600
 contact=dist<=324;literal=pinned_intersection(x1,y1,x2,y2,p['x'],p['y'])
 q=dict(distanceSquared=dist,removed=removed,contact=not removed and contact,pinnedEngineContact=not removed and literal,geometryAgreement=contact==literal)
 close(q,retained,'oracle');return q
manifest=read(RAW/'artifact-files.json')
for name,h in manifest.items():assert sha(RAW/name)==h,name
assert sha(RAW/'freeze.json')=='093a4ad1e203b16d73461a751e6d7c761bebc316cc41d6894bf2b47d71cd1403'
assert (RAW/'actual-manifest.json').read_bytes()==(RAW/'freeze.json').read_bytes()
freeze=read(RAW/'freeze.json');clear=read(RAW/'clearance.json');attempt=read(RAW/'attempt.json')
assert clear['decision']=='approved' and clear['freezeSHA256']==sha(RAW/'freeze.json')
assert attempt['head']==clear['sourceHead']=='e57d71c86789af43e02a4d107fae79054b57cfee'
assert attempt['freezeSHA256']==sha(RAW/'freeze.json')
for name,h in freeze['sourceFiles'].items():
 if '/build/' not in name or A.verify_local_runtime:assert sha(ROOT/name)==h,name
archive=REG/'artifact/wave-support-diagnostic.jar'
with zipfile.ZipFile(archive) as z:
 classes={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith('.class')}
assert classes==freeze['diagnosticClasses']
for name,h in freeze['sourceFiles'].items():
 marker='build/classes/java/diagnostic/'
 if marker in name:assert classes[name.split(marker,1)[1]]==h
for name,item in freeze['engine']['files'].items():
 p=REG/'engine-source'/name;data=p.read_bytes()
 assert sha(p)==item['sha256'];assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['gitBlob']
assert sha(archive)==freeze['classpath'][0]['sha256']
assert sha(ROOT/'experiments/Meridian/Meridian.jar')==freeze['classpath'][1]['sha256']
if A.verify_local_runtime:
 for entry in freeze['classpath']:assert sha(pathlib.Path(entry['path']))==entry['sha256']
 for name,h in freeze['runtime']['jdkFiles'].items():assert sha(pathlib.Path(freeze['runtime']['jdkRoot'])/name)==h
 for name,h in freeze['runtime']['pythonFiles'].items():assert sha(pathlib.Path(name))==h
C=collections.Counter(); states={};hist={};routes={};coverage=collections.Counter();summary=read(RAW/'summary.json')
# Independent Cartesian construction verifies every frozen state, without evaluating a new route.
expected=[]
for power in [.1,.4,1,1.65,1.95,2.8,3]:
 v=20-3*power
 for ei,e in enumerate([-(v+18)-1e-6,-(v+18)+1e-6,-22-1e-6,-22+1e-6,18-1e-6,18+1e-6,22+1e-6]):
  for heading in [0,90,180,270]:
   for velocity in [-8,-3,0,3,8]:
    for placement in ['open','wall','corner']:
     r=15*v+e
     if placement=='open':sx,sy,x,y,b=400,300,400+r,300,0
     elif placement=='wall':sx,sy,x,y,b=780-r,300,780,300,0
     else:sx,sy,x,y,b=780-r/math.sqrt(2),580-r/math.sqrt(2),780,580,45
     expected.append(dict(id=len(expected),placement=placement,errorIndex=ei,headingOffset=heading,power=power,sx=sx,sy=sy,bearing=b,own=dict(x=x,y=y,h=b+heading,v=velocity),anchor=False))
expected.append(dict(id=2940,placement='anchor',errorIndex=-1,headingOffset=270,power=1.95,sx=400,sy=300,bearing=0,own=dict(x=600.4,y=318.1,h=270,v=8),anchor=True))
for item in lines('states.jsonl.gz'):
 s=item['state'];sid=s['id'];assert sid==len(states);close(expected[sid],s,'state');assert legal(s['own'])
 p=history(s,-1);angle=math.degrees(math.atan2(p['y']-s['sy'],p['x']-s['sx']));lead=angle+math.degrees(math.asin(p['v']*sn(p['h']-angle)/(20-3*s['power'])))
 close(dict(t=0,x=s['sx'],y=s['sy'],center=angle,lead=lead,power=s['power']),item['wavePrior'],'prior')
 far=15*(20-3*s['power']);close(far,item['currentFar']);assert item['retired']==(far>math.hypot(s['own']['x']-s['sx'],s['own']['y']-s['sy'])+36)
 states[sid]=item;C['states']+=1
for item in lines('history-rays.jsonl.gz'):
 sid,h=item['stateID'],item['heading'];s=states[sid]['state'];key=(sid,h);assert key not in hist
 poses=[history(s,t) for t in range(-1,14)];close(poses,item['historyPosesTurnsMinus1To13'],'historyposes')
 reasons=[f'historyOutsideArena@{t}' for t,p in zip(range(-1,14),poses) if not legal(p)]; hit=wall=None
 for t,raw in enumerate(item['historyOracle']):
  assert raw['turn']==t;close(poses[t+1],raw['pose']);q=oracle(s,t,h,raw['pose'],raw['oracle']);C['historyContactChecks']+=1
  C['geometryClassificationDisagreements']+=not q['geometryAgreement']
  if q['removed']:wall=t;reasons.append(f'historyBulletWall@{t}');break
  if q['contact']:hit=t;reasons.append(f'historyContact@{t}');break
 assert len(item['historyOracle'])==(wall+1 if wall is not None else hit+1 if hit is not None else 14)
 assert item['reasons']==reasons and item['eligible']==(not reasons)
 e=dict(heading=h,reasons=reasons,historyContact=hit,historyWall=wall);close(e,next(x for x in states[sid]['eligibleRays'] if x['heading']==h))
 hist[key]=dict(eligible=not reasons,reasons=reasons);C['eligibleStateRays' if not reasons else 'excludedStateRays']+=1
for sid,item in states.items():
 s=item['state'];headings=[0] if s['anchor'] else [s['bearing']+x for x in [-30,-15,0,15,30]]
 assert len(item['eligibleRays'])==len(headings) and [e['heading'] for e in item['eligibleRays']]==headings
 assert all((sid,h) in hist for h in headings)
lastkey=None;rowindex=0;maxerrors=dict(X=0.,Y=0.,Speed=0.,Heading=0.)
for row in lines('positions.jsonl.gz'):
 sid,sign,i=row['stateID'],row['sign'],row['step'];s=states[sid]['state'];item=states[sid]
 expectedkey=(rowindex//100,-1 if rowindex%100<50 else 1,rowindex%50+1);assert (sid,sign,i)==expectedkey
 key=(sid,sign)
 if i==1:
  route=dict(old=[0. if sign==-1 else 2.,0. if sign==1 else 2.],new=[0. if sign==-1 else 2.,0. if sign==1 else 2.],walls=0,om=0,nm=0,rays={},pose=s['own'],motion=0,firstdiv=None,first=None)
  routes[key]=route
 else:route=routes[key]
 close(route['pose'],row['before'],'continuity',0)
 command=orbit(row['before'],s,sign);close(command,row['command'],'orbitcommand')
 pred,unrestricted,first=step(row['before'],row['command'],1);close(pred,row['pose'],'onepass')
 ref,_,_=step(row['before'],row['command'],2);storedref=row['motionReferencePose'];close(ref['x'],storedref['x']);close(ref['y'],storedref['y']);close(ref['v'],storedref['v']);close(math.remainder(ref['h']-storedref['h'],360),0.)
 # Java retained poses define the exact recorded segment/risk inputs after independent transition verification.
 p=row['pose'];independentErrors=[abs(p['x']-ref['x']),abs(p['y']-ref['y']),abs(p['v']-ref['v']),abs(math.remainder(p['h']-ref['h'],360))]
 assert any(x>1e-9 for x in independentErrors)==row['motionMismatch']
 coverage['independentMotionMaximumAbsoluteError']=max(coverage['independentMotionMaximumAbsoluteError'],*independentErrors)
 errs=dict(X=abs(p['x']-storedref['x']),Y=abs(p['y']-storedref['y']),Speed=abs(p['v']-storedref['v']),Heading=abs(math.remainder(p['h']-storedref['h'],360)))
 mismatch=any(v>1e-9 for v in errs.values());outside=not legal(row['before'])
 close(dict(motionErrorX=errs['X'],motionErrorY=errs['Y'],motionErrorSpeed=errs['Speed'],motionErrorWrappedHeading=errs['Heading'],motionInputOutside=outside,motionMismatch=mismatch),row)
 for k,v in errs.items():maxerrors[k]=max(maxerrors[k],v)
 C['motionReferenceComparisons']+=1;C['motionMismatchRows']+=mismatch;C['motionInputOutsideRows']+=outside
 if mismatch:route['motion']+=1;route['firstdiv']=i if route['firstdiv'] is None else route['firstdiv']
 assert route['firstdiv']==row['firstMotionDivergenceStep']
 x,y=unrestricted;wall=p['x']!=row['unrestrictedX'] or p['y']!=row['unrestrictedY'];close(x,row['unrestrictedX']);close(y,row['unrestrictedY']);assert wall==row['wallContact']
 coverage['rawDoubleAxisBoundaryCrossings']+=(x<18 or x>782) and (y<18 or y>582)
 coverage['onePassOutsideAfterConstraint']+=not (18<=first[0]<=782 and 18<=first[1]<=582)
 coverage['wallConstraintSteps']+=wall
 cost=max(0,40-min(p['x'],800-p['x'],p['y'],600-p['y']))*.1;close(cost,row['wallCost'])
 dx,dy=p['x']-s['sx'],p['y']-s['sy'];r=math.hypot(dx,dy);v=20-3*s['power'];far=(15+i)*v
 close(dict(radius=r,far=far,speed=v,radialError=r-far),row)
 old=abs(r-far)<=22;new=far-v-18<=r<=far+18
 assert (old,new,item['retired'])==(row['oldAdmitted'],row['newAdmitted'],row['retired'])
 risk=sum(40/(1+(dx*sn(a)-dy*cs(a))**2/500) for a in [item['wavePrior']['center'],item['wavePrior']['lead']]);close(risk,row['rawRisk'])
 route['om']+=old;route['nm']+=new;route['walls']+=wall
 oldcon=risk if old and not item['retired'] else 0.;newcon=risk if new and not item['retired'] else 0.
 close(dict(oldRawContribution=risk if old else 0.,newRawContribution=risk if new else 0.,oldPolicyContribution=oldcon,newPolicyContribution=newcon,oldMultiplicity=route['om'],newMultiplicity=route['nm'],oldPolicyMultiplicity=0 if item['retired'] else route['om'],newPolicyMultiplicity=0 if item['retired'] else route['nm']),row)
 for idx in [0,1]:route['old'][idx]+=cost;route['new'][idx]+=cost;route['old'][idx]+=oldcon;route['new'][idx]+=newcon
 close(route['old'],row['oldAccumByCommittedMinusPlus'],'oldscore');close(route['new'],row['newAccumByCommittedMinusPlus'],'newscore')
 assert [a['heading'] for a in row['rayChecks']]==[a['heading'] for a in item['eligibleRays']]
 for rr in row['rayChecks']:
  h=rr['heading'];e=hist[sid,h];finished=route['rays'].get(h)
  if not e['eligible']:assert rr['status']=='ineligible';continue
  if finished:assert rr['status']==('stoppedAfterContact' if finished['hit'] else 'stoppedAfterWall');continue
  assert rr['status']=='checked';q=oracle(s,13+i,h,p,rr['oracle']);C['futureContactChecks']+=1;C['geometryClassificationDisagreements']+=not q['geometryAgreement']
  if q['removed']:route['rays'][h]=dict(hit=None,wall=i,old=None,new=None,dist=None)
  elif q['contact']:
   route['rays'][h]=dict(hit=i,wall=None,old=old,new=new,dist=q['distanceSquared']);C['oldOmittedFirstContacts']+=not old;C['newOmittedFirstContacts']+=not new;C['retiredFutureContacts']+=item['retired']
 if i==1:route['first']=p
 route['pose']=p;C['positionRows']+=1;rowindex+=1
 if i==50:C['trajectories']+=1;C['motionDivergentTrajectories']+=route['firstdiv'] is not None
for row in lines('contexts.jsonl.gz'):
 sid,side=row['stateID'],row['committedSide'];idx=0 if side==-1 else 1;minus,plus=routes[sid,-1],routes[sid,1]
 assert (sid,side)==(C['contexts']//2,-1 if C['contexts']%2==0 else 1)
 old=[minus['old'][idx],plus['old'][idx]];new=[minus['new'][idx],plus['new'][idx]]
 close(old,row['actualSourceScoresMinusPlus']);close(old,row['helperOldScoresMinusPlus']);close(new,row['newScoresMinusPlus'])
 # Actual Java source/helper parity is bit exact in the recorded evidence, not a Python tolerance claim.
 assert row['actualSourceScoresMinusPlus']==row['helperOldScoresMinusPlus']
 bits=[struct.unpack('>q',struct.pack('>d',x))[0] for x in row['actualSourceScoresMinusPlus']];assert bits==row['oldScoreBitsMinusPlus'];C['parityComparisons']+=2
 oc=1 if row['helperOldScoresMinusPlus'][1]<row['helperOldScoresMinusPlus'][0] else -1
 nc=1 if row['newScoresMinusPlus'][1]<row['newScoresMinusPlus'][0] else -1
 assert (oc,nc)==(row['oldChoice'],row['newChoice']);o,n=routes[sid,oc],routes[sid,nc]
 eligible=[h for state,h in hist if state==sid and hist[state,h]['eligible']];count=len(eligible)
 exposure=lambda route:sum(route['rays'].get(h,{}).get('hit') is not None for h in eligible)
 oe,ne=exposure(o),exposure(n)
 close(dict(eligibleRayCount=count,oldSelectedExposures=oe,newSelectedExposures=ne,oldSelectedWallContacts=o['walls'],newSelectedWallContacts=n['walls'],minusFirstMotionDivergenceStep=minus['firstdiv'],plusFirstMotionDivergenceStep=plus['firstdiv'],minusMotionMismatchRows=minus['motion'],plusMotionMismatchRows=plus['motion']),row)
 for label,route in [('minusRouteRays',minus),('plusRouteRays',plus)]:
  for rr in row[label]:
   h=rr['heading'];e=hist[sid,h];result=route['rays'].get(h,dict(hit=None,wall=None,old=None,new=None,dist=None))
   close(dict(eligible=e['eligible'],reasons=e['reasons'],firstContact=result['hit'],wallRemoval=result['wall'],oldAdmitted=result['old'],newAdmitted=result['new'],firstContactDistanceSquared=result['dist']),rr)
 C['contexts']+=1;C['actionChanges']+=oc!=nc;C['motionMismatchContexts']+=bool(minus['motion'] or plus['motion'])
 C['oldSelectedWallsAllContexts']+=o['walls'];C['newSelectedWallsAllContexts']+=n['walls']
 if count:C['eligibleContexts']+=1;C['eligibleActionChanges']+=oc!=nc;C['oldSelectedWallsEligibleContexts']+=o['walls'];C['newSelectedWallsEligibleContexts']+=n['walls']
 C['commonEligibleRayContexts']+=count;C['oldSelectedExposures']+=oe;C['newSelectedExposures']+=ne;C['oldSelectedWallsRayWeighted']+=o['walls']*count;C['newSelectedWallsRayWeighted']+=n['walls']*count
anchorplus,anchorminus=routes[2940,1],routes[2940,-1]
C['anchorContinuingContact']=anchorplus['rays'][0.]['hit']==1
C['anchorReverseFirstMiss']=anchorminus['rays'][0.]['hit']!=1
C['anchorOldOmittedAll50']=anchorminus['om']==0 and anchorplus['om']==0
C['anchorNewAdmittedFirst']=anchorplus['rays'][0.]['new'] is True
C['anchorFirstPoses']=math.isclose(anchorplus['first']['x'],600.4,abs_tol=1e-10) and math.isclose(anchorplus['first']['y'],310.1,abs_tol=1e-10) and math.isclose(anchorminus['first']['x'],600.4,abs_tol=1e-10) and math.isclose(anchorminus['first']['y'],315.1,abs_tol=1e-10)
for k,v in maxerrors.items():C['maximumMotionError'+k]=v
for k,v in summary['counts'].items():close(C[k],v,'summarycount.'+k)
valid=C['states']==2941 and C['contexts']==5882 and C['trajectories']==5882 and C['positionRows']==294100 and C['parityComparisons']==11764 and C['geometryClassificationDisagreements']==0
passed=valid and all(C[k] for k in ['anchorContinuingContact','anchorReverseFirstMiss','anchorOldOmittedAll50','anchorNewAdmittedFirst','anchorFirstPoses']) and C['newOmittedFirstContacts']==0 and C['retiredFutureContacts']==0
advance=passed and C['eligibleActionChanges']>0 and C['newSelectedExposures']<C['oldSelectedExposures'] and C['newSelectedWallsEligibleContexts']<=C['oldSelectedWallsEligibleContexts'] and C['motionMismatchRows']==0
assert (valid,passed,advance)==(summary['diagnosticValid'],summary['diagnosticPass'],summary['advanceToProposal'])
result=dict(audit='PASS',independent=True,localRuntimeReverified=A.verify_local_runtime,counts=dict(C),coverage=dict(coverage),diagnosticValid=valid,diagnosticPass=passed,advanceToProposal=advance,competitiveRounds=0,freezeSHA256=sha(RAW/'freeze.json'),rawFilesSHA256=manifest,limitations='Raw-transition audit only; no new case or candidate execution. Zero motion mismatches cover this fixed sampled command family, not all legal corner states. Python general numeric comparisons use 1e-8 absolute/1e-12 relative tolerance; independently recomputed motion mismatch decisions use the registered 1e-9 absolute threshold; recorded Java parity bits are checked exactly. Retained segment endpoint doubles classify contact independently of cross-language trig rounding. Frozen and recorded actual runtime manifests must match exactly; optional local remeasurement requires the original host/cache paths.')
if A.output:A.output.parent.mkdir(parents=True,exist_ok=True);A.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
