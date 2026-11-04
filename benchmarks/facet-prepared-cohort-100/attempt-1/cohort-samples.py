"""Bounded own FACET critical cohort provenance check; no battle or opponent access."""
import argparse,json,re,math
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
count=common=different=0;maximum=0
for path in sorted(args.root.glob('diamond-port-g*/FACET_metrics.jsonl.critical.jsonl')):
 for line in path.read_text().splitlines():
  row=json.loads(line)
  if row['kind']!='gun.cohort.confirmed':continue
  text=row['value']
  def field(name):return re.search(r'(?:^| )'+name+r'=([^ ]+)',text).group(1)
  selected=field('selected');actual=float(field('actualAngle'));request=int(field('request'));creation=int(field('creation'));prepared=int(field('preparedTurn'));source=int(field('preparedWave'))
  assert creation==row['turn']==request+1 and request-prepared==1
  predictions=re.findall(r'(\w+)=Prediction\[gun=(\w+), wave=(-?\d+), turn=(\d+), angle=([^,]+), power=([^,]+), version=(\d+)\]',text)
  assert len(predictions) in (1,3) and selected in [p[0] for p in predictions]
  for name,gun,wave,turn,angle,power,version in predictions:
   assert name==gun and int(wave)==source and int(turn)==prepared
   if name==selected:
    error=abs(math.remainder(actual-float(angle),2*math.pi));assert error<=math.radians(.5)+1e-9;maximum=max(maximum,error)
  views=re.search(r'modelViews=\{(.*)\}',text).group(1);assert 'analytic=[]' in views
  if len(predictions)==3:
   epoch=int(re.search(r'long=\[(\d+)\]',views).group(1));recent=list(map(int,re.search(r'recent=\[(.*?)\]',views).group(1).split(', ')));assert len(recent)==3
   common+=1;different+=int(any(v!=epoch for v in recent))
  count+=1
result=dict(sampledConfirmations=count,sampledCommonCohorts=common,exactRecentVectorsDifferFromLongEpoch=different,maxSelectedActualHeadingErrorRadians=maximum,timingAndOriginalPredictionIdentity='PASS',scope='Bounded critical samples only; all-shot cumulative guards verified separately')
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
