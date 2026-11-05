import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
import benchmark
root=Path.cwd();base=Path('/tmp/facet-adapter');output=base/'attempt-1'
assert not output.exists()
assert benchmark.digest(root/'experiments/Facet/Facet.jar')=='73be04d7386a60474743a78fcebf3fd0c143e37120b410b18aa668bd5ed50c16'
output.mkdir()
report={'status':'running','games':[],'purpose':'bounded actual adapter integration, no competitive interpretation'}
def save(): (output/'status.json').write_text(json.dumps(report,indent=2)+'\n')
save();java='/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/bin/java';begin=time.monotonic()
try:
 for index,seed in enumerate((49667,49668)):
  if time.monotonic()-begin>350: raise RuntimeError('insufficient remaining 600-second total budget')
  run=output/f'game-{index+1}';run.mkdir()
  bot=run/'Facet';bot.mkdir()
  for name in ('Facet.jar','Facet.json','Facet.sh'):shutil.copy2(root/'experiments/Facet'/name,bot/name)
  (bot/'Facet.sh').chmod(0o755)
  partner=run/'ContractPartner';shutil.copytree(base/'ContractPartner',partner)
  shutil.copy2('/tmp/facet-clean-room/official/robocode-tankroyale-bot-api-1.4.0.jar',partner/'api.jar')
  prefixes=('FACET_','AISURVIVOR_','RETCON_','ROBOCODE_','CHALLENGER_')
  env={k:v for k,v in os.environ.items() if not k.startswith(prefixes) and k not in ('SERVER_URL','SERVER_SECRET','BOT_SECRET','JAVA_TOOL_OPTIONS','_JAVA_OPTIONS','JDK_JAVA_OPTIONS')}
  env.update(JAVA_HOME=str(Path(java).parents[1]),PATH=str(Path(java).parent)+os.pathsep+env.get('PATH',''),FACET_SEED=str(seed),FACET_METRICS_PATH=str(run/'FACET_metrics.jsonl'),FACET_TRACE_PATH=str(run/'FACET_trace.jsonl'),FACET_STATE_PATH=str(run/'FACET_model.dat'),FACET_PROBE_PATH=str(run/'FACET_probe.jsonl'))
  order=[bot,partner] if index==0 else [partner,bot]
  item={'seed':seed,'rounds':2,'launchOrder':[p.name for p in order],'status':'running'};report['games'].append(item);save()
  cmd=[java,'-cp','/tmp/facet-build/robocode-tankroyale-runner-1.4.0.jar',str(root/'tools/BenchmarkBattle.java'),'2','240',str(run/'battle.json'),*map(str,order)]
  benchmark.run_process(cmd,root,env,run/'runner.log',250)
  time.sleep(.2)
  result=json.loads((run/'battle.json').read_text());metrics=[json.loads(x) for x in (run/'FACET_metrics.jsonl').read_text().splitlines()]
  if result['numberOfRounds']!=2 or len(metrics)!=2:raise ValueError('incomplete game')
  item.update(status='complete',official=result);save()
 report['status']='complete'
except BaseException as e:report.update(status='invalid',error=str(e));raise
finally:report['elapsedSeconds']=time.monotonic()-begin;save()
