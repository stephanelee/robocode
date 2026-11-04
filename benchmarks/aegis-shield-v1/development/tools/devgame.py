"""Development-only (unregistered) game vs pinned Diamond Port. Not benchmark evidence."""
import sys, json, os, shutil, argparse
from pathlib import Path
sys.path.insert(0,'/home/user/robocode/tools')
import benchmark, strong_melee_opponents as native
p=argparse.ArgumentParser();p.add_argument('--bot',type=Path,required=True);p.add_argument('--rounds',type=int,default=10)
p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--opp',default='diamond-port')
p.add_argument('--prefix',default='CHALLENGER_');p.add_argument('--oppdir',default=None)
a=p.parse_args();a.out=a.out.resolve();a.bot=a.bot.resolve()
cache=Path('/home/user/robocode/.benchmark-cache').resolve()
base=cache/'strong-python'
venv_py=None
import glob
c=glob.glob(str(base/'*/venv/bin/python'))
lock=native.manifest();src=base/lock['commit']/'source'
if c: venv_py=Path(c[0])
else:
    src,venv_py,prov=native.prepare(cache,'/usr/local/bin/python3.13')
runner=benchmark.download(cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
a.out.mkdir(parents=True);run=a.out/'run';run.mkdir()
bot=run/a.bot.name;shutil.copytree(a.bot,bot)
for s in bot.glob('*.sh'):s.chmod(0o755)
env=native.clean_environment(os.environ);env={k:v for k,v in env.items() if not k.startswith(a.prefix)}
env.update(native.stage(src,run,str(venv_py)))
opp=run/'opponents'/a.opp;benchmark.isolate_opponent(Path(a.oppdir) if a.oppdir else src/'bots/ports'/a.opp,opp)
if a.oppdir: env['DIAMOND_DEV_LOG']=str(a.out/'diamond.jsonl')
env[a.prefix+'SEED']=str(a.seed);env[a.prefix+'TRACE_PATH']=str(a.out/'trace.jsonl');env[a.prefix+'MODEL_PATH']=str(a.out/'model.json');env[a.prefix+'METRICS_PATH']=str(a.out/'metrics.jsonl')
official=run/'battle.json'
cmd=['java','-cp',str(runner),'/home/user/robocode/tools/BenchmarkBattle.java',str(a.rounds),'3000',str(official),str(bot),str(opp)]
benchmark.run_process(cmd,Path('/home/user/robocode'),env,a.out/'runner.log',3100)
d=json.loads(official.read_bytes())
tot=sum(r['totalScore'] for r in d['results'])
for r in d['results']:print(r['name'],r['totalScore'],round(r['totalScore']/tot*100,2),'%','1st:',r['firstPlaces'],'bd',r['bulletDamage'],'surv',r['survival'])
