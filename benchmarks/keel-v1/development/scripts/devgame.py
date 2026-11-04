"""Exploratory local dev games only (not registered evidence). Usage: devgame.py BOTDIR ENEMY ROUNDS PORT OUT"""
import os,sys,shutil,subprocess,json,gzip
from pathlib import Path
ROOT=Path('/home/user/robocode'); sys.path.insert(0,str(ROOT/'tools'))
import benchmark, retcon, strong_melee_opponents as native
CACHE=ROOT/'.benchmark-cache'
bot,enemy,rounds,port,out=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),Path(sys.argv[5])
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True); stage=out/'stage'; stage.mkdir()
own=stage/next(bot.glob('*.json')).stem; shutil.copytree(bot,own)
env={k:os.environ[k] for k in ('PATH','HOME','JAVA_HOME','LANG','JAVA_TOOL_OPTIONS') if k in os.environ}
env.update({k:v for k,v in os.environ.items() if k.startswith(('KEEL_','MERIDIAN_'))})
env['PYTHONDONTWRITEBYTECODE']='1'
if enemy=='diamond':
    base=CACHE/'strong-python/67b68c22e41c18281a3fce9ef638e3c23a0b47cf'; src=base/'source'
    env.update(native.stage(src,stage,base/'venv/bin/python'))
    other=stage/'opponents/diamond-port'; benchmark.isolate_opponent(src/'bots/ports/diamond-port',other)
elif enemy=='retcon':
    src,api,rc=retcon.prepare(CACHE); other=stage/'Retcon'; benchmark.isolate_opponent(src,other); env['RUMBLE_JAVA_CACHE']=str(api)
elif enemy=='ais':
    other=stage/'AISurvivor'; shutil.copytree(ROOT/'AISurvivor',other)
    env.update(AISURVIVOR_MODEL_PATH=str(out/'ais-model.dat'),AISURVIVOR_METRICS_PATH=str(out/'ais.jsonl'),AISURVIVOR_SEED='7',AISURVIVOR_MODE='train')
else:
    other=Path(enemy)
    other2=stage/other.name; benchmark.isolate_opponent(other,other2); other=other2
cmd=['java','-cp',str(CACHE/'robocode-tankroyale-runner-1.4.0.jar'),str(Path(__file__).parent/'DevBattle.java'),str(port),str(rounds),str(out),str(own),str(other)]
with open(out/'runner.log','w') as log:
    r=subprocess.run(cmd,cwd=stage,env=env,stdout=subprocess.PIPE,stderr=log,text=True,timeout=3600)
print(r.stdout.strip())
