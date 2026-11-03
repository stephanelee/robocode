"""One descriptive Retcon/Adaptive Prime/Diamond Port four-bot baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from unittest.mock import patch
import benchmark
import compare
import melee_default_baseline as measured
import retcon
import retcon_stats
import strong_melee_opponents as native

SOURCE='1e31cceca5cf7f0ffe59dcbdc5f71015a4945871'
JAR=measured.JAR
REGISTRATION='19ed4d758d7d72b6f2a4fba965f45bb1c33dcfe3'
PLAN='docs/improvements/strong-melee-baseline.md'
LAYOUT='tools/fixtures/strong-melee-baseline/games.json'
OPPONENTS=['Retcon','adaptive-prime','diamond-port']


def expected_layout():return [dict(gameId=f'strong946-g{i:02d}',seed=20262600+i) for i in (1,2,3)]


def play(args,directory,runner,source,python,retcon_source,api,game):
    args.seed=game['seed'];args.timeout=900;args.training_rounds=20;args.require_metrics=True;args.require_retcon_stats=True
    args.retcon_source=retcon_source;run=directory/'Melee-train'
    args.opponent_env=native.stage(source,run,python);args.opponent_env['RUMBLE_JAVA_CACHE']=str(api)
    model=directory/'model.dat'
    with patch.dict(os.environ,native.clean_environment(os.environ),clear=True):
        battle=benchmark.battle(args,directory,runner,source,'Melee',OPPONENTS,'train',20,model)
    expected=[json.loads((run/'AISurvivor/AISurvivor.json').read_bytes())]
    expected += [json.loads((run/'opponents'/name/(name+'.json')).read_bytes()) for name in OPPONENTS]
    if (len({r['id'] for r in battle['results']})!=4 or sorted((r['name'],r['version']) for r in battle['results'])
            != sorted((r['name'],r['version']) for r in expected)):raise ValueError('Official stronger melee roster differs')
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_bytes().splitlines()]
    return dict(**game,battle=battle,scoreShare=compare.score_share(battle),selection=measured.selection(rows),
        endpoint=measured.reusable.validate_endpoint(model,20),executedPythonSourceHashes=native.verify_staged(run,['adaptive-prime','diamond-port']),
        launchEnvironment=dict(args.opponent_env,AISURVIVOR_MODE='train',AISURVIVOR_SEED=str(game['seed']),
            AISURVIVOR_MODEL_PATH=str(model),AISURVIVOR_METRICS_PATH=str(run/'metrics.jsonl'),duelOverride='unset'))


def summarize(games):
    if [{k:g[k] for k in ('gameId','seed')} for g in games]!=expected_layout():raise ValueError('Complete fixed three-game layout required')
    totals={k:sum(g['battle']['metrics'][k] for g in games) for k in ('turns','shotsFired','bulletHits','damageDealt','damageTaken','wallCollisions','botCollisions','skippedTurns')}
    names=[r['name'] for r in games[0]['battle']['results']]
    opponents={}
    for name in names:
        rows=[next(r for r in g['battle']['results'] if r['name']==name) for g in games]
        shares=[r['totalScore']/sum(v['totalScore'] for v in g['battle']['results']) for r,g in zip(rows,games)]
        opponents[name]=dict(officialGameRanks=[r['rank'] for r in rows],equalGameMeanShare=statistics.mean(shares),officialFirstPlaces=sum(r['firstPlaces'] for r in rows),scores=[r['totalScore'] for r in rows])
    return dict(disposition='descriptive-baseline',rounds=60,opponents=opponents,metrics=totals,
        hitRate=totals['bulletHits']/totals['shotsFired'] if totals['shotsFired'] else None,
        actionCounts=[sum(g['selection']['actionCounts'][i] for g in games) for i in range(6)],
        duelEligibleTurns=sum(g['selection']['duelEligibleTurns'] for g in games),forcedDuelTurns=sum(g['selection']['forcedDuelTurns'] for g in games))


def main(args):
    args.output,args.bot_dir,args.cache=args.output.resolve(),args.bot_dir.resolve(),args.cache.resolve()
    if args.output.exists():raise ValueError('Never replace a measured attempt')
    args.output.mkdir(parents=True)
    plan=subprocess.check_output(['git','show',REGISTRATION+':'+PLAN],cwd=benchmark.ROOT)
    layout=subprocess.check_output(['git','show',REGISTRATION+':'+LAYOUT],cwd=benchmark.ROOT)
    locked=subprocess.check_output(['git','show',REGISTRATION+':tools/opponents/strong-melee.lock.json'],cwd=benchmark.ROOT)
    if native.LOCK.read_bytes()!=locked or json.loads(layout)['games']!=expected_layout():raise ValueError('Immutable source/runtime/layout registration differs')
    if subprocess.check_output(['git','diff','--name-only',SOURCE,'HEAD','--','src/main/java','src/test/java','AISurvivor',
            'tools/benchmark.py','tools/BenchmarkBattle.java','tools/retcon.py','tools/retcon_stats.py','tools/melee_default_baseline.py'],cwd=benchmark.ROOT).strip():raise ValueError('Frozen bot/acquisition helper changed')
    if benchmark.digest(args.bot_dir/'AISurvivor.jar')!=JAR:raise ValueError('Frozen AIS JAR differs')
    (args.output/'registered-plan.md').write_bytes(plan);(args.output/'registered-games.json').write_bytes(layout);(args.output/'python-opponents.lock.json').write_bytes(locked)
    report=dict(status='running',botSource=SOURCE,botJarSHA256=JAR,registrationCommit=REGISTRATION,
        executionSource=subprocess.check_output(['git','rev-parse','HEAD'],cwd=benchmark.ROOT,text=True).strip(),
        planSHA256=hashlib.sha256(plan).hexdigest(),layoutSHA256=hashlib.sha256(layout).hexdigest(),runtimeOS=retcon_stats.os_provenance(),
        setup=dict(arenaWidth=1000,arenaHeight=1000,participants=4,turnTimeoutMicros=100000,roundsPerGame=20),games=[])
    output=args.output/'baseline.json'
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        report['runtimeJava']=subprocess.check_output([args.java,'-XshowSettings:properties','-version'],stderr=subprocess.STDOUT,text=True,timeout=15)
        source,python,provenance=native.prepare(args.cache,args.python);report['pythonOpponents']=provenance;save()
        opponent,api,retcon_provenance=retcon.prepare(args.cache);report['retconProvenance']=retcon_provenance
        notices=args.output/'licenses';notices.mkdir()
        shutil.copyfile(source/'LICENSE',notices/'Apache-2.0.txt');shutil.copyfile(source/'bots/ports/diamond-port/LICENSE',notices/'Diamond-Zlib.txt')
        runner=benchmark.download(args.cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar']);report['runnerSHA256']=benchmark.digest(runner)
        for game in expected_layout():
            report['inProgressGame']=game['gameId'];save();directory=args.output/game['gameId'];directory.mkdir()
            report['games'].append(play(args,directory,runner,source,python,opponent,api,game));del report['inProgressGame'];save()
        report.update(status='complete',summary=summarize(report['games']));print(json.dumps(report['summary'],indent=2))
    except BaseException as error:report.update(status='invalid',error=str(error));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','bot-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--java',required=True);parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--cache',type=Path,default=benchmark.ROOT/'.benchmark-cache')
    main(parser.parse_args())
