"""Three descriptive default-TRAIN sample melees; no gameplay changes or acceptance rule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from unittest.mock import patch
import benchmark
import compare
import coherent_duel_v2_experiment as reusable
import retcon_stats

SOURCE='e4e4c94a45c770a494f6a60f6b98456e215bee42'
JAR='9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6'
REGISTRATION='266261ca6de4b7392f01d2b58b12e3013a0e76cb'
PLAN='docs/improvements/melee-default-baseline.md'
LAYOUT='tools/fixtures/melee-default-baseline/games.json'
OPPONENTS=['Target','SpinBot','Walls']


def expected_layout():return [dict(gameId=f'melee946-g{i:02d}',seed=20262500+i) for i in (1,2,3)]


def selection(rows):
    counts=[0]*6;eligible=forced=0
    for row in rows:
        values=row.get('actionCounts');e=row.get('duelEligibleTurns');f=row.get('forcedDuelTurns')
        if (row.get('duelPolicy')!='controller' or not isinstance(values,list) or len(values)!=6
                or any(type(v) is not int or v<0 for v in values) or type(e) is not int or type(f) is not int
                or e<0 or f!=e or values[0]<f):raise ValueError('Invalid default selection telemetry')
        counts=[a+b for a,b in zip(counts,values)];eligible+=e;forced+=f
    return dict(actionCounts=counts,duelEligibleTurns=eligible,forcedDuelTurns=forced)


def play(args,directory,runner,samples,game):
    args.seed=game['seed'];args.timeout=900;args.training_rounds=20
    args.require_metrics=True;args.opponent_env={};args.retcon_source=None
    model=directory/'model.dat'
    # Only this synchronous child invocation sees the clean default environment.
    with patch.dict(os.environ,reusable.clean_environment(os.environ),clear=True):
        result=benchmark.battle(args,directory,runner,samples,'Melee',OPPONENTS,'train',20,model)
    run_dir=directory/'Melee-train'
    expected=[json.loads((run_dir/'AISurvivor/AISurvivor.json').read_bytes())]
    expected += [json.loads((run_dir/'opponents'/name/(name+'.json')).read_bytes()) for name in OPPONENTS]
    actual=result['results']
    if (len({row['id'] for row in actual})!=4 or sorted((r['name'],r['version']) for r in actual)
            != sorted((r['name'],r['version']) for r in expected)):raise ValueError('Official four-bot roster differs')
    rows=[json.loads(line) for line in (run_dir/'metrics.jsonl').read_bytes().splitlines()]
    return dict(**game,battle=result,scoreShare=compare.score_share(result),selection=selection(rows),
                endpoint=reusable.validate_endpoint(model,20),
                launchEnvironment=dict(AISURVIVOR_MODE='train',AISURVIVOR_SEED=str(game['seed']),
                    AISURVIVOR_MODEL_PATH=str(model),AISURVIVOR_METRICS_PATH=str(run_dir/'metrics.jsonl'),duelOverride='unset'))


def summarize(games):
    if [{k:g[k] for k in ('gameId','seed')} for g in games]!=expected_layout():raise ValueError('Complete fixed three-game layout required')
    totals={k:sum(g['battle']['metrics'][k] for g in games) for k in ('turns','shotsFired','bulletHits','damageDealt','damageTaken','wallCollisions','botCollisions','skippedTurns')}
    return dict(disposition='descriptive-baseline',rounds=60,officialGameRanks=[g['battle']['rank'] for g in games],
        meanGameRank=statistics.mean(g['battle']['rank'] for g in games),equalGameMeanShare=statistics.mean(g['scoreShare'] for g in games),
        officialFirstPlaces=sum(g['battle']['firstPlaces'] for g in games),metrics=totals,
        hitRate=totals['bulletHits']/totals['shotsFired'] if totals['shotsFired'] else None,
        actionCounts=[sum(g['selection']['actionCounts'][i] for g in games) for i in range(6)],
        duelEligibleTurns=sum(g['selection']['duelEligibleTurns'] for g in games),forcedDuelTurns=sum(g['selection']['forcedDuelTurns'] for g in games))


def main(args):
    args.output,args.bot_dir,args.cache=args.output.resolve(),args.bot_dir.resolve(),args.cache.resolve()
    if args.output.exists():raise ValueError('Never replace an attempt')
    args.output.mkdir(parents=True)
    plan=subprocess.check_output(['git','show',REGISTRATION+':'+PLAN],cwd=benchmark.ROOT)
    layout=subprocess.check_output(['git','show',REGISTRATION+':'+LAYOUT],cwd=benchmark.ROOT)
    games=json.loads(layout)['games']
    if games!=expected_layout():raise ValueError('Immutable layout differs')
    if subprocess.check_output(['git','diff','--name-only',SOURCE,'HEAD','--','src/main/java','src/test/java','AISurvivor',
            'tools/benchmark.py','tools/BenchmarkBattle.java','tools/coherent_duel_v2_experiment.py'],cwd=benchmark.ROOT).strip():raise ValueError('Frozen source/helper changed')
    if benchmark.digest(args.bot_dir/'AISurvivor.jar')!=JAR:raise ValueError('Frozen JAR differs')
    (args.output/'registered-plan.md').write_bytes(plan);(args.output/'registered-games.json').write_bytes(layout)
    report=dict(status='running',botSource=SOURCE,botJarSHA256=JAR,registrationCommit=REGISTRATION,
        executionSource=subprocess.check_output(['git','rev-parse','HEAD'],cwd=benchmark.ROOT,text=True).strip(),
        planSHA256=hashlib.sha256(plan).hexdigest(),layoutSHA256=hashlib.sha256(layout).hexdigest(),runtimeOS=retcon_stats.os_provenance(),
        botAssetHashes={name:benchmark.digest(args.bot_dir/name) for name in ('AISurvivor.jar','AISurvivor.json','AISurvivor.sh','AISurvivor.cmd')},
        setup=dict(arenaWidth=1000,arenaHeight=1000,participants=4,turnTimeoutMicros=100000,roundsPerGame=20),games=[])
    output=args.output/'baseline.json'
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        java=subprocess.run([args.java,'-XshowSettings:properties','-version'],capture_output=True,text=True,check=True,timeout=15)
        report['runtimeJava']=java.stdout+java.stderr
        runner=benchmark.download(args.cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        archive=benchmark.download(args.cache,'sample-bots-Java-1.4.0.zip',benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples=args.output/'samples';benchmark.extract(archive,samples)
        report.update(runnerSHA256=benchmark.digest(runner),sampleAssetSHA256=benchmark.digest(archive))
        for game in games:
            report['inProgressGame']=game['gameId'];save();directory=args.output/game['gameId'];directory.mkdir()
            report['games'].append(play(args,directory,runner,samples,game));del report['inProgressGame'];save()
        report.update(status='complete',summary=summarize(report['games']));print(json.dumps(report['summary'],indent=2))
    except BaseException as error:report.update(status='invalid',error=str(error));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','bot-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--java',required=True);parser.add_argument('--cache',type=Path,default=benchmark.ROOT/'.benchmark-cache')
    main(parser.parse_args())
