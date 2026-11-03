"""Fixed descriptive default-TRAIN duels against two pinned native Python bots."""
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
import melee_default_baseline as telemetry
import retcon_stats
import strong_melee_opponents as opponents

SOURCE='fca6d2ff1101fcf4dd07908cce8df9d18b32dc63'
JAR='9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6'
REGISTRATION='b4fdd57a2c905e180cb16eb21ddce0fe093053d4'
CONTINUATION='c573055dd78784d8464a62027cc7f17321ec0b6c'
CONTINUATION_PLAN='docs/improvements/advanced-python-duels-continuation.md'
PLAN='docs/improvements/advanced-python-duels.md'
LAYOUT='tools/fixtures/advanced-python-duels/games.json'
LOCK='tools/opponents/strong-melee.lock.json'
NAMES={'diamond-port':('Diamond Port','1.8.28'),'adaptive-prime':('Adaptive Prime','1.0')}


def layout():
    return [dict(gameId=f'duelpy946-{label}-g{i:02d}',opponent=name,index=i,seed=20262700+offset+i)
            for name,label,offset in [('diamond-port','diamond',0),('adaptive-prime','adaptive',10)] for i in (1,2,3)]


def summary(games):
    totals={key:sum(g['battle']['metrics'][key] for g in games)
            for key in ('turns','shotsFired','bulletHits','damageDealt','damageTaken','wallCollisions','botCollisions','skippedTurns')}
    return dict(disposition='descriptive-baseline',rounds=sum(g['battle']['rounds'] for g in games),
        officialFirstPlaces=sum(g['battle']['firstPlaces'] for g in games),
        officialGameRanks=[g['battle']['rank'] for g in games],
        equalGameMeanShare=statistics.mean(g['scoreShare'] for g in games),metrics=totals,
        hitRate=totals['bulletHits']/totals['shotsFired'] if totals['shotsFired'] else None,
        duelEligibleTurns=sum(g['selection']['duelEligibleTurns'] for g in games),
        forcedDuelTurns=sum(g['selection']['forcedDuelTurns'] for g in games))


def play(args,source,python,runner,game):
    directory=args.output/game['gameId'];directory.mkdir()
    run_dir=directory/'Duel-train'
    env=opponents.stage(source,run_dir,python)
    args.seed=game['seed'];args.timeout=900;args.training_rounds=20
    args.require_metrics=True;args.opponent_env=env;args.retcon_source=None
    model=directory/'model.dat'
    with patch.dict(os.environ,opponents.clean_environment(os.environ),clear=True):
        result=benchmark.battle(args,directory,runner,source,'Duel',[game['opponent']],'train',20,model)
    expected=json.loads((run_dir/'AISurvivor/AISurvivor.json').read_bytes())
    actual=result['results']
    if (len({row['id'] for row in actual})!=2 or sorted((r['name'],r['version']) for r in actual)
            != sorted([(expected['name'],expected['version']),NAMES[game['opponent']]])):
        raise ValueError('Official duel roster differs from frozen identities')
    rows=[json.loads(line) for line in (run_dir/'metrics.jsonl').read_bytes().splitlines()]
    return dict(**game,battle=result,scoreShare=compare.score_share(result),selection=telemetry.selection(rows),
        endpoint=reusable.validate_endpoint(model,20),sourceHashes=opponents.verify_staged(run_dir,[game['opponent']]),
        launchEnvironment=dict(AISURVIVOR_MODE='train',AISURVIVOR_SEED=str(game['seed']),
            AISURVIVOR_MODEL_PATH=str(model),AISURVIVOR_METRICS_PATH=str(run_dir/'metrics.jsonl'),duelOverride='unset',
            opponent=env))


def main(args):
    args.output,args.bot_dir,args.cache=args.output.resolve(),args.bot_dir.resolve(),args.cache.resolve()
    if args.output.exists():raise ValueError('Never replace a prior attempt')
    args.output.mkdir(parents=True)
    plan=subprocess.check_output(['git','show',REGISTRATION+':'+PLAN],cwd=benchmark.ROOT)
    registered=subprocess.check_output(['git','show',REGISTRATION+':'+LAYOUT],cwd=benchmark.ROOT)
    continued=subprocess.check_output(['git','show',CONTINUATION+':'+CONTINUATION_PLAN],cwd=benchmark.ROOT)
    games=json.loads(registered)['games']
    if games!=layout():raise ValueError('Immutable duel population differs')
    games=[g for g in games if g['opponent']==args.opponent]
    if len(games)!=3:raise ValueError('Exactly three games for this opponent required')
    if subprocess.check_output(['git','diff','--name-only',SOURCE,'HEAD','--','src/main/java','src/test/java','AISurvivor',
            'tools/benchmark.py','tools/BenchmarkBattle.java','tools/coherent_duel_v2_experiment.py','tools/melee_default_baseline.py'],cwd=benchmark.ROOT).strip():
        raise ValueError('Frozen production assets or reused battle helpers changed')
    if benchmark.digest(args.bot_dir/'AISurvivor.jar')!=JAR:raise ValueError('Frozen production JAR differs')
    if subprocess.check_output(['git','diff','--name-only',CONTINUATION,'HEAD','--',
            'tools/strong_melee_opponents.py',LOCK],cwd=benchmark.ROOT).strip():
        raise ValueError('Reviewed corrected source loader or lock changed')
    (args.output/'registered-plan.md').write_bytes(plan);(args.output/'registered-games.json').write_bytes(registered)
    (args.output/'registered-continuation.md').write_bytes(continued)
    report=dict(status='running',botSource=SOURCE,gameplaySource='94640028a1b5bf072713bc19ea1b608d14f9d26d',
        botJarSHA256=JAR,opponent=args.opponent,registrationCommit=REGISTRATION,
        planSHA256=hashlib.sha256(plan).hexdigest(),layoutSHA256=hashlib.sha256(registered).hexdigest(),
        lockSHA256=benchmark.digest(benchmark.ROOT/LOCK),
        continuationCommit=CONTINUATION,continuationSHA256=hashlib.sha256(continued).hexdigest(),
        loaderSHA256=benchmark.digest(benchmark.ROOT/'tools/strong_melee_opponents.py'),
        executionSource=subprocess.check_output(['git','rev-parse','HEAD'],cwd=benchmark.ROOT,text=True).strip(),
        runtimeOS=retcon_stats.os_provenance(),setup=dict(arenaWidth=800,arenaHeight=600,participants=2,
            turnTimeoutMicros=100000,roundsPerGame=20),
        botAssetHashes={name:benchmark.digest(args.bot_dir/name) for name in reusable.ASSETS},games=[])
    output=args.output/'duels.json'
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        java=subprocess.run([args.java,'-XshowSettings:properties','-version'],capture_output=True,text=True,check=True,timeout=15)
        report['runtimeJava']=java.stdout+java.stderr
        source,python,provenance=opponents.prepare(args.cache,args.python)
        report['opponentProvenance']=provenance
        runner=benchmark.download(args.cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        report['runnerSHA256']=benchmark.digest(runner)
        for game in games:
            report['inProgressGame']=game['gameId'];save()
            report['games'].append(play(args,source,python,runner,game));del report['inProgressGame'];save()
        report.update(status='complete',summary=summary(report['games']))
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:report.update(status='invalid',error=str(error));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','bot-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--opponent',choices=NAMES,required=True)
    parser.add_argument('--java',required=True);parser.add_argument('--python',required=True)
    parser.add_argument('--cache',type=Path,default=benchmark.ROOT/'.benchmark-cache')
    main(parser.parse_args())
