"""User-authorized fresh redo; unchanged frozen bot and confirmation rules."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import benchmark
import coherent_duel_v2_experiment as harness
import compare
import retcon
import retcon_stats

SOURCE='94640028a1b5bf072713bc19ea1b608d14f9d26d'
JAR='9679fc5ee27b68877f0b6d89a2104ffff115d6251ec44d3036fdd1f1d64988a6'
REGISTRATION='2645d2359edcd24eac5b569284cc34337f51194c'
PLAN='docs/improvements/final-default-confirmation-redo.md'
LAYOUT='tools/fixtures/final-default-confirmation-redo/games.json'


def expected_layout():
    return [dict(gameId=f'dc946r2-b{b}-g{i:02d}',batch=b,index=i,seed=20262400+(b-1)*10+i)
            for b in (1,2) for i in range(1,6)]


def registration():
    plan=subprocess.check_output(['git','show',REGISTRATION+':'+PLAN],cwd=benchmark.ROOT)
    layout=subprocess.check_output(['git','show',REGISTRATION+':'+LAYOUT],cwd=benchmark.ROOT)
    games=json.loads(layout)['games']
    if games!=expected_layout():raise ValueError('Immutable registration differs from fixed fresh budget')
    return games,plan,layout


def decision(games):
    if [{k:g[k] for k in ('gameId','batch','index','seed')} for g in games]!=expected_layout():
        raise ValueError('Exactly the ten registered games in order are required')
    battles=[]
    for game in games:
        if (game['rounds']!=20 or game['kind']!='Retcon' or game['arms']!=['candidate']
                or set(game['battles'])!={'candidate'}):raise ValueError('Changed roster or budget')
        battle=game['battles']['candidate'];battles.append(battle)
        if battle['rounds']!=20 or type(battle['firstPlaces']) is not int or not 0<=battle['firstPlaces']<=20:
            raise ValueError('Invalid official round counts')
    shares=[compare.score_share(b) for b in battles]
    mean=statistics.mean(shares);sd=statistics.stdev(shares)
    lower=mean-1.8331129326536335*sd/math.sqrt(10)
    wins=sum(b['firstPlaces'] for b in battles)
    batch_means={str(b):statistics.mean(compare.score_share(g['battles']['candidate']) for g in games if g['batch']==b) for b in (1,2)}
    evidence=all(b['metrics']['roundsRecorded']==20 and b['metrics']['skippedTurns']==0
        and b['metrics']['checkpointLoaded'] is False and b['endpoint']['totalRounds']==20
        and b['mechanism']['forcedDuelTurns']>0 and g['native']['status']=='complete' for g,b in zip(games,battles))
    checks=dict(meanShare=mean>.5,majorityRoundWins=wins>100,majorityGames=sum(s>.5 for s in shares)>5,
                bothBatches=all(m>.5 for m in batch_means.values()),gameMeanTLower95=lower>.5,validEvidence=evidence)
    return dict(status='confirmed' if all(checks.values()) else 'not-confirmed',checks=checks,
        meanShare=mean,wins=wins,rounds=200,gamesAboveHalf=sum(s>.5 for s in shares),gameShares=shares,
        batchMeans=batch_means,gameShareSD=sd,gameMeanTLower95=lower,descriptiveRoundWinWilson95=compare.wilson(wins,200),
        limitation='Frozen default TRAIN, pinned Retcon and ten games only. Game-level t assumptions are approximate; adaptive rounds are dependent. No broad tournament or 200-IID-trial claim.')


def run(args):
    args.output,args.cache,args.bot_dir=args.output.resolve(),args.cache.resolve(),args.bot_dir.resolve()
    if args.output.exists():raise ValueError('Never replace an earlier attempt')
    args.output.mkdir(parents=True)
    games,plan,layout=registration()
    if subprocess.check_output(['git','diff','--name-only',SOURCE,'HEAD','--','src/main/java','src/test/java','AISurvivor',
            'tools/coherent_duel_v2_experiment.py','tools/benchmark.py','tools/compare.py','tools/default_duel_policy_screen.py',
            'tools/retcon.py','tools/retcon_stats.py','tools/BenchmarkBattle.java'],cwd=benchmark.ROOT).strip():
        raise ValueError('Frozen gameplay/assets or reused acquisition helpers changed')
    (args.output/'registered-plan.md').write_bytes(plan);(args.output/'registered-games.json').write_bytes(layout)
    report=dict(status='running',batch=args.batch,botSource=SOURCE,botJarSHA256=JAR,registrationCommit=REGISTRATION,
        planSHA256=hashlib.sha256(plan).hexdigest(),layoutSHA256=hashlib.sha256(layout).hexdigest(),
        executionSource=subprocess.check_output(['git','rev-parse','HEAD'],cwd=benchmark.ROOT,text=True).strip(),
        runtimeOS=retcon_stats.os_provenance(),games=[])
    output=args.output/'batch.json'
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        if benchmark.digest(args.bot_dir/'AISurvivor.jar')!=JAR:raise ValueError('Frozen JAR differs')
        java=subprocess.run([args.java,'-XshowSettings:properties','-version'],capture_output=True,text=True,check=True,timeout=15)
        report['runtimeJava']=java.stdout+java.stderr
        opponent,api,provenance=retcon.prepare(args.cache)
        runner=benchmark.download(args.cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        report.update(opponentProvenance=provenance,runnerSHA256=benchmark.digest(runner))
        for registered in (g for g in games if g['batch']==args.batch):
            report['inProgressGame']=registered['gameId'];save()
            game=dict(**registered,kind='Retcon',block=(args.batch-1)*5+registered['index'],rounds=20,
                      arms=['candidate'],seeds=dict(candidate=registered['seed']))
            report['games'].append(harness.play(args,game,dict(candidate=args.bot_dir),runner,None,opponent,api))
            del report['inProgressGame'];save()
        report['status']='complete'
    except BaseException as error:report.update(status='invalid',error=str(error));raise
    finally:save()


def aggregate(args):
    args.input,args.output=args.input.resolve(),args.output.resolve()
    if args.output.exists():raise ValueError('Never replace an earlier aggregate')
    games,plan,layout=registration();paths=sorted(args.input.rglob('batch.json'))
    reports=[json.loads(p.read_bytes()) for p in paths]
    if len(reports)!=2 or {r['batch'] for r in reports}!={1,2}:raise ValueError('Both distinct five-game batches required')
    if len({r['executionSource'] for r in reports})!=1:raise ValueError('Execution sources differ')
    for path,report in zip(paths,reports):
        if (report['status']!='complete' or report['botSource']!=SOURCE or report['botJarSHA256']!=JAR
                or report['registrationCommit']!=REGISTRATION or report['planSHA256']!=hashlib.sha256(plan).hexdigest()
                or report['layoutSHA256']!=hashlib.sha256(layout).hexdigest()):raise ValueError('Invalid batch provenance')
        expected=[g for g in games if g['batch']==report['batch']]
        if [{k:g[k] for k in ('gameId','batch','index','seed')} for g in report['games']]!=expected:
            raise ValueError('Batch games do not match immutable registration')
        for game in report['games']:
            directory=path.parent/game['gameId'];bot=directory/'candidate/AISurvivor';battle=game['battles']['candidate']
            rows=[json.loads(line) for line in (bot/'metrics.jsonl').read_bytes().splitlines()]
            benchmark.validate_metrics(rows,20,'train',game['seed'],20)
            official=json.loads((directory/'battle.json').read_bytes())
            identities=[r for r in official['results'] if r['name']==battle['bot']['name']]
            if (official['numberOfRounds']!=20 or len(identities)!=1
                    or any(battle[k]!=v for k,v in identities[0].items())):
                raise ValueError('Reported score/wins/identity differ from official result')
            if (benchmark.summarize_metrics(rows)!=battle['metrics'] or harness.policy.mechanism(rows)!=battle['mechanism']
                    or harness.validate_endpoint(bot/'model.dat',20)!=battle['endpoint']
                    or official['results']!=battle['results']):
                raise ValueError('Raw evidence differs from batch report')
    actual=sorted([g for r in reports for g in r['games']],key=lambda g:(g['batch'],g['index']))
    result=dict(status='complete',botSource=SOURCE,botJarSHA256=JAR,registrationCommit=REGISTRATION,games=actual,
                executionSource=reports[0]['executionSource'],decision=decision(actual))
    args.output.mkdir(parents=True)
    (args.output/'confirmation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['decision'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--input',type=Path)
    parser.add_argument('--batch',type=int,choices=(1,2));parser.add_argument('--bot-dir',type=Path)
    parser.add_argument('--java');parser.add_argument('--cache',type=Path,default=benchmark.ROOT/'.benchmark-cache')
    args=parser.parse_args()
    if args.input:aggregate(args)
    elif args.batch and args.bot_dir and args.java:run(args)
    else:parser.error('Use --input or --batch plus --bot-dir and --java')
