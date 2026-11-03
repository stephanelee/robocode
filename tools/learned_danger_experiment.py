"""One fixed screen then conditional fresh confirmation of the frozen movement-only candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
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

# Filled from the root-owned immutable registration before exact premeasurement review.
REGISTRATION='fe286963d8b1c7c4151357c9c5a73b6a20d614d6'
PLAN='docs/improvements/learned-duel-danger.md'
PROTOCOL='tools/fixtures/learned-duel-danger/protocol.json'
BASELINE_SOURCE='12348fb656c64fa9eb36f400b28aa96cbb5fd383'
CANDIDATE_SOURCE='768f6647dab0265475c5812825c8186c2d01f2c3'
JARS=dict(previous='21c5c3d2cb4007e3e0603556af313df7f5717732e384f039b005e9e72b2daecd',candidate='b2da09ef774f8c699d4e12e7a69ae77d202914bbe8048317c61bb4fcefe7d5c8')
OPS=('diamond-port','adaptive-prime','Retcon')
LABELS={'diamond-port':'diamond','adaptive-prime':'adaptive','Retcon':'retcon'}


def layout(stage):
    result=[]
    if stage=='screen':
        for i,op in enumerate(OPS):
            for arm in (('previous','candidate') if i%2==0 else ('candidate','previous')):
                result.append(dict(gameId=f'danger-v1-screen-{LABELS[op]}-{arm}',stage=stage,opponent=op,arm=arm,block=1,seed=20262901+i,rounds=10))
    elif stage=='confirmation':
        for i,op in enumerate(OPS):
            for block in (1,2):
                for arm in (('previous','candidate') if (i+block)%2 else ('candidate','previous')):
                    result.append(dict(gameId=f'danger-v1-confirm-{LABELS[op]}-b{block}-{arm}',stage=stage,opponent=op,arm=arm,block=block,seed=20262911+2*i+block-1,rounds=20))
    elif stage=='samples':
        for i,op in enumerate(('Target','SpinBot','Walls')):
            for arm in (('previous','candidate') if i%2==0 else ('candidate','previous')):
                result.append(dict(gameId=f'danger-v1-sample-{op}-{arm}',stage=stage,opponent=op,arm=arm,block=1,seed=20262921,rounds=5))
    else:raise ValueError('Unknown fixed stage')
    return result


def opponent_result(rows,op):
    selected=[r for r in rows if r['opponent']==op]
    blocks=[{arm:next(r['battle'] for r in selected if r['block']==block and r['arm']==arm) for arm in ('previous','candidate')}
            for block in sorted({r['block'] for r in selected})]
    deltas=[compare.score_share(b['candidate'])-compare.score_share(b['previous']) for b in blocks]
    totals={arm:compare.aggregate(blocks,arm) for arm in ('previous','candidate')}
    return dict(deltas=deltas,meanGain=statistics.mean(deltas),totals=totals,
        candidateShares=[compare.score_share(b['candidate']) for b in blocks])


def validate_layout(rows,stages):
    expected=[g for stage in stages for g in layout(stage)]
    if [{key:r[key] for key in g} for r,g in zip(rows,expected)]!=expected or len(rows)!=len(expected):
        raise ValueError('Actual rows differ from complete preregistered stage/order/budget')


def safety(result):
    previous,candidate=(result['totals'][a] for a in ('previous','candidate'))
    return candidate['skippedTurns']<=previous['skippedTurns'] and candidate['wallsPer1000Turns']<=previous['wallsPer1000Turns']+10


def screen_decision(rows):
    validate_layout(rows,['screen']);results={op:opponent_result(rows,op) for op in OPS}
    ret=results['Retcon'];checks=dict(advancedPositive=all(results[o]['meanGain']>0 for o in OPS[:2]),
        advancedMeanGain=statistics.mean(results[o]['meanGain'] for o in OPS[:2])>=.03,
        retconRelative=ret['meanGain']>=-.05,retconAbsolute=ret['candidateShares'][0]>.5 and ret['totals']['candidate']['wins']>5,
        safety=all(safety(r) for r in results.values()))
    return dict(status='advance' if all(checks.values()) else 'reject-at-screen',checks=checks,opponents=results)


def confirmation_decision(rows):
    validate_layout(rows,['confirmation','samples']);results={op:opponent_result(rows,op) for op in OPS}
    ret=results['Retcon'];samples={op:opponent_result(rows,op) for op in ('Target','SpinBot','Walls')}
    checks=dict(advancedGains=all(results[o]['meanGain']>=.05 and all(d>0 for d in results[o]['deltas']) for o in OPS[:2]),
        retconRelative=ret['meanGain']>=-.05,retconAbsolute=all(s>.5 for s in ret['candidateShares']) and ret['totals']['candidate']['wins']>20,
        safety=all(safety(r) for r in results.values()),
        samples=all(r['meanGain']>=-.05 and r['totals']['candidate']['skippedTurns']==0 for r in samples.values()))
    return dict(status='retain' if all(checks.values()) else 'reject-at-confirmation',checks=checks,opponents=results,samples=samples,
        limitation='Two fresh blocks per opponent are directional replication, not an IID-round superiority or power claim.')


def play(args,game,sources,runner,python_source,python,retcon_source,api,samples):
    directory=args.output/game['gameId'];directory.mkdir();run=directory/'Duel-train';op=game['opponent']
    args.bot_dir=sources[game['arm']];args.seed=game['seed'];args.timeout=900;args.training_rounds=game['rounds']
    args.require_metrics=True;args.require_retcon_stats=op=='Retcon';args.retcon_source=retcon_source
    args.opponent_env=native.stage(python_source,run,python) if op in OPS[:2] else {}
    args.opponent_env['RUMBLE_JAVA_CACHE']=str(api);model=directory/'model.dat'
    with patch.dict(os.environ,native.clean_environment(os.environ),clear=True):
        battle=benchmark.battle(args,directory,runner,python_source if op in OPS[:2] else samples,'Duel',[op],'train',game['rounds'],model)
    descriptor=json.loads((run/'opponents'/op/(op+'.json')).read_bytes())
    own=json.loads((run/'AISurvivor/AISurvivor.json').read_bytes())
    if len(battle['results'])!=2 or len({r['id'] for r in battle['results']})!=2 or len([r for r in battle['results'] if (r['name'],r['version'])==(descriptor['name'],descriptor['version'])])!=1:
        raise ValueError('Official opponent identity differs')
    if len([r for r in battle['results'] if (r['name'],r['version'])==(own['name'],own['version'])])!=1:raise ValueError('Official AIS identity differs')
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_bytes().splitlines()]
    return dict(**game,battle=battle,selection=measured.selection(rows),endpoint=measured.reusable.validate_endpoint(model,game['rounds']),
                botAssetHashes={name:benchmark.digest(run/'AISurvivor'/name) for name in ('AISurvivor.jar','AISurvivor.json','AISurvivor.sh','AISurvivor.cmd')},
                pythonSourceHashes=native.verify_staged(run,[op]) if op in OPS[:2] else {})


def main(args):
    if len(REGISTRATION)!=40 or len(CANDIDATE_SOURCE)!=40 or any(len(v)!=64 for v in JARS.values()):raise ValueError('Root source/JAR/registration must be frozen before execution')
    args.output,args.cache=args.output.resolve(),args.cache.resolve();sources=dict(previous=args.previous.resolve(),candidate=args.candidate.resolve())
    if args.output.exists():raise ValueError('Never replace an attempt')
    args.output.mkdir(parents=True)
    plan=subprocess.check_output(['git','show',REGISTRATION+':'+PLAN],cwd=benchmark.ROOT);(args.output/'registered-plan.md').write_bytes(plan)
    encoded=subprocess.check_output(['git','show',REGISTRATION+':'+PROTOCOL],cwd=benchmark.ROOT);registered=json.loads(encoded)
    if (registered['baseline']!=dict(source=BASELINE_SOURCE,jarSHA256=JARS['previous'])
            or registered['candidate']!=dict(source=CANDIDATE_SOURCE,jarSHA256=JARS['candidate'])):raise ValueError('Source/JAR registration differs')
    if [registered[k] for k in ('mode','duelPolicy','turnTimeoutMicros','arena','maximumRounds','maximumModelEndpoints')]!=['train','controller',100000,[800,600],330,24]:
        raise ValueError('Registered runtime/setup limits differ')
    for stage,expected in dict(screen=dict(blocks=1,rounds=10,opponents=list(OPS),seeds=[20262901,20262902,20262903]),
            confirmation=dict(blocks=2,rounds=20,opponents=list(OPS),seeds=list(range(20262911,20262917))),
            samples=dict(rounds=5,opponents=['Target','SpinBot','Walls'],seed=20262921)).items():
        if any(registered[stage].get(k)!=v for k,v in expected.items()):raise ValueError('Registered stage budget/seeds differ')
    (args.output/'registered-protocol.json').write_bytes(encoded)
    report=dict(status='running',registrationCommit=REGISTRATION,planSHA256=hashlib.sha256(plan).hexdigest(),
        sources=dict(previous=BASELINE_SOURCE,candidate=CANDIDATE_SOURCE),jarSHA256=JARS,protocolSHA256=hashlib.sha256(encoded).hexdigest(),
        executionSource=subprocess.check_output(['git','rev-parse','HEAD'],cwd=benchmark.ROOT,text=True).strip(),runtimeOS=retcon_stats.os_provenance(),games=[])
    target=args.output/'experiment.json'
    def save():target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        for arm,source in sources.items():
            if benchmark.digest(source/'AISurvivor.jar')!=JARS[arm]:raise ValueError('Source-specific JAR differs')
        if subprocess.check_output(['git','diff','--name-only',CANDIDATE_SOURCE,'HEAD','--','src/main/java','src/test/java','AISurvivor'],cwd=benchmark.ROOT).strip():raise ValueError('Candidate production changed after freeze')
        report['runtimeJava']=subprocess.check_output([args.java,'-XshowSettings:properties','-version'],stderr=subprocess.STDOUT,text=True,timeout=15)
        python_source,python,provenance=native.prepare(args.cache,args.python);report['pythonOpponents']=provenance
        retcon_source,api,provenance=retcon.prepare(args.cache);report['retconProvenance']=provenance
        runner=benchmark.download(args.cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        archive=benchmark.download(args.cache,'sample-bots-Java-1.4.0.zip',benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples=args.output/'samples';benchmark.extract(archive,samples)
        for stage in ('screen','confirmation','samples'):
            if stage!='screen' and report['screenDecision']['status']!='advance':break
            for game in layout(stage):
                report['inProgressGame']=game['gameId'];save();report['games'].append(play(args,game,sources,runner,python_source,python,retcon_source,api,samples));del report['inProgressGame'];save()
            if stage=='screen':report['screenDecision']=screen_decision(report['games']);save();print(json.dumps(report['screenDecision'],indent=2),flush=True)
        if report['screenDecision']['status']=='advance':report['decision']=confirmation_decision([g for g in report['games'] if g['stage']!='screen'])
        else:report['decision']=dict(status='reject-at-screen',confirmationRounds=0)
        report['status']='complete';print(json.dumps(report['decision'],indent=2))
    except BaseException as error:report.update(status='invalid',error=str(error));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','candidate','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--java',required=True);parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--cache',type=Path,default=benchmark.ROOT/'.benchmark-cache');main(parser.parse_args())
