"""One registered whole-product default duel comparison, with isolated AIS processes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import struct
import subprocess

import benchmark
import compare
import default_duel_policy_screen as policy
import retcon
import retcon_stats

PLAN_COMMIT = '0d595721f7ff952dd938c0d31e16290c8d70848c'
PLAN = 'docs/improvements/coherent-duel-v1.md'
SOURCE = '9c79e4e26b3323704df5dad467af182e0b48fafd'
BASE = policy.BASE
JARS = dict(candidate='9daf9cf0e24f319900fee1c8869d50df4ffb004719b1380a8db40fb4961551f6', previous=policy.BASE_JAR)
ASSETS = ('AISurvivor.jar', 'AISurvivor.json', 'AISurvivor.sh', 'AISurvivor.cmd')


def layout():
    games = []
    for i in range(3):
        for arm in (('previous', 'candidate') if i % 2 == 0 else ('candidate', 'previous')):
            games.append(dict(gameId=f'coherent-retcon-{i+1}-{arm}', kind='Retcon', block=i+1,
                              rounds=20, arms=[arm], seeds={arm: 20261901+i}))
    for i in range(3):
        arms = ['candidate', 'previous'] if i % 2 == 0 else ['previous', 'candidate']
        games.append(dict(gameId=f'coherent-self-{i+1}', kind='self', block=i+1, rounds=10,
                          arms=arms, seeds=dict(candidate=20261911+i, previous=20262011+i)))
    for i, opponent in enumerate(('Target', 'SpinBot', 'Walls')):
        for arm in (('previous', 'candidate') if i % 2 == 0 else ('candidate', 'previous')):
            games.append(dict(gameId=f'coherent-{opponent}-{arm}', kind=opponent, block=i+1,
                              rounds=3, arms=[arm], seeds={arm: 20261921}))
    return games


def clean_environment(environment):
    return {k: v for k, v in environment.items() if not k.startswith('AISURVIVOR_') and k != 'RETCON_STATS_DIR'}


def prepare_bot(source, destination, arm, seed, java):
    if destination.name != 'AISurvivor':
        raise ValueError('Official script/config discovery requires the AISurvivor directory basename')
    destination.mkdir(parents=True)
    originals = {}
    for name in ASSETS:
        shutil.copy2(source / name, destination / name)
        originals[name] = benchmark.digest(source / name)
    if originals['AISurvivor.jar'] != JARS[arm]:
        raise ValueError('Bot JAR differs from registered version')
    original_text = {name: (source/name).read_text(encoding='utf-8') for name in ASSETS if not name.endswith('.jar')}
    descriptor = json.loads((destination / 'AISurvivor.json').read_bytes())
    descriptor['name'] = 'AISurvivor-Candidate' if arm == 'candidate' else 'AISurvivor-Accepted'
    (destination / 'AISurvivor.json').write_text(json.dumps(descriptor, indent=2)+'\n', encoding='utf-8')
    model, metrics = destination / 'model.dat', destination / 'metrics.jsonl'
    variables = dict(AISURVIVOR_MODE='train', AISURVIVOR_SEED=str(seed),
                     AISURVIVOR_MODEL_PATH=str(model), AISURVIVOR_METRICS_PATH=str(metrics))
    # Official runner launches every participant in one inherited environment. Bind these
    # values inside EACH shell, so self-play never shares model or telemetry state.
    script = '#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\n'
    script += 'unset AISURVIVOR_DUEL_POLICY AISURVIVOR_ACTION AISURVIVOR_DATASET_PATH\n'
    script += ''.join('export '+k+'='+shlex.quote(v)+'\n' for k, v in variables.items())
    script += 'exec '+shlex.quote(java)+' -jar AISurvivor.jar\n'
    (destination / 'AISurvivor.sh').write_text(script, encoding='utf-8', newline='\n')
    (destination / 'AISurvivor.sh').chmod(0o755)
    (destination/'launch-provenance.json').write_text(json.dumps(dict(originalHashes=originals, originalText=original_text,
        variables=variables),indent=2)+'\n',encoding='utf-8')
    return dict(name=descriptor['name'], version=descriptor['version'], seed=seed,
                originalHashes=originals, launchedHashes={n: benchmark.digest(destination/n) for n in ASSETS},
                model=str(model), metrics=str(metrics))


def validate_endpoint(path, rounds):
    if compare.checkpoint_rounds(path) != rounds:
        raise ValueError('Endpoint checkpoint counter differs from scored rounds')
    data = path.read_bytes()
    compare.checkpoint_diagnostics(path)
    # Header28 + tau8; all remaining values are double arrays.
    if any(not math.isfinite(v) for v in struct.unpack('>'+str((len(data)-28)//8)+'d', data[28:])):
        raise ValueError('Endpoint contains nonfinite numeric state')
    return dict(sha256=benchmark.digest(path), bytes=len(data), totalRounds=rounds)


def play(args, game, sources, runner, samples, retcon_source, api):
    directory = args.output / game['gameId']; directory.mkdir()
    entries, bots = [], {}
    for arm in game['arms']:
        path = directory / arm / 'AISurvivor'
        bots[arm] = prepare_bot(sources[arm], path, arm, game['seeds'][arm], args.java)
        entries.append(str(path))
    env = clean_environment(os.environ)
    env['PATH'] = str(Path(args.java).parent)+os.pathsep+env.get('PATH', '')
    capture = None
    if game['kind'] != 'self':
        if game['kind'] == 'Retcon':
            opponent = retcon_source
            if retcon.source_hash(opponent) != retcon_stats.SOURCE_SHA256:
                raise ValueError('Retcon source changed')
            capture = retcon_stats.Capture(retcon_stats.java_temp_directory(args.java))
            env.update(RUMBLE_JAVA_CACHE=str(api), RETCON_STATS_DIR=str(capture.path))
        else:
            configs = list(samples.rglob(game['kind']+'.json'))
            if len(configs) != 1:
                raise ValueError('Sample identity is ambiguous')
            opponent = configs[0].parent
        isolated = directory / 'opponents' / game['kind']
        benchmark.isolate_opponent(opponent, isolated); entries.append(str(isolated))
    result_path = directory / 'battle.json'
    command = [args.java, '-cp', str(runner), str(benchmark.ROOT/'tools/BenchmarkBattle.java'),
               str(game['rounds']), '900', str(result_path), *entries]
    native = None
    try:
        benchmark.run_process(command, benchmark.ROOT, env, directory/'runner.log', 990)
        official = json.loads(result_path.read_bytes())
        if (official['numberOfRounds'] != game['rounds'] or len(official['results']) != 2
                or len({r['id'] for r in official['results']}) != 2):
            raise ValueError('Official budget/roster mismatch')
        if game['kind'] not in ('self','Retcon'):
            descriptor=json.loads(configs[0].read_bytes())
            if len([r for r in official['results'] if r['name']==descriptor['name'] and r['version']==descriptor['version']])!=1:
                raise ValueError('Sample official identity mismatch')
        if capture:
            enemies = [r for r in official['results'] if r['name'] == 'Retcon' and r['version'] == '0.2.0']
            if len(enemies) != 1:
                raise ValueError('Retcon official identity mismatch')
            native = capture.collect(directory/'retcon-stats', game['rounds'], enemies[0]['id'], require=True)
    finally:
        if capture:
            try:
                if native is None:
                    capture.collect(directory/'retcon-stats', game['rounds'], None, require=False)
            finally:
                capture.close()
    battles = {}
    for arm, bot in bots.items():
        identity = [r for r in official['results'] if r['name'] == bot['name'] and r['version'] == bot['version']]
        if len(identity) != 1:
            raise ValueError('AIS official identity mismatch')
        records = [json.loads(line) for line in Path(bot['metrics']).read_text().splitlines() if line.strip()]
        benchmark.validate_metrics(records, game['rounds'], 'train', bot['seed'], game['rounds'])
        metrics = benchmark.summarize_metrics(records)
        if metrics['skippedTurns'] or metrics['checkpointLoaded']:
            raise ValueError('Skipped turn or loaded model invalidates collection')
        endpoint = validate_endpoint(Path(bot['model']), game['rounds'])
        mechanism = policy.mechanism(records) if arm == 'candidate' else None
        battles[arm] = dict(**identity[0], rounds=game['rounds'], results=official['results'],
                            metrics=metrics, endpoint=endpoint, mechanism=mechanism, bot=bot)
    return dict(**game, battles=battles, native=native)


def decision(games):
    if [{k: g[k] for k in ('gameId','kind','block','rounds','arms','seeds')} for g in games] != layout():
        raise ValueError('Incomplete or changed registered game sequence')
    blocks = [{arm: next(g['battles'][arm] for g in games if g['kind']=='Retcon' and g['block']==i and arm in g['arms'])
               for arm in ('previous','candidate')} for i in range(1,4)]
    deltas = [compare.score_share(b['candidate'])-compare.score_share(b['previous']) for b in blocks]
    totals = {a: compare.aggregate(blocks,a) for a in ('previous','candidate')}
    selves = [g['battles']['candidate'] for g in games if g['kind']=='self']
    self_shares = [compare.score_share(b) for b in selves]
    sample_deltas = {name: compare.score_share(next(g['battles']['candidate'] for g in games if g['kind']==name and 'candidate' in g['arms']))-
        compare.score_share(next(g['battles']['previous'] for g in games if g['kind']==name and 'previous' in g['arms'])) for name in ('Target','SpinBot','Walls')}
    checks = dict(retconGain=statistics.mean(deltas)>=.05, positiveBlocks=sum(d>0 for d in deltas)>=2,
        worstBlock=min(deltas)>=-.05, wins=totals['candidate']['wins']>=totals['previous']['wins']-3,
        wallRate=totals['candidate']['wallsPer1000Turns']<=1.2*totals['previous']['wallsPer1000Turns'],
        selfMean=statistics.mean(self_shares)>.5, selfWins=sum(b['firstPlaces'] for b in selves)>=16,
        selfGames=sum(s>.5 for s in self_shares)>=2, samples=all(d>=-.05 for d in sample_deltas.values()),
        zeroSkips=all(b['metrics']['skippedTurns']==0 for g in games for b in g['battles'].values()))
    return dict(status='retain' if all(checks.values()) else 'reject', checks=checks,
                retconDeltas=deltas, meanRetconDelta=statistics.mean(deltas), retcon=totals,
                selfShares=self_shares, selfWins=sum(b['firstPlaces'] for b in selves), sampleDeltas=sample_deltas)


def main(args):
    args.output, args.cache = args.output.resolve(), args.cache.resolve()
    sources = dict(candidate=args.candidate.resolve(), previous=args.previous.resolve())
    if args.output.exists():
        raise ValueError('Never replace an earlier attempt')
    args.output.mkdir(parents=True)
    plan = subprocess.check_output(['git','show',PLAN_COMMIT+':'+PLAN], cwd=benchmark.ROOT)
    (args.output/'registered-plan.md').write_bytes(plan)
    (args.output/'registered-games.json').write_text(json.dumps(layout(),indent=2)+'\n',encoding='utf-8')
    for paths, source in ((['src/main/java','src/test/java','AISurvivor'],SOURCE),):
        if subprocess.check_output(['git','diff','--name-only',source,'HEAD','--',*paths], cwd=benchmark.ROOT).strip():
            raise ValueError('Production source/assets changed after source freeze')
    report = dict(status='running', candidateSource=SOURCE, previousSource=BASE, jarSHA256=JARS,
                  registrationCommit=PLAN_COMMIT, planSHA256=hashlib.sha256(plan).hexdigest(),
                  executionSource=subprocess.check_output(['git','rev-parse','HEAD'], cwd=benchmark.ROOT,text=True).strip(),
                  runtimeOS=retcon_stats.os_provenance(), games=[])
    output = args.output/'experiment.json'
    def save(): output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        java = subprocess.run([args.java,'-XshowSettings:properties','-version'],capture_output=True,text=True,check=True,timeout=15)
        report['runtimeJava']=java.stdout+java.stderr
        runner=benchmark.download(args.cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        archive=benchmark.download(args.cache,'sample-bots-Java-1.4.0.zip',benchmark.ASSETS['sample-bots-Java-1.4.0.zip'])
        samples=args.output/'samples'; benchmark.extract(archive,samples)
        opponent,api,provenance=retcon.prepare(args.cache)
        report.update(opponentProvenance=provenance, runnerSHA256=benchmark.digest(runner), sampleAssetSHA256=benchmark.digest(archive))
        for game in layout():
            report['inProgressGame']=game['gameId']; save()
            report['games'].append(play(args,game,sources,runner,samples,opponent,api))
            del report['inProgressGame']; save()
        report.update(status='complete',decision=decision(report['games']))
        print(json.dumps(report['decision'],indent=2))
    except BaseException as error:
        report.update(status='invalid',error=str(error)); raise
    finally: save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','previous','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=benchmark.ROOT/'.benchmark-cache')
    parser.add_argument('--java',required=True)
    main(parser.parse_args())
