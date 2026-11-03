"""One isolated wave-defense pilot, then fresh Diamond-only confirmation if its fixed screen passes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import benchmark
import compare
import retcon_stats
import strong_melee_opponents as native

REGISTRATION='cdbdc39157704fc8ba95878c328869c06329ebf1'
PLAN='docs/improvements/diamond-wave-horizon-v4.md'
SOURCE='60d7ff072c08543ddb9517c7896aa39db1ec90ce'
JAR='5e3b47787e22af00dff340f58c7e93bf72a017b8521941cdd37f071c8d1679f0'
AIS_JAR='b41b5de479b1288f087e0cbc9efa64abb68f12381b41d33ad65ff48be5914800'
PRODUCTION_SOURCE='9b1624da2d9b09ec9cc4f580b51bc82edd2c1485'
NAME,VERSION='DiamondChallenger','0.4.0'
FILES=('DiamondChallenger.jar','DiamondChallenger.json','DiamondChallenger.sh','DiamondChallenger.cmd')

def layout(stage):
    if stage=='pilot':return [dict(gameId='diamond-wave-horizon-v4-pilot',stage=stage,seed=20263401,rounds=20)]
    if stage=='confirmation':return [dict(gameId=f'diamond-wave-horizon-v4-confirm-g{i:02d}',stage=stage,seed=20263410+i,rounds=20) for i in range(1,4)]
    raise ValueError('Unregistered stage')

def check_rows(rows,rounds,seed):
    if len(rows)!=rounds:raise ValueError('Incomplete challenger telemetry')
    integers=('turns','shotsFired','bulletHits','wallCollisions','botCollisions','skippedTurns','rangeObservedTurns','closeTurns','phaseDecisions',
              'learnedAimDecisions','fallbackAimDecisions','acceptedPassageSamples','confirmedLearnedShots','contextMemorySize','inferredWaves','matchedHits','unmatchedHits','rejectedScanPairs','incomingHits','plannerDecisions','profileDecisions','cwDecisions','stopDecisions','ccwDecisions')
    for i,row in enumerate(rows,1):
        if any(row.get(k)!=v for k,v in dict(schemaVersion=1,name=NAME,version=VERSION,round=i,roundsCompleted=i,seed=seed,mode='train',checkpointLoaded=False).items()):
            raise ValueError('Wrong challenger identity/round/cold policy')
        if any(type(row[k]) is not int for k in ('schemaVersion','round','roundsCompleted','seed')) or row['checkpointLoaded'] is not False:raise ValueError('Invalid telemetry types')
        for key in integers:
            if type(row.get(key)) is not int or row[key]<0:raise ValueError('Invalid challenger counter '+key)
        for key in ('damageDealt','damageTaken','rangeSum','energySpent'):
            if type(row.get(key)) not in (float,int) or not math.isfinite(row[key]) or row[key]<0:raise ValueError('Nonfinite challenger measurement')
        if row['matchedHits']+row['unmatchedHits']!=row['incomingHits'] or row['profileDecisions']>row['plannerDecisions']:raise ValueError('Inconsistent wave activation')
        if row['bulletHits']>row['shotsFired'] or row['closeTurns']>row['rangeObservedTurns']:raise ValueError('Inconsistent challenger counters')
        if row['contextMemorySize']>512 or row['confirmedLearnedShots']>row['shotsFired'] or row['acceptedPassageSamples']>row['shotsFired']:raise ValueError('Impossible context activity')
        if not .1*row['shotsFired']-1e-6<=row['energySpent']<=3*row['shotsFired']+1e-6:raise ValueError('Confirmed shot energy differs from legal power range')
        if any(type(row.get(k)) is not bool for k in ('won','died')) or row['won'] and row['died']:raise ValueError('Invalid challenger terminal flags')

def endpoint(path,rounds,seed):
    data=json.loads(path.read_bytes())
    def finite(value):
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,list):return all(finite(v) for v in value)
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        return True
    if not finite(data):raise ValueError('Nonfinite own model snapshot')
    if any(data.get(k)!=v for k,v in dict(schemaVersion=1,name=NAME,version=VERSION,seed=seed,roundsCompleted=rounds,checkpointLoaded=False).items()):
        raise ValueError('Endpoint provenance differs')
    if any(type(data[k]) is not int for k in ('schemaVersion','seed','roundsCompleted')) or data['checkpointLoaded'] is not False:raise ValueError('Invalid endpoint types')
    for key in ('gunAttempts','gunHits'):
        if not isinstance(data.get(key),list) or len(data[key])!=3 or any(type(v) is not int or v<0 for v in data[key]):raise ValueError('Invalid own gun snapshot')
    if any(h>a for h,a in zip(data['gunHits'],data['gunAttempts'])):raise ValueError('Impossible virtual counts')
    memory=data.get('contextMemory')
    if not isinstance(memory,list) or len(memory)>512 or type(data.get('contextMemorySize')) is not int or len(memory)!=data['contextMemorySize']:
        raise ValueError('Invalid bounded context memory')
    if type(data.get('contextAccepted')) is not int or data['contextAccepted']<len(memory):raise ValueError('Invalid passage total')
    bins=data.get('dangerBins')
    if not isinstance(bins,list) or len(bins)!=4 or any(not isinstance(b,list) or len(b)!=47 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in b) or sum(b)>32+1e-8 for b in bins):raise ValueError('Invalid bounded danger memory')
    if any(type(data.get(k)) is not int or data[k]<0 for k in ('inferredWavesTotal','matchedHitsTotal')):raise ValueError('Invalid wave totals')
    previous=0
    for sample in memory:
        f=sample.get('features')
        if not isinstance(f,list) or len(f)!=7 or any(type(v) not in (int,float) or not math.isfinite(v) for v in f):raise ValueError('Invalid context features')
        if any(not (0<=v<=1 if i in (0,1,5,6) else -1<=v<=1) for i,v in enumerate(f)):raise ValueError('Unclipped context feature')
        gf=sample.get('guessFactor')
        if type(gf) not in (int,float) or not math.isfinite(gf) or not -1<=gf<=1:raise ValueError('Invalid GF proxy label')
        for key in ('aimTurn','shotTurn','observedTurn','createdSequence','observedSequence','insertionSequence'):
            if type(sample.get(key)) is not int or sample[key]<0:raise ValueError('Invalid context availability metadata')
        if not (sample['aimTurn']+1==sample['shotTurn']<=sample['observedTurn'] and sample['createdSequence']<sample['observedSequence']):
            raise ValueError('Future or unconfirmed passage sample')
        if sample['insertionSequence']<=previous or sample['insertionSequence']>data['contextAccepted']:raise ValueError('Non-FIFO context snapshot')
        for key in ('originX','originY','referenceX','referenceY','observedX','observedY','escape'):
            if type(sample.get(key)) not in (int,float) or not math.isfinite(sample[key]):raise ValueError('Invalid passage geometry')
        if type(sample.get('generation')) is not int or sample['generation']<0 or type(sample.get('sign')) is not int or sample['sign'] not in (-1,1) or sample['escape']<=0:
            raise ValueError('Invalid passage frame')
        reference=math.degrees(math.atan2(sample['referenceY']-sample['originY'],sample['referenceX']-sample['originX']))
        observed=math.degrees(math.atan2(sample['observedY']-sample['originY'],sample['observedX']-sample['originX']))
        proxy=max(-1,min(1,((observed-reference+180)%360-180)*sample['sign']/sample['escape']))
        if not math.isclose(gf,proxy,abs_tol=1e-9):raise ValueError('GF proxy differs from confirmed-origin geometry')
        previous=sample['insertionSequence']
    return dict(sha256=benchmark.digest(path),bytes=path.stat().st_size,state=data)

def decision(games,stage):
    expected=layout(stage)
    if [{k:g[k] for k in e} for g,e in zip(games,expected)]!=expected or len(games)!=len(expected):raise ValueError('Incomplete registered game cohort')
    shares=[g['scoreShare'] for g in games];wins=sum(g['battle']['firstPlaces'] for g in games)
    checks=dict(complete=True,noChallengerSkips=all(g['metrics']['skippedTurns']==0 for g in games))
    if stage=='pilot':checks.update(progressShare=shares[0]>.20,oneWin=wins>=1)
    else:checks.update(winningMean=statistics.mean(shares)>.50,majorityWins=wins>30,majorityGames=sum(v>.50 for v in shares)>=2)
    result=dict(status=('advance' if stage=='pilot' else 'confirmed') if all(checks.values()) else 'reject',checks=checks,
                meanShare=statistics.mean(shares),firstPlaces=wins,shares=shares)
    if stage=='confirmation':
        result['descriptiveTLower95']=statistics.mean(shares)-2.919985580353724*statistics.stdev(shares)/math.sqrt(3)
        result['intervalLimitation']='Descriptive game-level Student-t assumptions; not a binding gate or independent-round inference.'
    return result

def play(args,game,runner,source,python):
    root=args.output/game['gameId'];root.mkdir();run=root/'run';run.mkdir()
    bot=run/NAME;shutil.copytree(args.bot_dir,bot)
    if {f.name for f in bot.iterdir() if f.is_file()}!=set(FILES):raise ValueError('Unexpected challenger bundle assets')
    if benchmark.digest(bot/'DiamondChallenger.jar')!=JAR:raise ValueError('Challenger JAR changed')
    (bot/'DiamondChallenger.sh').chmod(0o755)
    descriptor=json.loads((bot/'DiamondChallenger.json').read_bytes())
    if (descriptor['name'],descriptor['version'])!=(NAME,VERSION):raise ValueError('Wrong own descriptor')
    env=native.clean_environment(os.environ)
    env={k:v for k,v in env.items() if not k.startswith('CHALLENGER_')}
    env.update(native.stage(source,run,python))
    opponent=run/'opponents/diamond-port';benchmark.isolate_opponent(source/'bots/ports/diamond-port',opponent)
    opponent_info=json.loads((opponent/'diamond-port.json').read_bytes())
    model=root/'model.json';metrics=root/'metrics.jsonl'
    env.update(CHALLENGER_SEED=str(game['seed']),CHALLENGER_MODEL_PATH=str(model),CHALLENGER_METRICS_PATH=str(metrics))
    env['PATH']=str(Path(args.java).parent)+os.pathsep+env.get('PATH','')
    official=run/'battle.json'
    command=[args.java,'-cp',str(runner),str(benchmark.ROOT/'tools/BenchmarkBattle.java'),str(game['rounds']),'900',str(official),str(bot),str(opponent)]
    benchmark.run_process(command,benchmark.ROOT,env,run/'runner.log',990)
    data=json.loads(official.read_bytes())
    expected=sorted([(NAME,VERSION),(opponent_info['name'],opponent_info['version'])])
    if data['numberOfRounds']!=game['rounds'] or sorted((r['name'],r['version']) for r in data['results'])!=expected or len({r['id'] for r in data['results']})!=2:
        raise ValueError('Official round count/identities differ')
    own=next(r for r in data['results'] if r['name']==NAME)
    rows=[json.loads(line) for line in metrics.read_bytes().splitlines()];check_rows(rows,game['rounds'],game['seed'])
    summary=benchmark.summarize_metrics(rows)
    if sum(r['won'] for r in rows)!=own['firstPlaces']:raise ValueError('Official wins differ from terminal events')
    ranges=sum(r['rangeObservedTurns'] for r in rows)
    state=endpoint(model,game['rounds'],game['seed'])
    if sum(row['acceptedPassageSamples'] for row in rows)!=state['state']['contextAccepted'] or rows[-1]['contextMemorySize']!=state['state']['contextMemorySize']:
        raise ValueError('Passage activity and endpoint differ')
    if sum(r['inferredWaves'] for r in rows)!=state['state']['inferredWavesTotal'] or sum(r['matchedHits'] for r in rows)!=state['state']['matchedHitsTotal']:raise ValueError('Wave totals differ from endpoint')
    activity={key:sum(row[key] for row in rows) for key in ('learnedAimDecisions','fallbackAimDecisions','acceptedPassageSamples','confirmedLearnedShots')}
    waveActivity={key:sum(row[key] for row in rows) for key in ('inferredWaves','matchedHits','unmatchedHits','rejectedScanPairs','incomingHits','plannerDecisions','profileDecisions','cwDecisions','stopDecisions','ccwDecisions')}
    activity['endpointMemorySize']=state['state']['contextMemorySize']
    return dict(**game,battle=dict(**own,results=data['results']),scoreShare=compare.score_share(dict(**own,results=data['results'])),metrics=summary,contextActivity=activity,waveActivity=waveActivity,
        rangeDiagnostics=dict(observedTurns=ranges,meanObservedRange=sum(r['rangeSum'] for r in rows)/ranges if ranges else None,
            closeTurns=sum(r['closeTurns'] for r in rows),phaseDecisions=sum(r['phaseDecisions'] for r in rows)),energySpent=sum(r['energySpent'] for r in rows),endpoint=state,
        botAssetHashes={name:benchmark.digest(bot/name) for name in FILES},pythonSourceHashes=native.verify_staged(run,['diamond-port']))

def main(args):
    args.output,args.cache,args.bot_dir=args.output.resolve(),args.cache.resolve(),args.bot_dir.resolve()
    if len(REGISTRATION)!=40 or len(SOURCE)!=40 or len(JAR)!=64:raise ValueError('Freeze source/JAR/plan before execution')
    if args.output.exists():raise ValueError('Never replace an attempt')
    args.output.mkdir(parents=True)
    plan=subprocess.check_output(['git','show',REGISTRATION+':'+PLAN],cwd=benchmark.ROOT)
    (args.output/'registered-plan.md').write_bytes(plan)
    if subprocess.check_output(['git','diff','--name-only',PRODUCTION_SOURCE,SOURCE,'--','src/main/java','src/test/java','AISurvivor','build.gradle','settings.gradle'],cwd=benchmark.ROOT).strip():raise ValueError('Experiment changed production AIS')
    if subprocess.check_output(['git','diff','--name-only',SOURCE,'HEAD','--','experiments/DiamondChallenger','src/main/java','src/test/java','AISurvivor','build.gradle','settings.gradle'],cwd=benchmark.ROOT).strip():raise ValueError('Frozen source changed')
    if benchmark.digest(benchmark.ROOT/'AISurvivor/AISurvivor.jar')!=AIS_JAR:raise ValueError('Production AIS changed')
    report=dict(status='running',registration=REGISTRATION,planSHA256=hashlib.sha256(plan).hexdigest(),source=SOURCE,jarSHA256=JAR,productionAISJarSHA256=AIS_JAR,
        executionSource=subprocess.check_output(['git','rev-parse','HEAD'],cwd=benchmark.ROOT,text=True).strip(),runtimeOS=retcon_stats.os_provenance(),games=[])
    target=args.output/'experiment.json'
    def save():target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    try:
        report['runtimeJava']=subprocess.check_output([args.java,'-XshowSettings:properties','-version'],stderr=subprocess.STDOUT,text=True,timeout=15)
        source,python,provenance=native.prepare(args.cache,args.python);report['pythonOpponents']=provenance;save()
        runner=benchmark.download(args.cache,'robocode-tankroyale-runner-1.4.0.jar',benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
        for stage in ('pilot','confirmation'):
            if stage=='confirmation' and report['pilotDecision']['status']!='advance':break
            for game in layout(stage):
                report['inProgress']=game['gameId'];save();report['games'].append(play(args,game,runner,source,python));del report['inProgress'];save()
                print(json.dumps({k:report['games'][-1][k] for k in ('gameId','scoreShare','metrics','rangeDiagnostics')}),flush=True)
            report[stage+'Decision']=decision([g for g in report['games'] if g['stage']==stage],stage);save()
        report.update(status='complete',decision=report.get('confirmationDecision',report['pilotDecision']));print(json.dumps(report['decision']),flush=True)
    except BaseException as error:report.update(status='invalid',failure=str(error));raise
    finally:save()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','bot-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=benchmark.ROOT/'.benchmark-cache');parser.add_argument('--java',required=True)
    parser.add_argument('--python',default=sys.executable);main(parser.parse_args())
