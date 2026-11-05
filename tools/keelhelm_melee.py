"""Issue 123: fixed minimind top-four comparison; prepare once, review, run once."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER_SHA = '02f6d1d8e9346a4aeae1b91e7abb278c98ac1f1d30c605f867758566f5f327cf'
RETC = '28981c7a469ac064194b41001cd7183ec5ecf189'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, value):
    p.with_suffix('.tmp').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    p.with_suffix('.tmp').replace(p)

def files(root):
    return [p for p in sorted(root.rglob('*')) if p.is_file()
            and not set(p.relative_to(root).parts) & {'build', '.gradle', '__pycache__', '.git'}
            and p.suffix != '.pyc']

def schedule():
    games = []
    for slot in range(4):
        order = ['KeelHelm', 'Diamond Port', 'Keel', 'Prism']
        order = order[slot:] + order[:slot]
        for arm in (['control', 'candidate'] if slot % 2 == 0 else ['candidate', 'control']):
            games.append(dict(id=f'melee-{slot+1}-{arm}', arm=arm, scenario='melee',
                              rounds=20, seed=2026110501+slot, order=order))
    for slot in range(2):
        for arm in (['control', 'candidate'] if slot == 0 else ['candidate', 'control']):
            games.append(dict(id=f'retcon-{slot+1}-{arm}', arm=arm, scenario='Retcon', rounds=10,
                              seed=2026110511+slot, order=['KeelHelm', 'Retcon'] if slot == 0 else ['Retcon', 'KeelHelm']))
    for slot, name in enumerate(['Walls', 'SpinBot']):
        for arm in (['control', 'candidate'] if slot == 0 else ['candidate', 'control']):
            games.append(dict(id=f'{name.lower()}-{arm}', arm=arm, scenario=name, rounds=5,
                              seed=2026110521+slot, order=['KeelHelm', name]))
    return games

def collect(source, evidence, pins, java, python, *, control_jar_sha=None, schedule_rows=None, extra_paths=()):
    candidate = ROOT/'experiments/KeelHelm'
    control = evidence/'control-source'
    paths = files(candidate) + files(control) + [pins/'entry-manifests.json',pins/'shared-inputs.json']
    paths += list(extra_paths) + [Path(__file__), ROOT/'tools/test_keelhelm_melee.py', ROOT/'tournament/scripts/TournamentBattle.java',
              ROOT/'docs/improvements/keelhelm-melee-v1-protocol.md',
              ROOT/'docs/improvements/keelhelm-melee-v2-protocol.md']
    bases = {'Diamond Port': source/'guest-source/bots/ports/diamond-port',
             'Keel': source/'experiments/Keel/bot', 'Prism': source/'experiments/Prism/bot',
             'Retcon': source/f'.benchmark-cache/retcon/{RETC}/Retcon',
             'Walls': source/'samples/Walls', 'SpinBot': source/'samples/SpinBot'}
    registered = json.loads((pins/'entry-manifests.json').read_text())
    for name in list(bases) + ['KeelHelm']:
        base = bases[name] if name != 'KeelHelm' else source/'experiments/KeelHelm/bot'
        entrant = files(base)
        if name in ('KeelHelm','Keel','Prism'):
            entrant += files(source/f'experiments/{name}/src')
            entrant += [source/f'experiments/{name}/{name}.jar',source/f'experiments/{name}/build.gradle']
        if {str(p.relative_to(source)):sha(p) for p in entrant} != registered[name]:
            raise ValueError('Tournament opponent pin changed: '+name)
    if sha(control/'KeelHelm.jar') != (control_jar_sha or sha(source/'experiments/KeelHelm/KeelHelm.jar')):
        raise ValueError('Control differs from the frozen champion JAR')
    for name, base in bases.items():
        paths += files(base)
        if name in ('Keel', 'Prism'):
            paths += files(source/f'experiments/{name}/src') + [source/f'experiments/{name}/{name}.jar',source/f'experiments/{name}/build.gradle']
    for folder in [source/'guest-source/bots/bot_core', source/'samples/lib', source/'.benchmark-cache/retcon-api']:
        paths += files(folder)
    paths += [source/'guest-source/LICENSE', source/'tools/opponents/strong-melee.lock.json',
              source/'tools/opponents/retcon.lock.json']
    runner = source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'
    if sha(runner) != RUNNER_SHA:
        raise ValueError('Runner differs from release pin')
    paths += [runner]
    # Runtime binaries, libraries, source modules and installed wheel bytes are read-only inputs.
    shared = json.loads((pins/'shared-inputs.json').read_text())
    # Exact edition shared inputs (sample/API/wheel/runtime hashes) are also rechecked.
    for rel, expected in shared['files'].items():
        p = source/rel
        if sha(p) != expected: raise ValueError('Tournament shared pin changed: '+rel)
        paths.append(p)
    paths += files(java.parent.parent)
    paths += files(python.parent.parent)
    paths += files(python.resolve().parent.parent)
    paths += [java.resolve(), python.resolve()]
    if sha(java) != shared['javaSHA256'] or sha(python) != shared['pythonSHA256']:
        raise ValueError('Tournament Java/Python executable pin changed')
    java_version = subprocess.check_output([str(java),'-version'],stderr=subprocess.STDOUT,text=True)
    if 'version "25.' not in java_version: raise ValueError('JDK25 required after Java17 builds')
    python_runtime = json.loads(subprocess.check_output([str(python),'-c',
        'import sys,json,importlib.metadata as m; print(json.dumps(dict(version=sys.version,packages=sorted((d.metadata["Name"],d.version) for d in m.distributions()))))'],text=True))
    if java_version != shared['javaVersion'] or python_runtime != shared['pythonRuntime']:
        raise ValueError('Tournament Java/Python version/package inventory changed')
    identities = {name:json.loads(next(base.glob('*.json')).read_text()) for name, base in bases.items()}
    return dict(source=str(source),evidence=str(evidence),files={str(p):sha(p) for p in sorted(set(paths))}, bases={n:str(p) for n,p in bases.items()},
                identities={n:{k:v[k] for k in ('name','version')} for n,v in identities.items()},
                java=str(java), python=str(python), schedule=schedule() if schedule_rows is None else schedule_rows, parent=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                runtime=dict(host=platform.node(), platform=platform.platform(), python=sys.version,
                    java=java_version, botPython=python_runtime))

def verify_inputs(freeze):
    for p, expected in freeze['files'].items():
        if sha(p) != expected:
            raise ValueError('Frozen input changed: '+p)

def audit(recording, official, game, identities):
    starts = ends = 0
    round_ends = []; ticks = {}; final = None; ids = set(); own = None
    metrics = dict(shots=0, bulletHits=0, wallHits=0, botHits=0, skippedTurns=0, meleeTicks=0, duelTicks=0)
    with gzip.open(recording, 'rt') as stream:
        for line in stream:
            event = json.loads(line); kind = event['type']
            if kind == 'GameStartedEventForObserver':
                starts += 1; setup = event['gameSetup']; participants = event['participants']
                observed = sorted((r['name'],r['version']) for r in participants)
                if observed != sorted(identities): raise ValueError('Recorded identities differ')
                ids = {r['id'] for r in participants}
                own = next(r['id'] for r in participants if r['name'] == 'KeelHelm')
                size = (1000,1000) if game['scenario'] == 'melee' else (800,600)
                if (setup['arenaWidth'], setup['arenaHeight']) != size or setup['numberOfRounds'] != game['rounds'] or setup['turnTimeout'] != 100000:
                    raise ValueError('Recorded setup differs')
            elif kind == 'TickEventForObserver':
                r, t = event['roundNumber'], event['turnNumber']
                if t != ticks.get(r,0)+1: raise ValueError('Noncontiguous observer ticks')
                ticks[r] = t
                if t == 1 and {b['id'] for b in event['botStates']} != ids: raise ValueError('Initial roster differs')
                for b in event['botStates']:
                    if b['id'] == own: metrics['meleeTicks' if b['enemyCount'] > 1 else 'duelTicks'] += 1
                for e in event['events']:
                    k = e['type']
                    if k == 'SkippedTurnEvent' and e.get('botId') == own: metrics['skippedTurns'] += 1
                    if k == 'BulletFiredEvent' and e['bullet']['ownerId'] == own: metrics['shots'] += 1
                    if k == 'BulletHitBotEvent' and e['bullet']['ownerId'] == own: metrics['bulletHits'] += 1
                    if k == 'BotHitWallEvent' and e.get('victimId') == own: metrics['wallHits'] += 1
                    if k == 'BotHitBotEvent' and e.get('victimId') == own: metrics['botHits'] += 1
            elif kind == 'RoundEndedEventForObserver':
                round_ends.append(event['roundNumber'])
                if event['turnNumber'] != ticks.get(event['roundNumber']): raise ValueError('Endpoint differs')
            elif kind == 'GameEndedEventForObserver':
                ends += 1; final = event
    if starts != 1 or ends != 1 or round_ends != list(range(1,game['rounds']+1)) or sorted(ticks) != round_ends:
        raise ValueError('Incomplete observer coverage')
    if final['numberOfRounds'] != game['rounds']: raise ValueError('Final round count differs')
    for row in official['results']:
        recorded = next(r for r in final['results'] if r['id'] == row['id'])
        if any(recorded[k] != v for k,v in row.items()): raise ValueError('Official and observer final disagree')
    return dict(recordingSHA256=sha(recording), roundTurns=ticks, metrics=metrics)

def telemetry(path, rounds):
    raw = path.read_text(); rows = raw.splitlines()
    if any(' skip ' in row for row in rows): raise ValueError('Own skipped turn')
    for r in range(1,rounds+1):
        for suffix in ('start','end'):
            if rows.count(f'{r} {suffix}') != 1: raise ValueError('Telemetry round coverage differs')
        stats = [s for s in rows if s.startswith(f'{r} stats ')]
        if len(stats) != 1 or int(re.search(r' maxMs=(\d+)',stats[0])[1]) >= 75:
            raise ValueError('Own timing/coverage guard failed')
    return rows

def owned_group_live(pgid):
    rows=subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,stat='],text=True,timeout=3).splitlines()
    return [int(pid) for pid,group,state in (line.split() for line in rows)
            if int(group)==pgid and not state.startswith('Z')]

def cleanup_owned(process):
    """Retain signal failures, boundedly drain only our session, and verify rather than assume teardown."""
    result=dict(processGroup=process.pid,signalErrors=[],remainingPids=[],verified=False)
    try:
        for sig,drain in ((signal.SIGTERM,10),(signal.SIGKILL,5)):
            if not owned_group_live(process.pid):break
            try:os.killpg(process.pid,sig)
            except ProcessLookupError:pass
            except OSError as error:
                result['signalErrors'].append(dict(signal=int(sig),error=repr(error)))
                # A direct Popen child is unambiguously ours even if group signalling was denied.
                if process.poll() is None:
                    try:process.terminate() if sig==signal.SIGTERM else process.kill()
                    except OSError as fallback:result['signalErrors'].append(dict(directChild=True,error=repr(fallback)))
            deadline=time.monotonic()+drain
            while owned_group_live(process.pid) and time.monotonic()<deadline:time.sleep(.1)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:result['parentReapTimeout']=True
        result['remainingPids']=owned_group_live(process.pid)
        result['parentReturnCode']=process.poll()
        result['verified']=not result['remainingPids'] and process.poll() is not None
    except BaseException as error:
        result['validationError']=repr(error)
        # Enumeration can fail independently of our direct child handle. Always drain that child.
        for method in (process.terminate,process.kill):
            try:
                if process.poll() is None:method()
            except OSError as fallback:result['signalErrors'].append(dict(directChild=True,error=repr(fallback)))
            try:process.wait(timeout=3);break
            except subprocess.TimeoutExpired:continue
            except BaseException as reap:result['parentReapError']=repr(reap)
        result['parentReturnCode']=process.poll()
        # Group absence was not established; reaping the parent alone cannot verify the group.
        result['verified']=False
    return result

def execute(command, env, run, process_callback=None, failure_check=None):
    start = time.monotonic();primary=None
    with (run/'runner.log').open('x') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if process_callback is not None: process_callback(process.pid)
            while process.poll() is None:
                if failure_check is not None: failure_check()
                if time.monotonic()-start > 960: raise TimeoutError('Game timeout')
                metrics = run/'helm-metrics.txt'
                if metrics.exists() and re.search(r'^\d+ skip ',metrics.read_text(),re.M): raise ValueError('Own skipped turn')
                time.sleep(.25)
            if process.returncode: raise ValueError('Battle process failed: '+str(process.returncode))
        except BaseException as error:primary=error;raise
        finally:
            cleanup=cleanup_owned(process)
            try:
                save(run/'cleanup.json',cleanup)
                if not cleanup['verified']:raise RuntimeError('Owned teardown could not be verified: '+repr(cleanup))
            except BaseException as error:
                if primary is not None:primary.add_note('Additional teardown issue: '+repr(error))
                else:raise

def summary(games):
    result = {}; exact = {}
    for scenario in ('melee','Retcon','Walls','SpinBot'):
        result[scenario] = {}
        for arm in ('control','candidate'):
            rows = [g for g in games if g['scenario']==scenario and g['arm']==arm]
            shares = [g['scoreShare'] for g in rows]
            exact[scenario,arm] = sum((Fraction(g['helm']['totalScore'],sum(r['totalScore'] for r in g['official']['results'])) for g in rows),Fraction())/len(rows)
            result[scenario][arm] = dict(meanShare=statistics.mean(shares), gameShares=shares,
                firstPlaces=sum(g['helm']['firstPlaces'] for g in rows), scores=[g['helm']['totalScore'] for g in rows])
        result[scenario]['deltaPP'] = 100*(result[scenario]['candidate']['meanShare']-result[scenario]['control']['meanShare'])
    gates = dict(meleeShare=exact['melee','candidate']>=Fraction(1,5),
                 meleeGain=exact['melee','candidate']-exact['melee','control']>=Fraction(1,20),
                 **{n:exact[n,'candidate']-exact[n,'control']>=-Fraction(1,20) for n in ('Retcon','Walls','SpinBot')})
    return dict(scenarios=result,gates=gates,decision='accept-measured-melee-improvement' if all(gates.values()) else 'reject-improvement-claim')

def stage_bots(game, source, evidence, freeze, run):
    botdirs=[]; identities=[]
    for name in game['order']:
        if name == 'KeelHelm':
            project = ROOT/'experiments/KeelHelm' if game['arm']=='candidate' else evidence/'control-source'
            base=project/'bot'; folder=name
        else:
            base=Path(freeze['bases'][name]); folder=name if name in ('Keel','Prism') else base.name
        dst=run/'bots'/folder; shutil.copytree(base,dst)
        if name in ('KeelHelm','Keel','Prism'):
            jar=project/'KeelHelm.jar' if name=='KeelHelm' else source/f'experiments/{name}/{name}.jar'
            shutil.copy2(jar,dst/jar.name)
        for sh in dst.glob('*.sh'): sh.chmod(0o755)
        d=json.loads(next(dst.glob('*.json')).read_text());identities.append((d['name'],d['version']));botdirs.append(str(dst))
    if 'Diamond Port' in game['order']:
        shutil.copytree(source/'guest-source/bots/bot_core',run/'bots/bot_core')
        shutil.copy2(source/'guest-source/LICENSE',run/'bots/LICENSE.shared-Apache-2.0')
    if game['scenario'] in ('Walls','SpinBot'): shutil.copytree(source/'samples/lib',run/'bots/lib')
    return botdirs, identities

def main(args):
    source, evidence = args.source.resolve(), args.evidence.resolve()
    freeze_path = evidence/'freeze.json'
    if args.prepare:
        if freeze_path.exists(): raise ValueError('Never overwrite a frozen registration')
        save(freeze_path,collect(source,evidence,args.pins.resolve(),args.java.absolute(),args.python.absolute())); print('PREPARED '+sha(freeze_path)); return
    if not args.approval_sha or sha(freeze_path) != args.approval_sha: raise ValueError('Exact reviewed freeze required')
    freeze = json.loads(freeze_path.read_text()); verify_inputs(freeze)
    if freeze['source'] != str(source) or freeze['evidence'] != str(evidence): raise ValueError('Frozen execution paths differ')
    java, python = Path(freeze['java']), Path(freeze['python'])
    if args.stage_only:
        for game in freeze['schedule']:
            run=evidence/'staging-preflight'/game['id'];run.mkdir(parents=True)
            dirs,identities=stage_bots(game,source,evidence,freeze,run)
            if len(set(dirs)) != len(game['order']) or len(identities) != len(game['order']): raise ValueError('Staging collision')
        print('ALL 16 ROSTERS STAGED; ZERO GAMES LAUNCHED');return
    result_path = evidence/'results.json'
    if result_path.exists(): raise ValueError('Never resume or replace an attempted comparison')
    report = dict(status='running',freezeSHA256=sha(freeze_path),plannedGames=16,plannedRounds=220,games=[])
    save(result_path,report)
    try:
        for game in freeze['schedule']:
            verify_inputs(freeze)
            if shutil.disk_usage(evidence).free < 2*1024**3: raise ValueError('Insufficient disk; no new child')
            run = evidence/'games'/game['id']; run.mkdir(parents=True)
            botdirs,identities=stage_bots(game,source,evidence,freeze,run)
            home=run/'home';home.mkdir()
            env={k:os.environ[k] for k in ('TMPDIR','LANG','LC_ALL','TZ') if k in os.environ}
            env.update(HOME=str(home),JAVA_HOME=str(java.parent.parent),PATH=str(java.parent)+':/usr/bin:/bin:/opt/homebrew/bin',
                ROBOCODE_PYTHON_BIN=str(python),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                ROBOCODE_ENV_FILE=str(run/'absent.env'),ROBOCODE_GUN_ENV_FILE=str(run/'absent-guns.env'),
                RUMBLE_JAVA_CACHE=str(source/'.benchmark-cache/retcon-api'),KEEL_HELM_METRICS=str(run/'helm-metrics.txt'),
                KEEL_HELM_SEED=str(game['seed']),KEEL_SEED=str(game['seed']),PRISM_SEED=str(game['seed']))
            rec=run/'recordings';rec.mkdir()
            command=[str(java),'-Dtournament.recordingDir='+str(rec),'-cp',str(source/'.benchmark-cache/robocode-tankroyale-runner-1.4.0.jar'),
                     str(ROOT/'tournament/scripts/TournamentBattle.java'),str(game['rounds']),'900',str(run/'battle.json'),*botdirs]
            row=dict(**game,status='attempted');report['games'].append(row);save(result_path,report)
            save(run/'invocation.json',dict(command=command,environment=env));print('START '+game['id'],flush=True)
            execute(command,env,run)
            official=json.loads((run/'battle.json').read_text())
            if official['numberOfRounds']!=game['rounds'] or sorted((r['name'],r['version']) for r in official['results'])!=sorted(identities): raise ValueError('Official coverage/identity differs')
            recordings=list(rec.glob('*.gz'))
            if len(recordings)!=1: raise ValueError('Expected one original recording')
            proof=audit(recordings[0],official,game,identities)
            if proof['metrics']['skippedTurns']: raise ValueError('Own observer skip')
            rows=telemetry(run/'helm-metrics.txt',game['rounds'])
            own=next(r for r in official['results'] if r['name']=='KeelHelm')
            row.update(status='complete',official=official,helm=own,scoreShare=own['totalScore']/sum(r['totalScore'] for r in official['results']),
                       recording=str(recordings[0].relative_to(evidence)),audit=proof,telemetry=rows)
            save(result_path,report);print('END '+game['id']+' share='+str(round(row['scoreShare']*100,2)),flush=True)
        report.update(status='complete',completedGames=len(report['games']),completedRounds=sum(g['rounds'] for g in report['games']),summary=summary(report['games']))
    except BaseException as error:
        report.update(status='invalid',error=str(error));raise
    finally: save(result_path,report)
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--pins',type=Path);p.add_argument('--java',type=Path);p.add_argument('--python',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--approval-sha')
    p.add_argument('--stage-only',action='store_true')
    args=p.parse_args()
    if args.prepare and any(v is None for v in (args.pins,args.java,args.python)):
        p.error('--prepare requires --pins, --java and --python')
    main(args)
