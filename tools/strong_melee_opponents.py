"""Pinned native Diamond Port/Adaptive Prime packaging; no opponent logic changes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import retcon

LOCK=Path(__file__).resolve().parent/'opponents/strong-melee.lock.json'


def manifest():return json.loads(LOCK.read_bytes())


def clean_environment(environment):
    return {k:v for k,v in environment.items() if not k.startswith(('AISURVIVOR_','ROBOCODE_','PYTHON','PIP_'))
            and k not in ('RETCON_STATS_DIR','RUMBLE_JAVA_CACHE')}


def prepare(cache,python):
    lock=manifest();base=cache.resolve()/'strong-python'/lock['commit'];source=base/'source';source.mkdir(parents=True,exist_ok=True)
    def fetch(entry):
        target=source/retcon.checked_relative(entry['path'])
        if target.is_symlink() or not target.resolve().is_relative_to(source):raise ValueError('Unsafe opponent cache')
        data=target.read_bytes() if target.exists() else b''
        if not target.is_file() or len(data)!=entry['size'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            data=retcon.read_url(f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}/{entry['path']}",entry['size'])
            if len(data)!=entry['size'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Pinned source mismatch: '+entry['path'])
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    with ThreadPoolExecutor(max_workers=8) as executor:list(executor.map(fetch,lock['files']))
    if {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}!={e['path'] for e in lock['files']}:
        raise ValueError('Unexpected opponent source/config files')
    canonical=''.join(e['path']+'\0'+e['sha256']+'\n' for e in sorted(lock['files'],key=lambda e:e['path']))
    if hashlib.sha256(canonical.encode()).hexdigest()!=lock['sourceSHA256']:raise ValueError('Source manifest tree hash mismatch')
    for bot in lock['opponents']:
        descriptor=json.loads((source/bot['sourcePath']/bot['descriptor']).read_bytes())
        if (descriptor['name'],descriptor['version'])!=(bot['name'],bot['version']) or 'melee' not in descriptor['gameTypes']:
            raise ValueError('Pinned native bot identity/melee eligibility differs')
    for script in source.rglob('*.sh'):script.chmod(0o755)
    wheels=base/'wheels';wheels.mkdir(exist_ok=True);paths=[]
    for entry in lock['runtime']['wheels']:
        path=wheels/retcon.checked_relative(entry['filename']);data=path.read_bytes() if path.exists() else b''
        if len(data)!=entry['size'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            data=retcon.read_url(entry['url'],entry['size'])
            if len(data)!=entry['size'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Pinned wheel mismatch')
            path.write_bytes(data)
        paths.append(str(path))
    env=clean_environment(os.environ);env['PYTHONNOUSERSITE']='1'
    subprocess.run([python,'-c','import sys; assert sys.version_info[:2]==(3,13)'],env=env,check=True,timeout=15)
    venv=base/'venv'
    if venv.exists():raise ValueError('Opponent venv must be fresh; never reuse ambient packages')
    subprocess.run([python,'-m','venv',str(venv)],env=env,check=True,timeout=120)
    executable=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    subprocess.run([str(executable),'-m','pip','install','--no-index','--no-deps','--no-input','--disable-pip-version-check',*paths],env=env,check=True,timeout=240)
    checked=subprocess.run([str(executable),'-m','pip','check'],env=env,check=True,capture_output=True,text=True,timeout=30)
    runtime=subprocess.check_output([str(executable),'-c','import sys,importlib.metadata as m,json;print(json.dumps(dict(version=sys.version,packages=sorted((d.metadata["Name"],d.version) for d in m.distributions()))))'],env=env,text=True,timeout=30)
    return source,executable,dict(repository=lock['repository'],commit=lock['commit'],sourceSHA256=lock['sourceSHA256'],
        lockSHA256=hashlib.sha256(LOCK.read_bytes()).hexdigest(),opponents=lock['opponents'],wheels=lock['runtime']['wheels'],runtime=json.loads(runtime),pipCheck=checked.stdout.strip())


def stage(source,run_dir,python):
    destination=run_dir/'opponents';destination.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'bots/bot_core',destination/'bot_core')
    shutil.copyfile(source/'LICENSE',destination/'LICENSE.shared-Apache-2.0')
    return dict(ROBOCODE_PYTHON_BIN=str(python),ROBOCODE_ENV_FILE=str(run_dir/'absent-developer.env'),
                ROBOCODE_GUN_ENV_FILE=str(run_dir/'absent-developer-guns.env'),PYTHONNOUSERSITE='1')


def verify_staged(run_dir,names):
    lock=manifest()
    if not names or not set(names).issubset({Path(b['descriptor']).stem for b in lock['opponents']}):raise ValueError('Unknown native opponent')
    prefixes={bot['sourcePath']:Path('opponents')/Path(bot['descriptor']).stem for bot in lock['opponents'] if Path(bot['descriptor']).stem in names}
    prefixes[lock['sharedSourcePath']]=Path('opponents/bot_core');hashes={}
    for entry in lock['files']:
        for prefix,destination in prefixes.items():
            if entry['path'].startswith(prefix+'/'):
                relative=destination/entry['path'][len(prefix)+1:];path=run_dir/relative
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if digest!=entry['sha256']:raise ValueError('Executed opponent source changed: '+str(relative))
                hashes[relative.as_posix()]=digest
    notice=run_dir/'opponents/LICENSE.shared-Apache-2.0';digest=hashlib.sha256(notice.read_bytes()).hexdigest()
    if digest!=next(e['sha256'] for e in lock['files'] if e['path']=='LICENSE'):raise ValueError('Shared license changed')
    hashes['opponents/LICENSE.shared-Apache-2.0']=digest
    return hashes
