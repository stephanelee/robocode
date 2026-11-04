"""Read-only validation of a freshly owned Peak environment; never prepare or repair it."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import benchmark
import retcon
import strong_melee_opponents as native
from meridian_peak_common import ROOT, CACHE

OS_RELEASE = Path('/etc/os-release')
VENV_INVENTORY_POLICY = 'regular .pyc/__pycache__ excluded; every symlink identity included'


def checked_file(path, expected, size=None):
    if path.is_symlink() or not path.is_file() or (size is not None and path.stat().st_size != size) or benchmark.digest(path) != expected:
        raise ValueError('Prepared pinned file changed or is missing: ' + str(path))


def native_layout():
    lock = native.manifest()
    base = CACHE.resolve() / 'strong-python' / lock['commit']
    source, venv = base / 'source', base / 'venv'
    return lock, base, source, venv, venv / 'bin/python'


def venv_inventory(venv):
    """Bind non-cache files and EVERY link, including directory/dangling links.

    Regular bytecode caches retain the existing pinned package-fingerprint exclusion.
    Verification itself uses -B so it does not create or refresh those caches.
    """
    result = {}
    for path in sorted(venv.rglob('*')):
        if path.is_symlink() or (path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'):
            result[path.relative_to(venv).as_posix()] = dict(
                sha256=benchmark.digest(path) if path.is_file() else None,
                symlink=os.readlink(path) if path.is_symlink() else None)
    return result


def creation_proof():
    """Record ownership/config identity only immediately after successful fresh collection."""
    _, _, _, venv, executable = native_layout()
    java17 = Path((ROOT / 'build/meridian/build-java-home.txt').read_text().strip())
    return dict(schemaVersion=1, venv=str(venv.resolve()), pythonExecutable=str(executable),
                venvInventoryPolicy=VENV_INVENTORY_POLICY,
                pyvenvConfigSHA256=benchmark.digest(venv / 'pyvenv.cfg'),
                venvFiles=venv_inventory(venv),
                java17Home=str(java17), java17JavaSHA256=benchmark.digest(java17 / 'bin/java'),
                java17HomeProofSHA256=benchmark.digest(ROOT / 'build/meridian/build-java-home.txt'))


def verify_native(proof):
    lock, base, source, venv, executable = native_layout()
    if proof.get('venvInventoryPolicy') != VENV_INVENTORY_POLICY:
        raise ValueError('Prepared venv inventory policy differs')
    if source.is_symlink() or venv.is_symlink() or str(venv.resolve()) != proof['venv'] or str(executable) != proof['pythonExecutable']:
        raise ValueError('Prepared native environment is not the freshly owned workspace environment')
    expected_files = {entry['path'] for entry in lock['files']}
    if {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()} != expected_files:
        raise ValueError('Prepared native source inventory changed')
    for entry in lock['files']:
        path = source / retcon.checked_relative(entry['path'])
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('Prepared source escapes owned tree')
        checked_file(path, entry['sha256'], entry['size'])
    canonical = ''.join(entry['path'] + '\0' + entry['sha256'] + '\n' for entry in sorted(lock['files'], key=lambda e: e['path']))
    if hashlib.sha256(canonical.encode()).hexdigest() != lock['sourceSHA256']:
        raise ValueError('Native source catalog/tree pin mismatch')
    for bot in lock['opponents']:
        descriptor = json.loads((source / bot['sourcePath'] / bot['descriptor']).read_text())
        if (descriptor['name'], descriptor['version']) != (bot['name'], bot['version']) or 'melee' not in descriptor['gameTypes']:
            raise ValueError('Native source descriptor differs')
    wheels = base / 'wheels'
    if {p.name for p in wheels.iterdir() if p.is_file()} != {e['filename'] for e in lock['runtime']['wheels']}:
        raise ValueError('Prepared native wheel inventory changed')
    for entry in lock['runtime']['wheels']:
        checked_file(wheels / retcon.checked_relative(entry['filename']), entry['sha256'], entry['size'])
    config = venv / 'pyvenv.cfg'
    checked_file(config, proof['pyvenvConfigSHA256'])
    settings = dict(line.split('=', 1) for line in config.read_text().splitlines() if '=' in line)
    if {key.strip().lower(): value.strip().lower() for key, value in settings.items()}.get('include-system-site-packages') != 'false':
        raise ValueError('Ambient system packages enabled in prepared venv')
    if not executable.is_file():
        raise ValueError('Prepared Python executable is missing')
    if venv_inventory(venv) != proof['venvFiles']:
        raise ValueError('Prepared venv file bytes, links or file inventory changed')
    env = native.clean_environment(os.environ)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    # sys.prefix proves this exact venv is executing; -I rejects ambient PYTHONPATH/user packages.
    code = ('import sys,site,importlib.metadata as m,json;'
            f'assert sys.prefix=={str(venv)!r} and sys.base_prefix!=sys.prefix and not site.ENABLE_USER_SITE;'
            'assert sys.version_info[:2]==(3,13);'
            'print(json.dumps(dict(version=sys.version,packages=sorted((d.metadata["Name"],d.version) for d in m.distributions()))))')
    runtime = json.loads(subprocess.check_output([str(executable), '-I', '-B', '-c', code], env=env, text=True, timeout=30))
    checked = subprocess.run([str(executable), '-I', '-B', '-m', 'pip', 'check'], env=env, check=True,
                             capture_output=True, text=True, timeout=30)
    provenance = dict(repository=lock['repository'], commit=lock['commit'], sourceSHA256=lock['sourceSHA256'],
        lockSHA256=benchmark.digest(native.LOCK), opponents=lock['opponents'], wheels=lock['runtime']['wheels'],
        runtime=runtime, pipCheck=checked.stdout.strip())
    return executable, provenance


def verify_retcon():
    manifest = json.loads(retcon.LOCK.read_text())
    source = CACHE / 'retcon' / manifest['commit'] / 'Retcon'
    if source.is_symlink():
        raise ValueError('Retcon prepared tree is a symlink')
    if {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()} != {e['path'] for e in manifest['files']}:
        raise ValueError('Prepared Retcon source inventory changed')
    for entry in manifest['files']:
        path = source / retcon.checked_relative(entry['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('Unsafe prepared Retcon source')
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if len(raw) != entry['size'] or blob != entry['gitBlobSHA']:
            raise ValueError('Prepared Retcon Git blob changed')
    if retcon.source_hash(source) != manifest['sourceSHA256']:
        raise ValueError('Prepared Retcon source hash changed')
    descriptor = json.loads((source / 'Retcon.json').read_text())
    if (descriptor['name'], descriptor['version'], descriptor['license']) != ('Retcon', '0.2.0', 'MIT') or '1v1' not in descriptor['gameTypes']:
        raise ValueError('Prepared Retcon descriptor changed')
    dependency = manifest['botApi']
    checked_file(CACHE / 'retcon-api' / f"robocode-tankroyale-bot-api-{dependency['version']}.jar", dependency['sha256'])
    return {key: manifest[key] for key in ('name', 'version', 'repository', 'commit', 'sourcePath', 'sourceSHA256', 'license', 'authors', 'botApi')}


def package_contents(executable):
    code = '''import hashlib,importlib.metadata as m,json
result={}
for d in sorted(m.distributions(),key=lambda d:d.metadata['Name'].lower()):
 rows={}
 for f in sorted(d.files or [],key=str):
  p=d.locate_file(f)
  if p.is_file() and not str(f).endswith('.pyc') and '__pycache__' not in str(f):
   rows[str(f)]=hashlib.sha256(p.read_bytes()).hexdigest()
 result[d.metadata['Name']]={'version':d.version,'files':rows}
print(json.dumps(result,sort_keys=True))'''
    return json.loads(subprocess.check_output([str(executable), '-I', '-B', '-c', code], text=True, timeout=60))


def runtime(proof):
    python, provenance = verify_native(proof)
    rc = verify_retcon()
    old = ROOT / 'build/meridian'
    checked_file(old / 'build-java-home.txt', proof['java17HomeProofSHA256'])
    java17 = Path(proof['java17Home'])
    if (old / 'build-java-home.txt').read_text().strip() != str(java17):
        raise ValueError('Java 17 build root changed')
    checked_file(java17 / 'bin/java', proof['java17JavaSHA256'])
    build_java = benchmark.digest(java17 / 'bin/javac') + '  javac\n' + benchmark.digest(java17 / 'lib/modules') + '  modules\n'
    version = subprocess.check_output([str(java17 / 'bin/java'), '-version'], stderr=subprocess.STDOUT, text=True)
    if build_java != (old / 'build-java.sha256').read_text() or version != (old / 'build-java.txt').read_text():
        raise ValueError('Actual Java 17 binaries/version differ from build provenance')
    java25 = Path(os.environ['JAVA_HOME'])
    runner = CACHE / 'robocode-tankroyale-runner-1.4.0.jar'
    checked_file(runner, benchmark.ASSETS[runner.name])
    api = list((Path.home() / '.gradle/caches/modules-2/files-2.1/dev.robocode.tankroyale/robocode-tankroyale-bot-api/1.4.0').glob('*/*.jar'))
    if len(api) != 1:
        raise ValueError('Expected one prepared combat API jar')
    return dict(runner=benchmark.digest(runner), botApi140=benchmark.digest(api[0]),
        java25={name: benchmark.digest(java25 / name) for name in ('bin/java','lib/modules','release','lib/server/libjvm.so')},
        buildJava=build_java, java17Version=version, os=OS_RELEASE.read_text(), arch=platform.machine(),
        image=os.environ.get('ImageVersion'), pythonBinary=benchmark.digest(python.resolve()), python=provenance, retcon=rc), python
