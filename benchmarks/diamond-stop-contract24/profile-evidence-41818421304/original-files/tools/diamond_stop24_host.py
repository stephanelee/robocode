"""Hosted-only STOP24 preparation/identity and once-only acquisition. No physics in preparation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark
import strong_melee_opponents as native
import meridian_peak_verify as prepared

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'benchmarks/diamond-stop-contract24'
OUT = ROOT / 'build/diamond-stop24'
CACHE = ROOT / '.benchmark-cache'
PROTOCOL = ROOT / 'docs/improvements/diamond-stop-contract24.md'
REF = 'refs/tags/experiment-receipts/diamond-stop-contract24-v1'
PACKAGE = '52b7bfcf10f8226711798f27784be64583ea4c8fc676452d27f6e16b906bf191'
SERVER = '16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c'
BRANCH = 'codex/diamond-stop-contract24'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load(path):
    return json.loads(path.read_text())


def clean_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith(('PYTHON', 'PIP_', 'ROBOCODE_', 'AISURVIVOR_', 'LD_', 'DYLD_'))
            and k not in ('CLASSPATH', 'JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'JDK_JAVA_OPTIONS', 'GH_TOKEN', 'GITHUB_TOKEN')}


def run(command, *, timeout=120, **kwargs):
    return subprocess.check_output([str(x) for x in command], env=clean_env(),
                                   text=True, stderr=subprocess.STDOUT, timeout=timeout, **kwargs)


def inventory(folder):
    return {str(p.relative_to(folder)): dict(sha256=benchmark.digest(p) if p.is_file() else None,
              symlink=os.readlink(p) if p.is_symlink() else None)
            for p in sorted(folder.rglob('*')) if p.is_symlink() or p.is_file()}


def closure():
    paths = [PROTOCOL, REG / 'cases.json', ROOT / '.github/workflows/diamond-stop24.yml',
             native.LOCK, ROOT / 'tools/benchmark.py', ROOT / 'tools/retcon.py', ROOT / 'tools/retcon_stats.py',
             ROOT / 'tools/strong_melee_opponents.py', ROOT / 'tools/meridian_peak_verify.py',
             ROOT / 'tools/meridian_peak_common.py', ROOT / 'tools/compare.py',
             ROOT / 'benchmarks/meridian-gun-plan/package-contents.json']
    paths += list((ROOT / 'tools').glob('diamond_stop24*.py'))
    paths += [p for p in (ROOT / 'tools/diamond_stop24').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += list((ROOT / 'benchmarks/meridian-wave-support/engine-source').rglob('*'))
    paths += [ROOT / 'benchmarks/meridian-aim' / name for name in ('profile.json', 'profile-20261026.json')]
    return {str(p.relative_to(ROOT)): benchmark.digest(p) for p in sorted(set(paths)) if p.is_file()}


def expected_runtime(actual):
    for name in ('profile.json', 'profile-20261026.json'):
        old = load(ROOT / 'benchmarks/meridian-aim' / name)
        expected = {k: old[k] for k in ('image', 'os', 'arch', 'buildJava', 'java17Version', 'pythonBinary', 'python')}
        # The selected exact profile constrains every preexisting runtime field;
        # additional JDK17 binary hashes are bound by the independently reviewed new freeze.
        if {k: actual[k] for k in expected} == expected:
            return name
    raise ValueError('Runtime is outside both preallowed exact Ubuntu26.04 image profiles')


def identity():
    if (os.environ.get('GITHUB_RUN_ATTEMPT') != '1' or not os.environ.get('GITHUB_RUN_ID')
            or not os.environ.get('GITHUB_ACTIONS') or platform.machine() != 'x86_64'):
        raise ValueError('Hosted x86_64 attempt1 only')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if not boot:
        raise ValueError('Missing boot identity')
    return dict(bootID=boot,
                workspace=str(ROOT.resolve()), runID=os.environ['GITHUB_RUN_ID'], runAttempt=1)


def current_profile():
    proof = load(OUT / 'prepared.json')
    python, provenance = prepared.verify_native(proof)
    java = Path(proof['javaHome'])
    runtime = dict(image=os.environ.get('ImageVersion'), os=Path('/etc/os-release').read_text(),
        arch=platform.machine(), buildJava=benchmark.digest(java / 'bin/javac') + '  javac\n' + benchmark.digest(java / 'lib/modules') + '  modules\n',
        java17Version=run([java / 'bin/java', '-version']), pythonBinary=benchmark.digest(python.resolve()), python=provenance,
        java17={name: benchmark.digest(java / name) for name in ('bin/java', 'bin/javac', 'lib/modules', 'release', 'lib/server/libjvm.so')},
        hostTools={name: benchmark.digest(Path(shutil.which(name)).resolve()) for name in ('gh', 'git', 'timeout')})
    selected = expected_runtime(runtime)
    if '17.0.21+9' not in runtime['java17Version']:
        raise ValueError('Wrong JDK')
    packages = prepared.package_contents(python)
    expected = ROOT / 'benchmarks/meridian-gun-plan/package-contents.json'
    if benchmark.digest(expected) != PACKAGE or packages != load(expected):
        raise ValueError('Actual non-cache installed package bytes differ')
    if benchmark.digest(OUT / 'core/server.jar') != SERVER:
        raise ValueError('Pinned engine JAR changed')
    run([python, '-I', '-B', '-S', ROOT / 'tools/diamond_stop24_python_proof.py',
         '--verify', OUT / 'python-runtime.json'])
    return dict(runtime=runtime, selectedProfile=selected, packageSHA256=PACKAGE,
                pythonRuntime=load(OUT / 'python-runtime.json'), core=inventory(OUT / 'core'), tooling=closure())


def host():
    return dict(**identity(), profileSHA256=benchmark.digest(OUT / 'profile.json'),
                preparationSHA256=benchmark.digest(OUT / 'prepared.json'),
                locationsSHA256=benchmark.digest(OUT / 'locations.json'),
                pythonRuntimeSHA256=benchmark.digest(OUT / 'python-runtime.json'),
                freezeSHA256=benchmark.digest(REG / 'freeze.json') if (REG / 'freeze.json').exists() else None)


def collect():
    if OUT.exists():
        raise FileExistsError('Preparation root already exists; retain it, never reuse or repair')
    OUT.mkdir(parents=True)
    save(OUT / 'initial-host.json', identity())
    source, python, _ = native.prepare(CACHE, sys.executable)
    venv = python.parent.parent
    java = Path(os.environ['JAVA_HOME']).resolve()
    proof = dict(schemaVersion=1, venv=str(venv.resolve()), pythonExecutable=str(python),
        venvInventoryPolicy=prepared.VENV_INVENTORY_POLICY,
        pyvenvConfigSHA256=benchmark.digest(venv / 'pyvenv.cfg'), venvFiles=prepared.venv_inventory(venv),
        javaHome=str(java))
    save(OUT / 'prepared.json', proof)
    proof_command = [str(python), '-I', '-B', '-S', str(ROOT / 'tools/diamond_stop24_python_proof.py'), '--collect']
    save(OUT / 'runtime-proof-command.json', proof_command)
    python_runtime = json.loads(run(proof_command))
    save(OUT / 'python-runtime.json', python_runtime)
    core = OUT / 'core'
    core.mkdir()
    runner = benchmark.download(CACHE, 'robocode-tankroyale-runner-1.4.0.jar', benchmark.ASSETS['robocode-tankroyale-runner-1.4.0.jar'])
    with zipfile.ZipFile(runner) as z:
        matches = [n for n in z.namelist() if n.endswith('/robocode-tankroyale-server.jar') or n == 'robocode-tankroyale-server.jar']
        if len(matches) != 1:
            raise ValueError('Expected one pinned nested engine')
        raw = z.read(matches[0])
    if hashlib.sha256(raw).hexdigest() != SERVER:
        raise ValueError('Nested engine bytes differ')
    (core / 'server.jar').write_bytes(raw)
    # Compile/metadata only: the core preparation contract MUST NOT import SDK,
    # invoke predictors or execute Java bridge main/MathKt methods.
    sdk_site = venv / 'lib/python3.13/site-packages'
    command = [sys.executable, '-I', '-B', '-S', str(ROOT / 'tools/diamond_stop24/prepare.py'),
               '--java-home', str(java), '--server-jar', str(core / 'server.jar'),
               '--port-root', str(source / 'bots/ports/diamond-port'), '--sdk-site', str(sdk_site), '--output', str(core)]
    save(OUT / 'prepare-command.json', command)
    with (OUT / 'prepare.log').open('w') as log:
        subprocess.run(command, env=clean_env(), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    save(OUT / 'locations.json', dict(python=str(python), javaHome=str(java), serverJar=str(core / 'server.jar'),
         coreDir=str(core), portRoot=str(source / 'bots/ports/diamond-port'), sdkSite=str(sdk_site)))
    save(OUT / 'profile.json', current_profile())
    save(OUT / 'host.json', host())


def verify(frozen=False):
    if load(OUT / 'initial-host.json') != identity() or load(OUT / 'host.json') != host():
        raise ValueError('Boot/workspace/run/profile/preparation/freeze identity changed')
    profile = current_profile()
    if profile != load(OUT / 'profile.json'):
        raise ValueError('Actual runtime/source/package/compiled byte closure changed')
    if frozen and profile != load(REG / 'freeze.json')['profile']:
        raise ValueError('Actual profile differs from independently frozen zero-case profile')
    return profile


def freeze(profile_path):
    target = REG / 'freeze.json'
    if target.exists():
        raise FileExistsError('Never overwrite freeze')
    profile = load(profile_path)
    if profile['tooling'] != closure():
        raise ValueError('Profile tooling differs from current payload; obtain new zero-case preparation')
    with target.open('x') as fd:
        json.dump(dict(schemaVersion=1, profile=profile), fd, indent=2, sort_keys=True, allow_nan=False)
        fd.write('\n')


def api(endpoint, value=None):
    command = ['gh', 'api', endpoint]
    if value is not None:
        command += ['--method', 'POST', '--input', '-']
    return json.loads(subprocess.check_output(command, input=json.dumps(value) if value is not None else None,
                                              text=True, timeout=30))


def acquire():
    verify(True)
    head = run(['git', 'rev-parse', 'HEAD'], cwd=ROOT).strip()
    if head != os.environ.get('PAYLOAD_SHA') or os.environ.get('GITHUB_HEAD_REF') != BRANCH:
        raise ValueError('Wrong independently cleared payload/branch')
    repo = os.environ['GITHUB_REPOSITORY']
    pull = api(f"repos/{repo}/pulls/{os.environ['PULL_NUMBER']}")
    if (pull['head']['sha'] != head or pull['head']['repo']['full_name'] != repo
            or 'DIAMOND-STOP24-CLEARED ' + head not in (pull.get('body') or '')):
        raise ValueError('Exact-head clearance missing')
    jobs = api(f"repos/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}/jobs")
    save(OUT / 'admission-jobs.json', jobs)
    rows = [j for j in jobs['jobs'] if j['name'] == 'acquire']
    if len(rows) != 1:
        raise ValueError('Ambiguous acquisition clock')
    start = datetime.datetime.fromisoformat(rows[0]['started_at'].replace('Z', '+00:00'))
    if (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds() > 3000:
        raise ValueError('Fewer than10min remaining: no receipt or cases')
    destination = OUT / 'acquisition'
    if destination.exists() or destination.is_symlink() or (OUT / 'activation.json').exists():
        raise FileExistsError('Partial/existing acquisition permanently aborts')
    receipt = dict(experiment='diamond-stop-contract24-v1', payloadSHA=head, receiptRef=REF,
                   hostIdentity=host(), hostDocumentSHA256=benchmark.digest(OUT / 'host.json'),
                   profileSHA256=benchmark.digest(OUT / 'profile.json'), packageSHA256=PACKAGE,
                   freezeSHA256=benchmark.digest(REG / 'freeze.json'))
    tag = api(f'repos/{repo}/git/tags', dict(tag=REF.removeprefix('refs/tags/'), message=json.dumps(receipt, sort_keys=True), object=head, type='commit'))
    # Atomic create only. Never update/delete/reset; existing ref fails permanently.
    api(f'repos/{repo}/git/refs', dict(ref=REF, sha=tag['sha']))
    receipt['receiptObject'] = tag['sha']
    save(OUT / 'activation.json', receipt)
    verify(True)
    if (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds() > 3000:
        raise ValueError('Fewer than10min remain after activation/verification; preserve spent receipt, no cases')
    locations = load(OUT / 'locations.json')
    command = [locations['python'], '-I', '-B', '-S', str(ROOT / 'tools/diamond_stop24/evaluate.py'),
               '--prepared', str(OUT / 'core'), '--locations', str(OUT / 'locations.json'),
               '--activation', str(OUT / 'activation.json'), '--output', str(destination)]
    # Entire evaluator process tree, including bridge, is bounded by the outer timeout.
    command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '300s', *command]
    save(OUT / 'acquisition-command.json', command)
    with (OUT / 'acquisition.log').open('x') as log:
        subprocess.run(command, env=clean_env(), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=310)


if __name__ == '__main__':
    # Keep the GitHub API credential only in this wrapper; never pass it to SDK/Java children.
    for key in set(os.environ) - set(clean_env()) - {'GH_TOKEN', 'GITHUB_TOKEN'}:
        del os.environ[key]
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', action='store_true')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--freeze', type=Path, metavar='ZERO_CASE_PROFILE')
    group.add_argument('--acquire', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.verify:
        verify()
    elif args.freeze:
        freeze(args.freeze)
    else:
        acquire()
