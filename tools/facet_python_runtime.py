#!/usr/bin/env python3
"""Build the pinned CPython runtime on Ubuntu26.04; no battle launch or fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import urllib.request

VERSION = '3.13.9'
URL = 'https://www.python.org/ftp/python/3.13.9/Python-3.13.9.tar.xz'
SHA256 = 'ed5ef34cda36cfa2f3a340f07cac7e7814f91c7f3c411f6d3562323a866c5c66'
SIZE = 22681368
CONFIGURE = ('--with-ensurepip=install', '--disable-test-modules')
BUILD_ENV = dict(CC='gcc', CXX='g++', CFLAGS='-O3 -fno-omit-frame-pointer', CPPFLAGS='', LDFLAGS='')


def check_archive(path):
    if path.stat().st_size != SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Official CPython3.13.9 archive differs from frozen bytes')


def main(args):
    prefix, evidence = args.prefix.resolve(), args.evidence.resolve()
    if platform.machine() != 'x86_64' or 'VERSION_ID="26.04"' not in Path('/etc/os-release').read_text():
        raise ValueError('Exactly Ubuntu26.04 x86_64 required; no fallback')
    if prefix.exists() or evidence.exists():
        raise ValueError('Runtime preparation paths must be fresh; no automatic resume')
    evidence.mkdir(parents=True)
    work = prefix.parent/'facet-cpython-build'
    work.mkdir()
    report = dict(status='preparing', version=VERSION, sourceURL=URL, sourceSHA256=SHA256,
                  sourceBytes=SIZE, configure=list(CONFIGURE), explicitBuildEnvironment=BUILD_ENV,
                  makeJobs=2, optimization='O3, no PGO, no LTO; normal GIL, no experimental JIT',
                  license='PSF-2.0 and historical bundled notices; LICENSE copied to evidence')
    def save():
        (evidence/'runtime-build.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    env = os.environ.copy()
    for name in ('PYTHONHOME', 'PYTHONPATH', 'CONFIG_SITE', 'PROFILE_TASK'):
        env.pop(name, None)
    env.update(BUILD_ENV)
    with (evidence/'runtime-build.log').open('w') as log:
        def run(command, cwd=work, timeout=900):
            report.setdefault('commands', []).append(command)
            save()
            subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        try:
            archive = work/'Python-3.13.9.tar.xz'
            request = urllib.request.Request(URL, headers={'User-Agent':'Facet-pinned-runtime'})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(SIZE+1)
            archive.write_bytes(data)
            check_archive(archive)
            with tarfile.open(archive) as tar:
                tar.extractall(work, filter='data')
            source = work/'Python-3.13.9'
            shutil.copy2(source/'LICENSE', evidence/'CPython-LICENSE.txt')
            report['compiler'] = subprocess.check_output(['gcc','--version'],text=True)
            run(['./configure', '--prefix='+str(prefix), *CONFIGURE], cwd=source, timeout=300)
            run(['make','-j2'], cwd=source, timeout=1200)
            run(['make','install'], cwd=source, timeout=600)
            (prefix/'bin/python').symlink_to('python3.13')
            probe = 'import ssl,venv,ctypes,lzma,bz2,sqlite3,sys,sysconfig,json; assert sys.version_info[:3]==(3,13,9); print(json.dumps(dict(version=sys.version,ssl=ssl.OPENSSL_VERSION,configArgs=sysconfig.get_config_var("CONFIG_ARGS"),cflags=sysconfig.get_config_var("CFLAGS"))))'
            report['observed'] = json.loads(subprocess.check_output([str(prefix/'bin/python'),'-c',probe],text=True,env=env))
            report['pythonExecutableSHA256'] = hashlib.sha256((prefix/'bin/python').read_bytes()).hexdigest()
            report['status'] = 'ready-before-any-game'
        except BaseException as error:
            report.update(status='invalid-preflight-no-games', error=str(error))
            raise
        finally:
            save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    main(p.parse_args())
