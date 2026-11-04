"""Metadata-only base Python source/cache/native proof; never import tested code.

All regular stdlib sources AND existing bytecode caches are pinned, including
lib-dynload; site/dist-packages are excluded here and pinned separately. Nothing
is deleted, regenerated or repaired. Builtin/frozen implementations are bound to
the exact executable/libpython/native-runtime files. Native DT_NEEDED closure is
resolved with the hosted image's ldd metadata mode during collection only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

POLICY = 'stdlib source+all existing pyc+lib-dynload; exclude site/dist-packages; include all links; native ldd closure+loader config/cache; no writes/repair'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base():
    return Path(sys.base_prefix).resolve(strict=True)


def key(path):
    path = Path(path).absolute()
    if path.is_relative_to(base()):
        return 'base/' + path.relative_to(base()).as_posix()
    return 'system/' + path.as_posix().lstrip('/')


def path_for(value):
    prefix, relative = value.split('/', 1)
    if prefix not in ('base', 'system') or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Unsafe runtime proof key')
    return (base() if prefix == 'base' else Path('/')) / relative


def row(path):
    return dict(sha256=digest(path) if path.is_file() else None,
                symlink=os.readlink(path) if path.is_symlink() else None)


def stdlib():
    folder = Path(sysconfig.get_path('stdlib')).resolve(strict=True)
    if not folder.is_relative_to(base()):
        raise ValueError('Stdlib outside owned base Python prefix')
    return folder


def stdlib_inventory():
    result = {}
    for directory, folders, files in os.walk(stdlib(), followlinks=False):
        folders[:] = sorted(name for name in folders if name not in ('site-packages', 'dist-packages'))
        for name in sorted(set(folders + files)):
            path = Path(directory) / name
            if path.is_symlink() or path.is_file():
                result[key(path)] = row(path)
    return result


def add_with_links(files, path):
    path = Path(path).absolute()
    files[key(path)] = row(path)
    resolved = path.resolve(strict=True)
    files[key(resolved)] = row(resolved)


def loader_configuration():
    paths = [Path('/etc/ld.so.cache'), Path('/etc/ld.so.conf'), Path('/etc/ld.so.preload')]
    paths += list(Path('/etc/ld.so.conf.d').rglob('*'))
    if Path('/etc/ld.so.preload').exists() and Path('/etc/ld.so.preload').read_text().strip():
        raise ValueError('Native preload configuration is not permitted')
    return {key(path): row(path) for path in sorted(paths) if path.is_file() or path.is_symlink()}


def startup_paths():
    result = []
    for value in sys.path:
        if not value:
            raise ValueError('Ambient current-directory startup import path')
        path = Path(value).absolute()
        if path.is_file() or path.is_symlink():
            raise ValueError('Archive/symlink startup paths are outside this diagnostic loader contract')
        result.append(dict(key=key(path), kind='directory' if path.is_dir() else 'absent'))
    return result


def collect():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode and sys.flags.no_site):
        raise ValueError('Use owned Python -I -B -S for runtime metadata')
    files = stdlib_inventory()
    executable = Path(sys.executable).resolve(strict=True)
    add_with_links(files, executable)
    library = Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY')
    if not library.is_file():
        raise ValueError('Expected actual shared libpython')
    add_with_links(files, library)
    native = {executable, library.resolve()}
    native.update(path_for(name).resolve() for name in files
                  if path_for(name).is_file() and path_for(name).name.endswith('.so'))
    dependency_keys = set()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'DYLD_', 'PYTHON'))}
    for path in sorted(native):
        output = subprocess.check_output(['/usr/bin/ldd', str(path)], env=env, text=True,
                                         stderr=subprocess.STDOUT, timeout=15)
        if 'not found' in output:
            raise ValueError('Unresolved native Python dependency: ' + str(path))
        for line in output.splitlines():
            match = re.search(r'=>\s+(/\S+)', line) or re.match(r'\s*(/\S+)', line)
            if match:
                dependency = Path(match.group(1))
                add_with_links(files, dependency)
                dependency_keys.update((key(dependency), key(dependency.resolve())))
    config = loader_configuration()
    for name in config:
        add_with_links(files, path_for(name))
    proof = dict(schemaVersion=1, policy=POLICY, stdlibRelative=stdlib().relative_to(base()).as_posix(),
        executableKey=key(executable), libpythonKey=key(library.resolve()),
        metadataToolSHA256=digest('/usr/bin/ldd'), nativeDependencyKeys=sorted(dependency_keys),
        nativeMetadataInputs=sorted(key(path) for path in native),
        nativeMetadataCommand=['/usr/bin/ldd', '<resolved nativeMetadataInputs entry>'],
        virtualNativePolicy='linux-vdso has no file; kernel implementation bound to hosted boot identity; not a separately hashed file',
        loaderConfiguration=config,
        startupPaths=startup_paths(),
        files=dict(sorted(files.items())))
    proof['builtinFrozenBinding'] = dict(executableKey=proof['executableKey'],
        executableSHA256=files[proof['executableKey']]['sha256'], libpythonKey=proof['libpythonKey'],
        libpythonSHA256=files[proof['libpythonKey']]['sha256'],
        nativeRuntime={name: files[name] for name in sorted(dependency_keys)})
    verify(proof)
    return proof


def verify(proof):
    """Read-only rehash before tested imports; return resolved allowed path→SHA."""
    if proof.get('schemaVersion') != 1 or proof.get('policy') != POLICY:
        raise ValueError('Unknown Python runtime proof policy')
    if stdlib().relative_to(base()).as_posix() != proof['stdlibRelative']:
        raise ValueError('Actual base stdlib layout differs')
    if loader_configuration() != proof['loaderConfiguration']:
        raise ValueError('Native loader configuration/cache changed')
    if startup_paths() != proof['startupPaths']:
        raise ValueError('Interpreter startup import paths changed')
    actual = stdlib_inventory()
    expected = {k: v for k, v in proof['files'].items()
                if k.startswith('base/' + proof['stdlibRelative'] + '/')}
    if actual != expected:
        raise ValueError('Base stdlib/source/cache/native file inventory changed')
    allowed = {}
    for name, expected_row in proof['files'].items():
        path = path_for(name)
        if not (path.is_file() or path.is_symlink()) or row(path) != expected_row:
            raise ValueError('Base Python runtime file/link changed: ' + name)
        if path.is_file():
            resolved = str(path.resolve(strict=True))
            if resolved in allowed and allowed[resolved] != expected_row['sha256']:
                raise ValueError('Ambiguous runtime proof target')
            allowed[resolved] = expected_row['sha256']
    if (key(Path(sys.executable).resolve()) != proof['executableKey']
            or proof['libpythonKey'] not in proof['files']
            or digest('/usr/bin/ldd') != proof['metadataToolSHA256']):
        raise ValueError('Actual interpreter/libpython/metadata tool identity differs')
    binding = dict(executableKey=proof['executableKey'],
        executableSHA256=proof['files'][proof['executableKey']]['sha256'],
        libpythonKey=proof['libpythonKey'], libpythonSHA256=proof['files'][proof['libpythonKey']]['sha256'],
        nativeRuntime={name: proof['files'][name] for name in proof['nativeDependencyKeys']})
    if proof.get('builtinFrozenBinding') != binding:
        raise ValueError('Builtin/frozen implementation binding differs')
    return allowed


def resolve_files(proof):
    return verify(proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', action='store_true')
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.collect:
        proof = collect()
        print(json.dumps(proof, sort_keys=True))
    else:
        verify(json.loads(args.verify.read_text()))
