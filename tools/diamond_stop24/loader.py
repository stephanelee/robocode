"""Reviewed acquisition-only source loader. No tested import occurs at module load."""
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

SHELLS = ('robocode_tank_royale', 'robocode_tank_royale.bot_api',
          'robocode_tank_royale.bot_api.internal', 'robocode_tank_royale.bot_api.util',
          'robocode_tank_royale.bot_api.graphics', 'robocode_tank_royale.schema', 'diamond_port')
SDK_LOAD = (
    ('robocode_tank_royale.bot_api.bot_exception', 'robocode_tank_royale/bot_api/bot_exception.py'),
    ('robocode_tank_royale.bot_api.constants', 'robocode_tank_royale/bot_api/constants.py'),
    ('robocode_tank_royale.bot_api.util.math_util', 'robocode_tank_royale/bot_api/util/math_util.py'),
    ('robocode_tank_royale.schema.color', 'robocode_tank_royale/schema/color.py'),
    ('robocode_tank_royale.bot_api.graphics.color', 'robocode_tank_royale/bot_api/graphics/color.py'),
    ('robocode_tank_royale.bot_api.internal.intent_validator', 'robocode_tank_royale/bot_api/internal/intent_validator.py'))
PORT_LOAD = (('diamond_port.dia_utils', 'diamond_port/dia_utils.py'),
             ('diamond_port.movement_predictor', 'diamond_port/movement_predictor.py'))
FORBIDDEN = {'socket', '_socket', 'ssl', '_ssl', 'websocket', 'websockets', 'asyncio',
             'requests', 'urllib', 'http', 'ctypes', 'multiprocessing'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ClosureGuard:
    def __init__(self, runtime_files, builtin_binding, proof_sha256):
        self.finders = tuple(sys.meta_path)
        if self.finders != (importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter, importlib.machinery.PathFinder):
            raise ImportError("Unexpected interpreter meta-path hooks")
        if sys.pycache_prefix is not None:
            raise ImportError('Nondefault bytecode cache prefix is not permitted')
        self.runtime_files = dict(runtime_files)
        if not self.runtime_files or any(not Path(name).is_absolute() for name in self.runtime_files):
            raise ImportError('Missing absolute frozen base-runtime file proof')
        self.binding_reference = dict(pythonRuntimeProofSHA256=proof_sha256,
            executableKey=builtin_binding['executableKey'],
            executableSHA256=builtin_binding['executableSHA256'],
            libpythonKey=builtin_binding['libpythonKey'],
            libpythonSHA256=builtin_binding['libpythonSHA256'])
        executable = self.bound_file(sys.executable)
        if self.runtime_files[str(executable)] != builtin_binding['executableSHA256']:
            raise ImportError('Builtin/frozen interpreter binary binding differs')

    def bound_file(self, path):
        resolved = Path(path).resolve(strict=True)
        expected = self.runtime_files.get(str(resolved))
        if expected is None or sha(resolved) != expected:
            raise ImportError('Unlisted/changed frozen Python runtime file: ' + str(resolved))
        return resolved

    def bound_cache(self, path):
        if not path:
            return None
        candidate = Path(path)
        # Missing caches cannot be used or created under -B. Existing cache bytes must be frozen.
        if candidate.exists() or candidate.is_symlink():
            return self.bound_file(candidate)
        return None

    def bound_interpreter_spec(self, spec):
        required = (importlib.machinery.BuiltinImporter if spec.origin == 'built-in'
                    else importlib.machinery.FrozenImporter)
        if spec.origin not in ('built-in', 'frozen') or spec.loader is not required:
            raise ImportError('Unexpected interpreter-bound import loader: ' + str(spec.name))

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in FORBIDDEN or root in ('diamond_port', 'robocode_tank_royale'):
            raise ImportError('Unreviewed implicit import: ' + fullname)
        for finder in self.finders:
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            origin = spec.origin
            if origin in ('built-in', 'frozen'):
                self.bound_interpreter_spec(spec)
                return spec  # implementation bound to frozen executable/libpython/native proof
            if isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                self.bound_file(origin)
                self.bound_cache(spec.cached)
                return spec
            if isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
                self.bound_file(origin)
                return spec  # all native dependencies/resolution metadata prefrozen by host proof
            # No sourceless .pyc/zip/third-party loader, even if it resolves inside a stdlib prefix.
            raise ImportError('Unreviewed source/cache/native loader: ' + fullname + ' ' + str(origin))
        raise ModuleNotFoundError(fullname)


def source_module(name, path, expected):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected or name in sys.modules:
        raise ImportError('Module identity/duplicate differs: ' + name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    module.__spec__ = importlib.util.spec_from_file_location(name, path)
    sys.modules[name] = module
    # Compile the validated source bytes directly: never use ambient/stale .pyc files.
    exec(compile(data, str(path), 'exec', dont_inherit=True), module.__dict__)
    parent, _, child = name.rpartition('.')
    if parent:
        setattr(sys.modules[parent], child, module)
    return module


def load(sdk_site, port_root, pins, runtime_files, builtin_binding, proof_sha256, own_files):
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode or not sys.flags.no_site:
        raise RuntimeError('Evaluator requires -I -B -S')
    if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError('Forbidden bootstrap module already present')
    guard = ClosureGuard(runtime_files, builtin_binding, proof_sha256)
    # The host has verified the proof before this interpreter starts. Recheck its already-loaded
    # source/cache/native origins before the first tested import, then guard future import execution.
    inventory(guard, own_files, pins, sdk_site, port_root)
    for name in SHELLS:
        if name in sys.modules:
            raise ImportError('SDK/port bootstrap already imported: ' + name)
        module = types.ModuleType(name)
        module.__path__ = []
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
        parent, _, child = name.rpartition('.')
        if parent:
            setattr(sys.modules[parent], child, module)
    sys.meta_path.insert(0, guard)
    loaded = {}
    for name, relative in SDK_LOAD:
        module = source_module(name, Path(sdk_site) / relative, pins['sdk'][relative])
        loaded[name] = module
        if name == 'robocode_tank_royale.schema.color':
            sys.modules['robocode_tank_royale.schema'].Color = module.Color
        elif name == 'robocode_tank_royale.bot_api.graphics.color':
            sys.modules['robocode_tank_royale.bot_api.graphics'].Color = module.Color
    for name, relative in PORT_LOAD:
        loaded[name] = source_module(name, Path(port_root) / relative, pins['port'][relative])
    validator = loaded['robocode_tank_royale.bot_api.internal.intent_validator'].IntentValidator
    dia = loaded['diamond_port.dia_utils']
    predictor = loaded['diamond_port.movement_predictor'].MovementPredictor
    if validator.__module__ != 'robocode_tank_royale.bot_api.internal.intent_validator' or predictor.__module__ != 'diamond_port.movement_predictor':
        raise ImportError('Unexpected tested class origin')
    return validator, dia, predictor, guard


def inventory(guard, own_files, pins, sdk_site, port_root):
    expected = {str((Path(sdk_site) / relative).resolve()): value for relative, value in pins['sdk'].items()}
    expected.update({str((Path(port_root) / relative).resolve()): value for relative, value in pins['port'].items()})
    expected.update({str(Path(path).resolve()): value for path, value in own_files.items()})
    result = {}
    for name, module in sorted(sys.modules.items()):
        if module is None:
            continue
        if name.split('.')[0] in FORBIDDEN:
            raise RuntimeError('Forbidden imported module: ' + name)
        file = getattr(module, '__file__', None)
        spec = getattr(module, '__spec__', None)
        origin = getattr(spec, 'origin', None)
        if origin in ('built-in', 'frozen'):
            guard.bound_interpreter_spec(spec)
        if name in SHELLS:
            result[name] = dict(origin='reviewed namespace shell; package initializer NOT executed')
        elif file:
            path = Path(file).resolve(strict=True)
            value = sha(path)
            if str(path) in expected:
                if value != expected[str(path)]:
                    raise RuntimeError('Imported source bytes changed: ' + name)
            else:
                guard.bound_file(path)
            row = dict(origin=origin, path=str(path), sha256=value)
            if origin in ('built-in', 'frozen'):
                row['interpreterCodeBinding'] = guard.binding_reference
                row['pathRole'] = 'exposed source metadata; implementation is interpreter-bound'
            cached = getattr(module, '__cached__', None)
            if cached:
                actual_cache = guard.bound_cache(cached)
                if actual_cache is not None:
                    row['cachedPath'] = str(actual_cache)
                    row['cachedSHA256'] = sha(actual_cache)
                    row['cachePolicy'] = 'potential cache is frozen; -B prevents creation/repair'
                else:
                    row['cachePolicy'] = 'absent cache; cannot be read/created under -B'
                    row['claimedCachePath'] = str(Path(cached).absolute())
            result[name] = row
        elif name in ('typing.io', 'typing.re') and getattr(sys.modules.get('typing'), name.rsplit('.', 1)[1], None) is module:
            owner = Path(sys.modules['typing'].__file__).resolve(strict=True)
            guard.bound_file(owner)
            result[name] = dict(origin='stdlib typing compatibility alias', ownerPath=str(owner), ownerSHA256=sha(owner))
        elif origin in ('built-in', 'frozen'):
            result[name] = dict(origin=origin, interpreterCodeBinding=guard.binding_reference)
        else:
            raise RuntimeError('Unknown origin for imported module: ' + name)
    return result


def capture_modules_on_failure():
    """Best-effort unvalidated provenance, including unexpected/partially imported modules."""
    result = {}
    for name, module in sorted(sys.modules.items()):
        if module is None:
            result[name] = dict(origin='None entry during failed import')
            continue
        entry = dict(origin=getattr(getattr(module, '__spec__', None), 'origin', None),
                     objectType=type(module).__name__)
        for attribute, prefix in (('__file__', 'source'), ('__cached__', 'cached')):
            value = getattr(module, attribute, None)
            if value:
                try:
                    path = Path(value).resolve(strict=True)
                    entry[prefix + 'Path'] = str(path)
                    entry[prefix + 'SHA256'] = sha(path)
                except Exception as error:
                    entry[prefix + 'CaptureError'] = repr(error)
        if name in SHELLS:
            entry['origin'] = 'reviewed namespace shell; initializer NOT executed'
        result[name] = entry
    return result
