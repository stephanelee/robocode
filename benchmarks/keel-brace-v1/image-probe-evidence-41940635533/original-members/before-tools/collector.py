#!/usr/bin/env python3
"""Bounded, zero-game image-metadata observation. No runtime acceptance or version mapping."""
import argparse
import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import time

FILE_CAP = 1024 * 1024
ARTIFACT_CAP = 10 * FILE_CAP
ZIP_OVERHEAD_RESERVE = 64 * 1024
PHASE_CAP = (ARTIFACT_CAP - ZIP_OVERHEAD_RESERVE) // 2
RAW_PHASE_CAP = 4 * FILE_CAP
REPORT_RESERVE = FILE_CAP
STATUS_RESERVE = 16 * 1024
PROC_LIMIT = 512
PROC_COMM_CAP = 128
RUNNER_MATCH_LIMIT = 16
JSON_GROUP_LIMIT = 128
DETAIL_CHAR_CAP = 65536
COLLECTION_SECONDS = 120
RUNNER_NAMES = {'Runner.Worker', 'Runner.Listener'}
FIXED_FILES = {
    'imagedata': '/imagegeneration/imagedata.json',
    'etc_environment': '/etc/environment',
    'etc_os_release': '/etc/os-release',
    'usr_lib_os_release': '/usr/lib/os-release',
    'etc_lsb_release': '/etc/lsb-release',
    'boot_id': '/proc/sys/kernel/random/boot_id',
}
CONTEXT_KEYS = ('GITHUB_REPOSITORY', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA',
                'GITHUB_REF', 'GITHUB_HEAD_REF', 'GITHUB_WORKFLOW', 'GITHUB_WORKFLOW_REF',
                'GITHUB_WORKFLOW_SHA', 'GITHUB_JOB', 'RUNNER_OS', 'RUNNER_ARCH',
                'KEEL_BRACE_IMAGE_PROBE_HEAD_SHA', 'KEEL_BRACE_IMAGE_PROBE_JAVA_SETUP_OUTCOME',
                'KEEL_BRACE_IMAGE_PROBE_PYTHON_SETUP_OUTCOME')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def error(operation, path, exc):
    message = str(exc)
    return {'operation': operation, 'path': str(path), 'type': type(exc).__name__,
            'message': message[:512], 'messageWasTruncated': len(message) > 512}


def problem(code, **details):
    return {'code': code, **details}


def stat_record(s):
    return {'mode': s.st_mode, 'modeOctal': oct(stat.S_IMODE(s.st_mode)), 'uid': s.st_uid,
            'gid': s.st_gid, 'size': s.st_size, 'mtimeNs': s.st_mtime_ns,
            'device': s.st_dev, 'inode': s.st_ino, 'regular': stat.S_ISREG(s.st_mode)}


class Artifact:
    """Account for every emitted byte, reserving room for diagnostic report/status."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if any(self.root.iterdir()):
            raise ValueError('refusing to mix a new capture with existing output files')
        self.bytes = 0
        self.index = {}

    def room(self, final=False):
        reserve = STATUS_RESERVE if final else REPORT_RESERVE + STATUS_RESERVE
        return max(0, min(FILE_CAP, PHASE_CAP - self.bytes - reserve))

    def write(self, name, data, final=False):
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe artifact path')
        limit = self.room(final)
        kept = data[:limit]
        complete = len(data) <= limit
        actual_name = name if complete else name + '.prefix'
        target = self.root / actual_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(kept)
        self.bytes += len(kept)
        row = {'artifactPath': actual_name, 'bytes': len(kept), 'sha256': sha(kept),
               'complete': complete, 'suppliedBytes': len(data)}
        self.index[actual_name] = row
        return row

    def json(self, name, obj, final=False):
        return self.write(name, json.dumps(obj, sort_keys=True, indent=2,
                                         allow_nan=False).encode() + b'\n', final)

    def status(self, obj):
        data = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
        if len(data) > STATUS_RESERVE or self.bytes + len(data) > PHASE_CAP:
            raise ValueError('status/index exceeded its reserved artifact budget')
        target = self.root / 'capture-status.json'
        with target.open('xb') as stream:
            stream.write(data)
        self.bytes += len(data)


def capture_file(path, label, artifact, allowed_resolved):
    """Only a named allowlisted regular metadata file, no arbitrary symlink target."""
    path = Path(path)
    row = {'requestedPath': str(path), 'resolvedPath': None, 'exists': None,
           'errors': [], 'raw': None, 'bytesComplete': False, 'utcStart': utc(),
           'monotonicStartNs': time.monotonic_ns()}
    try:
        try:
            ls = path.lstat()
            row['exists'] = True
            row['lstatBefore'] = stat_record(ls)
        except OSError as exc:
            row['exists'] = False if exc.errno == errno.ENOENT else None
            row['errors'].append(error('lstat', path, exc))
            return row, None
        try:
            resolved = path.resolve(strict=True)
            row['resolvedPath'] = str(resolved)
        except (OSError, RuntimeError) as exc:
            row['errors'].append(error('resolve', path, exc))
            return row, None
        if str(resolved) not in {str(p) for p in allowed_resolved}:
            row['errors'].append(problem('resolvedPathOutsideMetadataAllowlist', resolvedPath=str(resolved)))
            return row, None
        try:
            before = resolved.stat()
            row['statBefore'] = stat_record(before)
            if not stat.S_ISREG(before.st_mode):
                row['errors'].append(problem('notRegularMetadataFile'))
                return row, None
            limit = artifact.room()
            if limit == 0:
                row['errors'].append(problem('artifactBudgetExhaustedBeforeRead'))
                return row, None
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
            fd = os.open(str(resolved), flags)
            with os.fdopen(fd, 'rb') as stream:
                opened = os.fstat(stream.fileno())
                row['openedStat'] = stat_record(opened)
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    row['errors'].append(problem('metadataChangedBeforeOpen'))
                    return row, None
                data = stream.read(limit + 1)  # one bounded sentinel byte detects a prefix
                after = os.fstat(stream.fileno())
                row['statAfter'] = stat_record(after)
            complete = len(data) <= limit and before.st_size <= limit
            kept = data[:limit]
            row['bytesComplete'] = complete
            row['raw'] = artifact.write('raw/' + label + '.bin', kept)
            row['raw']['sourceBytesComplete'] = complete
            row['rawSHA256Scope'] = 'whole-observed-file' if complete else 'captured-prefix-only'
            if not complete:
                row['errors'].append(problem('metadataFileExceedsCaptureLimit', limitBytes=limit,
                                             observedStatSize=before.st_size, prefixBytes=len(kept)))
            if (opened.st_size, opened.st_mtime_ns, opened.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                row['errors'].append(problem('metadataChangedDuringRead'))
            return row, kept if complete else None
        except (OSError, ValueError) as exc:
            row['errors'].append(error('readOrRetainMetadata', path, exc))
            return row, None
    finally:
        row['utcEnd'] = utc()
        row['durationNs'] = time.monotonic_ns() - row['monotonicStartNs']


def discover_runner_roots(proc_root=Path('/proc'), deadline=None):
    """Read comm; only matching names permit reading exe. Never cmdline/environ/config."""
    result = {'procRoot': str(proc_root), 'scannedEntries': 0, 'nonmatchingCount': 0,
              'matches': [], 'roots': [], 'errors': [], 'truncated': False}
    try:
        entries = sorted((p for p in Path(proc_root).iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
    except OSError as exc:
        result['errors'].append(error('listProcPids', proc_root, exc))
        return result
    for entry in entries:
        if result['scannedEntries'] >= PROC_LIMIT or (deadline is not None and time.monotonic() >= deadline):
            result['truncated'] = True
            result['errors'].append(problem('processDiscoveryLimitOrDeadline', unscannedAtLeast=1))
            break
        result['scannedEntries'] += 1
        comm_path = entry / 'comm'
        try:
            fd = os.open(str(comm_path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
            with os.fdopen(fd, 'rb') as stream:
                comm_stat = os.fstat(stream.fileno())
                if not stat.S_ISREG(comm_stat.st_mode):
                    raise ValueError('comm is not regular proc metadata')
                raw_comm = stream.read(PROC_COMM_CAP + 1)
            if len(raw_comm) > PROC_COMM_CAP:
                raise ValueError('comm exceeds128-byte discovery cap')
            comm = raw_comm.decode('ascii').removesuffix('\n')
        except (OSError, ValueError, UnicodeError) as exc:
            result['errors'].append(error('readProcComm', comm_path, exc))
            continue
        if comm not in RUNNER_NAMES:
            result['nonmatchingCount'] += 1
            continue
        row = {'pid': int(entry.name), 'comm': comm, 'commRawASCII': raw_comm.decode('ascii'),
               'commSHA256': sha(raw_comm), 'commPath': str(comm_path),
               'commStat': stat_record(comm_stat), 'exePath': None, 'runnerRoot': None,
               'setupInfoPath': None, 'errors': []}
        result['matches'].append(row)
        try:
            exe_link = entry / 'exe'
            row['exeSymlinkStat'] = stat_record(exe_link.lstat())
            target_before = os.readlink(exe_link)
            row['exeReadlink'] = target_before
            row['exeReadlinkSHA256'] = sha(os.fsencode(target_before))
            executable = exe_link.resolve(strict=True)
            row['exePath'] = str(executable)
            if os.readlink(exe_link) != target_before:
                raise ValueError('runner executable link changed during discovery; no root inferred')
            # Observed runner apphost must identify the actual bin installation; no home-path guessing.
            if executable.name != comm or executable.parent.name != 'bin':
                raise ValueError('matching comm does not resolve to named runner executable in bin; no root inferred')
            root = executable.parent.parent
            row['runnerRoot'] = str(root)
            row['setupInfoPath'] = str(root / '.setup_info')
            if str(root) not in result['roots']:
                result['roots'].append(str(root))
        except (OSError, ValueError, RuntimeError) as exc:
            row['errors'].append(error('resolveObservedRunnerExecutable', entry / 'exe', exc))
        if len(result['matches']) >= RUNNER_MATCH_LIMIT:
            result['truncated'] = True
            result['errors'].append(problem('runnerMatchLimitReached', limit=RUNNER_MATCH_LIMIT))
            break
    if not result['roots']:
        result['errors'].append(problem('runnerRootNotDiscovered', setupInfoPath=None))
    return result


def unique_json_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key (raw bytes retained)')
        obj[key] = value
    return obj


def parse_image_detail(detail):
    expected = {'Image': 'image', 'Version': 'version', 'Included Software': 'includedSoftwareURL',
                'Image Release': 'imageReleaseURL'}
    result = {'detail': detail, 'fieldOccurrences': {v: [] for v in expected.values()},
              'unrecognizedLines': [], 'claim': None, 'errors': []}
    if len(detail) > DETAIL_CHAR_CAP:
        result['detail'] = None
        result['detailPrefix'] = detail[:DETAIL_CHAR_CAP]
        result['detailCharacters'] = len(detail)
        result['errors'].append(problem('runnerImageDetailExceedsParsedLimit', fullDetail='retained in original raw JSON'))
        return result
    for line in detail.replace('\r\n', '\n').split('\n'):
        if not line:
            continue
        key, separator, value = line.partition(':')
        if key not in expected:
            result['unrecognizedLines'].append(line)
            continue
        name = expected[key]
        occurrence = {'rawLine': line, 'value': value[1:] if value.startswith(' ') else value}
        result['fieldOccurrences'][name].append(occurrence)
        if not separator or not value.startswith(' ') or not occurrence['value']:
            result['errors'].append(problem('malformedRunnerImageDetailField', field=name))
    for name, occurrences in result['fieldOccurrences'].items():
        if len(occurrences) != 1:
            result['errors'].append(problem('missingOrAmbiguousRunnerImageDetailField', field=name, count=len(occurrences)))
    if not result['errors']:
        result['claim'] = {name: values[0]['value'] for name, values in result['fieldOccurrences'].items()}
    return result


def parse_image_json(data):
    result = {'format': 'array-of-group-detail-records', 'groupCount': None,
              'runnerImageCount': None, 'runnerImageRecords': [], 'claim': None, 'errors': [],
              'encoding': 'UTF-8-BOM' if data.startswith(b'\xef\xbb\xbf') else 'UTF-8'}
    try:
        obj = json.loads(data.decode('utf-8-sig'), object_pairs_hook=unique_json_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
    except (UnicodeError, ValueError) as exc:
        result['errors'].append(error('parseImageJSON', 'retained raw metadata', exc))
        return result
    if not isinstance(obj, list):
        result['errors'].append(problem('imageJSONMustBeGroupDetailArray'))
        return result
    result['groupCount'] = len(obj)
    if len(obj) > JSON_GROUP_LIMIT:
        result['errors'].append(problem('imageJSONGroupLimitExceeded', limit=JSON_GROUP_LIMIT))
        return result
    for index, entry in enumerate(obj):
        if not isinstance(entry, dict):
            result['errors'].append(problem('malformedImageGroupRecord', index=index))
            continue
        fields = {}
        for key in ('group', 'detail'):
            variants = [k for k in entry if k.lower() == key]
            if len(variants) != 1 or not isinstance(entry.get(variants[0]) if variants else None, str):
                result['errors'].append(problem('missingOrAmbiguousGroupDetailKey', index=index, key=key))
            else:
                fields[key] = entry[variants[0]]
        if len(fields) != 2:
            continue
        if fields['group'] == 'Runner Image':
            parsed = parse_image_detail(fields['detail'])
            result['runnerImageRecords'].append({'index': index, 'group': fields['group'], **parsed})
            result['errors'].extend(parsed['errors'])
    result['runnerImageCount'] = len(result['runnerImageRecords'])
    if result['runnerImageCount'] != 1:
        result['errors'].append(problem('runnerImageGroupMustBeUnique', count=result['runnerImageCount']))
    if not result['errors']:
        result['claim'] = result['runnerImageRecords'][0]['claim']
    return result


def parse_environment_file(data):
    result = {'matchingLines': [], 'claims': {}, 'errors': []}
    try:
        text = data.decode('utf-8')
    except UnicodeError as exc:
        result['errors'].append(error('decodeEtcEnvironment', '/etc/environment', exc))
        return result
    for number, line in enumerate(text.splitlines(), 1):
        found = re.match(r'^[ \t]*(?:export[ \t]+)?(ImageOS|ImageVersion)(?=[ \t=]|$)', line)
        if not found:
            continue
        name = found.group(1)
        row = {'line': number, 'rawLine': line, 'name': name, 'value': None, 'errors': []}
        result['matchingLines'].append(row)
        assignment = re.fullmatch(r'[ \t]*(ImageOS|ImageVersion)=([^\r\n]*)', line)
        if not assignment:
            row['errors'].append(problem('unsupportedOrMalformedImageAssignment'))
        else:
            value = assignment.group(2)
            if value.startswith(('"', "'")):
                quote = value[0]
                if len(value) < 2 or value[-1] != quote or quote in value[1:-1] or any(c in value[1:-1] for c in ('\\', '$', '`')):
                    row['errors'].append(problem('ambiguousOrExpandableQuotedImageValue'))
                else:
                    row['value'] = value[1:-1]
            elif any(c.isspace() or c in '$`\\\"\'' for c in value):
                row['errors'].append(problem('ambiguousOrExpandableUnquotedImageValue'))
            else:
                row['value'] = value  # literal only; empty stays empty; never source/eval/expand.
        result['errors'].extend(row['errors'])
    for name in ('ImageOS', 'ImageVersion'):
        rows = [r for r in result['matchingLines'] if r['name'] == name]
        if len(rows) == 1 and not rows[0]['errors']:
            result['claims'][name] = {'present': True, 'value': rows[0]['value']}
        else:
            result['claims'][name] = {'present': bool(rows), 'value': None}
            result['errors'].append(problem('missingOrAmbiguousImageAssignment', name=name, count=len(rows)))
    return result


def process_environment_claims(environ):
    return {key: {'present': key in environ, 'value': environ.get(key)} for key in ('ImageOS', 'ImageVersion')}


def collect(output_dir, phase, environ=None):
    if phase not in ('before-tools', 'after-tools'):
        raise ValueError('unknown probe phase')
    if Path(output_dir).name != phase:
        raise ValueError('output directory must identify the exact phase')
    environ = os.environ if environ is None else environ
    start_utc, start_ns = utc(), time.monotonic_ns()
    deadline = time.monotonic() + COLLECTION_SECONDS
    artifact = Artifact(output_dir)
    report = {'schemaVersion': 1, 'purpose': 'zero-game-image-identity-diagnostic',
              'notImmutableAttestation': True, 'noRuntimeAcceptance': True, 'phase': phase,
              'competitiveGames': 0, 'utcStart': start_utc, 'monotonicStartNs': start_ns,
              'context': {k: environ.get(k) for k in CONTEXT_KEYS},
              'processImageEnvironment': process_environment_claims(environ),
              'collectorPython': {'version': sys.version, 'executable': sys.executable},
              'platform': {'architecture': platform.machine(), 'kernel': platform.release(),
                           'kernelVersion': platform.version(), 'system': platform.system()},
              'collectorPID': os.getpid(), 'collectorUID': os.getuid(), 'collectorGID': os.getgid(),
              'limits': {'eachFileBytes': FILE_CAP, 'wholeTwoPhaseArtifactBytes': ARTIFACT_CAP, 'phasePayloadBytes': PHASE_CAP,
                         'rawPhaseBytesUpperBound': RAW_PHASE_CAP, 'ZIPOverheadReserveBytes': ZIP_OVERHEAD_RESERVE,
                         'procEntries': PROC_LIMIT, 'runnerMatches': RUNNER_MATCH_LIMIT,
                         'collectionSeconds': COLLECTION_SECONDS},
              'captures': {}, 'errors': []}
    try:
        with Path(__file__).open('rb') as stream:
            code = stream.read(FILE_CAP + 1)
        report['collectorSource'] = artifact.write('collector.py', code)
        if not report['collectorSource']['complete']:
            report['errors'].append(problem('collectorSourceExceededCaptureLimit'))
    except OSError as exc:
        report['errors'].append(error('retainExecutedCollectorCode', __file__, exc))
    discovery = discover_runner_roots(deadline=deadline)
    report['runnerDiscovery'] = discovery
    paths = dict(FIXED_FILES)
    for index, root in enumerate(discovery['roots']):
        paths['setup_info_' + str(index)] = str(Path(root) / '.setup_info')
    allowed = set(FIXED_FILES.values()) | {p for k, p in paths.items() if k.startswith('setup_info_')}
    for label, path in paths.items():
        if time.monotonic() >= deadline:
            report['captures'][label] = {'requestedPath': path, 'errors': [problem('collectionDeadlineBeforeRead')]}
            continue
        observation, data = capture_file(path, label, artifact, allowed)
        report['captures'][label] = observation
        if data is not None and (label == 'imagedata' or label.startswith('setup_info_')):
            parsed = parse_image_json(data)
        elif data is not None and label == 'etc_environment':
            parsed = parse_environment_file(data)
        else:
            parsed = None
        if label == 'boot_id' and data is not None:
            try:
                observation['literalASCII'] = data.decode('ascii')
            except UnicodeError as exc:
                observation['errors'].append(error('decodeBootIdASCII', path, exc))
        if parsed is not None:
            retained = artifact.json('parsed/' + label + '.json', parsed)
            observation['parsedArtifact'] = retained
            observation['parseErrors'] = parsed['errors']
            observation['parsedClaims'] = parsed.get('claim') if 'claim' in parsed else parsed.get('claims')
            if not retained['complete']:
                observation['errors'].append(problem('parsedMetadataExceededArtifactBudget'))
    report['utcEnd'] = utc()
    report['durationNs'] = time.monotonic_ns() - start_ns
    report['rawArtifactIndex'] = dict(artifact.index)
    errors = len(report['errors']) + len(discovery['errors']) + sum(len(p['errors']) for p in discovery['matches'])
    errors += sum(len(v.get('errors', [])) + len(v.get('parseErrors', [])) for v in report['captures'].values())
    retained_report = artifact.json('report.json', report, final=True)
    artifact.status({'schemaVersion': 1, 'purpose': report['purpose'], 'noRuntimeAcceptance': True,
                     'notImmutableAttestation': True, 'competitiveGames': 0,
                     'diagnosticErrorCount': errors, 'report': retained_report,
                     'retainedBytesBeforeStatus': artifact.bytes, 'maximumWholeArtifactBytes': ARTIFACT_CAP, 'maximumPhasePayloadBytes': PHASE_CAP,
                     'phase': phase,
                     'files': artifact.index,
                     'note': 'Separate sources are observations only. No version mapping, fallback, allowlist decision or runtime approval.'})
    return 0 if errors == 0 and retained_report['complete'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--phase', choices=('before-tools', 'after-tools'), required=True)
    args = parser.parse_args()
    try:
        return collect(args.output_dir, args.phase)
    except Exception as exc:
        # The workflow uploads any partial originals even when collection/retention itself fails.
        print(json.dumps({'purpose': 'zero-game-image-identity-diagnostic', 'noRuntimeAcceptance': True,
                          'phase': args.phase, 'fatalCollectorError': type(exc).__name__, 'message': str(exc)[:512]}), file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
