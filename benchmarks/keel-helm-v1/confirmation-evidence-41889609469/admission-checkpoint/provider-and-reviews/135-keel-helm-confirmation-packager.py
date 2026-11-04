#!/usr/bin/env python3
"""Private lossless evidence packaging only. No network, repo edits or battle execution.

This tool is outcome-agnostic: a failed/partial stage is preserved without assuming
18 recordings or any completed round. Existing owner downloads are the inputs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import traceback
import zipfile

LIMIT = 2 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name


def save(root, relative, raw):
    if not safe_relative(relative):
        raise ValueError('Unsafe output path: ' + relative)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Never overwrite prior physical evidence: ' + relative)
    path.write_bytes(raw)
    return relative


def save_parts(root, relative, raw):
    return [save(root, relative + '/part-%03d.bin' % (start // LIMIT + 1), raw[start:start + LIMIT])
            for start in range(0, len(raw), LIMIT)] or [save(root, relative + '/part-001.bin', b'')]


def assert_private_output(path):
    path = path.resolve()
    if not path.is_relative_to(Path('/private/tmp').resolve()):
        raise ValueError('Output must be inside /private/tmp only')
    if any((ancestor / '.git').exists() for ancestor in (path, *path.parents)):
        raise ValueError('Output must not be in any Git worktree/repository')
    if path.exists():
        raise FileExistsError('Output must be a fresh absent private directory')
    return path


def detect_prefix(names, stage):
    needle = 'build/keel-helm/' + stage + '/'
    choices = {name[:name.index(needle) + len(needle)] for name in names if needle in name}
    if len(choices) > 1:
        raise ValueError('Ambiguous stage prefixes; supply --stage-prefix explicitly')
    return next(iter(choices)) if choices else None


def extra_inputs(args):
    result = []
    if args.extras_manifest:
        index = json.loads(args.extras_manifest.read_bytes())
        values = index.get('files', index)
        if not isinstance(values, dict):
            raise ValueError('Extras manifest must map original paths to bytes/SHA256')
        for name, value in values.items():
            p = Path(name)
            raw = p.read_bytes()
            if len(raw) != value['bytes'] or sha(raw) != value['sha256']:
                raise ValueError('Owner-stable extra differs: ' + name)
            result.append((p, raw, True))
        result.append((args.extras_manifest, args.extras_manifest.read_bytes(), False))
    result += [(path, path.read_bytes(), False) for path in args.extra]
    return result


def build(args):
    output = assert_private_output(args.output)
    source_zip = args.zip.resolve(strict=True)
    raw_zip = source_zip.read_bytes()
    if len(raw_zip) != args.expected_zip_bytes or sha(raw_zip) != args.expected_zip_sha256:
        raise ValueError('Existing ORIGINAL provider ZIP differs from owner-verified identity')
    # Verify all declared stable extras before creating the archival checkpoint.
    extras = extra_inputs(args)
    output.mkdir(parents=True)
    try:
        parts = save_parts(output, 'original-provider.zip.parts', raw_zip)
        manifest = dict(schemaVersion=1, kind=args.kind, stage=args.stage, runId=args.run_id,
            runAttempt=1, expectedPayloadHead=args.expected_head,
            scope='Lossless original evidence only; no eligibility, validity, round-budget or strength conclusion is calculated.',
            originalZIP=dict(bytes=len(raw_zip), sha256=sha(raw_zip), parts=parts),
            originalMembers=[], wholeRecordings=[], originalDecisions=[], originalFailures=[],
            declaredRegisteredMaximumGames=18, actualRecordingCount=None, archiveWarnings=[], extras=[])
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zipped:
            members = [item for item in zipped.infolist() if not item.is_dir()]
            names = [item.filename for item in members]
            prefix = args.stage_prefix or detect_prefix(names, args.stage)
            manifest['stagePrefix'] = prefix
            seen = set()
            for index, item in enumerate(members):
                name = item.filename
                data = zipped.read(item)
                relative = name[len(prefix):] if prefix and name.startswith(prefix) else None
                member = dict(index=index, filename=name, bytes=len(data), sha256=sha(data),
                    zipCRC32=item.CRC, zipExternalAttributes=item.external_attr,
                    storage=dict(kind='original-provider-zip-member', index=index, filename=name))
                recording = name.endswith('.battle.gz')
                direct = recording or (relative is not None and not relative.startswith(('work/', 'provenance/')))
                # Provenance artifacts use a flat root: retain non-source metadata directly.
                if args.kind == 'provenance' and not name.startswith('source/'):
                    direct = True
                if prefix is None and not name.startswith('source/') and args.kind == 'evidence':
                    direct = True  # A build/preparation failure may have no stage output at all.
                if name in seen:
                    manifest['archiveWarnings'].append('Duplicate original ZIP filename: ' + name)
                seen.add(name)
                if direct and safe_relative(name) and not stat.S_ISLNK(item.external_attr >> 16):
                    dest = ('stage/' + relative) if relative is not None else ('outside-stage-originals/' + name)
                    if (output / dest).exists():
                        dest = 'duplicate-members/member-%04d.bin' % index
                    member['storage'] = dict(kind='direct-file', path=save(output, dest, data))
                elif not safe_relative(name):
                    manifest['archiveWarnings'].append('Unsafe original ZIP name retained opaque, not extracted: ' + name)
                if recording:
                    if member['storage']['kind'] != 'direct-file':
                        member['storage'] = dict(kind='direct-file', path=save(output, 'opaque-recordings/member-%04d.battle.gz' % index, data))
                    manifest['wholeRecordings'].append(dict(index=index, filename=name, bytes=len(data),
                        sha256=sha(data), path=member['storage']['path']))
                if name.endswith('/decision.json') or name == 'decision.json':
                    manifest['originalDecisions'].append(dict(index=index, filename=name, bytes=len(data),
                        sha256=sha(data), storage=member['storage']))
                if 'failure' in PurePosixPath(name).name.lower():
                    manifest['originalFailures'].append(dict(index=index, filename=name, bytes=len(data),
                        sha256=sha(data), storage=member['storage']))
                if args.kind == 'provenance' and name.startswith('source/') and safe_relative(name[7:]):
                    source_raw = data
                    binding = dict(commit=args.expected_head, path=name[7:],
                        blobSHA=hashlib.sha1(b'blob '+str(len(source_raw)).encode()+b'\0'+source_raw).hexdigest(),
                        commitPathVerified=False)
                    if getattr(args, 'source_repository', None):
                        committed = subprocess.check_output(['git','show',args.expected_head+':'+name[7:]],
                            cwd=args.source_repository, timeout=15)
                        if committed != source_raw: raise ValueError('Original exported source differs from immutable commit path')
                        binding['commitPathVerified'] = True
                    member['immutableSource'] = binding
                manifest['originalMembers'].append(member)
        manifest['actualRecordingCount'] = len(manifest['wholeRecordings'])
        if len(manifest['wholeRecordings']) != 18 and args.kind == 'evidence':
            manifest['archiveWarnings'].append('Actual recording count differs from full18-game confirmation; preserve partial/failure evidence without replacement or outcome inference.')
        original_by_hash = {(row['bytes'], row['sha256']): row for row in manifest['originalMembers']}
        unique_extras = set()
        for index, (path, data, owner_verified) in enumerate(extras):
            if path.resolve() == source_zip:
                continue  # Already retained as the exact authoritative ZIP parts.
            identity = (str(path), sha(data))
            if identity in unique_extras:
                continue
            unique_extras.add(identity)
            entry = dict(originalPath=str(path), bytes=len(data), sha256=sha(data), ownerStableManifestChecked=owner_verified)
            known = original_by_hash.get((len(data), sha(data)))
            if known is not None:
                entry['storage'] = dict(kind='original-provider-zip-member', index=known['index'], filename=known['filename'])
            elif path.suffix.lower() == '.zip':
                entry['storage'] = dict(kind='verbatim-parts', parts=save_parts(output, 'additional-original-zips/%03d-%s.parts' % (index, path.name), data))
            else:
                entry['storage'] = dict(kind='direct-file', path=save(output, 'provider-and-reviews/%03d-%s' % (index, path.name), data))
            manifest['extras'].append(entry)
        save(output, 'verify.py', VERIFIER.encode())
        save(output, 'README.md', README.encode())
        manifest['files'] = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
            for p in sorted(output.rglob('*')) if p.is_file()}
        save(output, 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
        return dict(output=str(output), manifestSHA256=sha((output / 'manifest.json').read_bytes()),
            logicalOriginalFiles=len(manifest['originalMembers']), wholeRecordings=len(manifest['wholeRecordings']),
            decisionFiles=len(manifest['originalDecisions']), failureArtifacts=len(manifest['originalFailures']),
            physicalFiles=len(manifest['files'])+1, physicalBytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            warnings=manifest['archiveWarnings'])
    except Exception:
        # Keep partial archival writes and failure evidence; never erase or replace them.
        failure = output / 'packaging-failure.json'
        if not failure.exists():
            failure.write_text(json.dumps(dict(error=traceback.format_exc(), noSourceOrGameExecution=True),indent=2)+'\n')
        raise


VERIFIER = r'''"""Portable original-byte verifier only: no network/Git/harness/bots/reducer."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
ROOT = Path(__file__).resolve().parent

def sha(data): return hashlib.sha256(data).hexdigest()
def safe(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name: raise ValueError('Unsafe physical reference')
    return p

def restore(storage):
    if storage['kind']=='direct-file': return (ROOT/safe(storage['path'])).read_bytes()
    if storage['kind']=='verbatim-parts':
        parts=[(ROOT/safe(name)).read_bytes() for name in storage['parts']]
        if any(len(b)>2*1024*1024 for b in parts): raise ValueError('Original ZIP part exceeds2MiB')
        return b''.join(parts)
    raise ValueError('Storage is an opaque original ZIP member; read the exact indexed member')

def main():
    m=json.loads((ROOT/'manifest.json').read_bytes())
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    if actual!=set(m['files'])|{'manifest.json'}: raise ValueError('Physical inventory differs')
    for name,v in m['files'].items():
        raw=(ROOT/safe(name)).read_bytes()
        if len(raw)!=v['bytes'] or sha(raw)!=v['sha256']: raise ValueError('Physical bytes differ:'+name)
    raw=restore(dict(kind='verbatim-parts',parts=m['originalZIP']['parts']))
    if len(raw)!=m['originalZIP']['bytes'] or sha(raw)!=m['originalZIP']['sha256']: raise ValueError('Original ZIP differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members=[i for i in z.infolist() if not i.is_dir()]
        if len(members)!=len(m['originalMembers']): raise ValueError('Logical original count differs')
        for index,(item,row) in enumerate(zip(members,m['originalMembers'])):
            original=z.read(item)
            if row['index']!=index or row['filename']!=item.filename or len(original)!=row['bytes'] or sha(original)!=row['sha256']: raise ValueError('Logical original bytes differ')
            if item.CRC!=row['zipCRC32'] or item.external_attr!=row['zipExternalAttributes']: raise ValueError('Original ZIP member metadata differs')
            if 'immutableSource' in row:
                binding=row['immutableSource']
                blob=b'blob '+str(len(original)).encode()+b'\0'+original
                if binding['commit']!=m['expectedPayloadHead'] or item.filename!='source/'+binding['path'] or hashlib.sha1(blob).hexdigest()!=binding['blobSHA']: raise ValueError('Immutable original source identity differs')
            storage=row['storage']
            if storage['kind']=='direct-file':
                if restore(storage)!=original: raise ValueError('Direct original differs')
            elif storage!={'kind':'original-provider-zip-member','index':index,'filename':item.filename}: raise ValueError('Opaque original index differs')
    for row in m['wholeRecordings']:
        p=ROOT/safe(row['path'])
        if not p.is_file() or len(p.read_bytes())!=row['bytes'] or sha(p.read_bytes())!=row['sha256']: raise ValueError('Whole recording differs')
    if len(m['wholeRecordings'])!=m['actualRecordingCount']: raise ValueError('Actual recording inventory differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries=[i for i in z.infolist() if not i.is_dir()]
        for row in m['extras']:
            storage=row['storage']
            if storage['kind']=='original-provider-zip-member':
                item=entries[storage['index']]
                if item.filename!=storage['filename']: raise ValueError('Original extra ZIP reference differs')
                b=z.read(item)
            else: b=restore(storage)
            if len(b)!=row['bytes'] or sha(b)!=row['sha256']: raise ValueError('Original extra differs')
    print('PASS: exact original ZIP/all logical files/actual whole recordings/extras verified; no outcome or budget calculation')

if __name__=='__main__': main()
'''

README = '''# Lossless confirmation evidence checkpoint

This is an outcome-agnostic original-byte archive, not a competitive result or
confirmation eligibility/acceptance claim. It uses existing owner-downloaded
originals only. No network, Git mutation, runtime preparation, bot/helper,
metric reducer or game is run by packaging or verification.

The exact ORIGINAL provider ZIP is retained in verbatim parts at most2MiB;
it is never recompressed. Every original logical member, including staged work
and any failure/partial metadata, remains inside that authoritative ZIP with
its indexed filename, byte length, SHA256 and original ZIP metadata. Key stage
metadata and ALL actual original .battle.gz files are also direct regular files.
No expected18-recording inventory is fabricated when a stage fails or stops.
The unchanged raw decisions, failures, budget and unplayed-ID artifacts are
retained without normalization; interpretation belongs to the independent audit.

Provider API/log/review/receipt/comment and independent audit extras retain
original bytes and hashes. Any additional original ZIP is split verbatim too.
A byte/hash-only portable check needs no Git, network or mutable source tree:

    python3 -I -B -S verify.py

Do not infer validity, consumed physical rounds, performance or stage acceptance
from this archive's successful byte verification. Preserve failed attempts and
obtain independent outcome/source/runtime review before any new authorized stage.
'''


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--expected-zip-sha256',required=True)
    p.add_argument('--expected-zip-bytes',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=('provenance','evidence'),required=True)
    p.add_argument('--stage',choices=('confirmation',),default='confirmation')
    p.add_argument('--stage-prefix')
    p.add_argument('--run-id',type=int,required=True)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--extras-manifest',type=Path)
    p.add_argument('--source-repository',type=Path,help='Optional read-only Git source-commit verification; never edits checkout')
    p.add_argument('--extra',type=Path,action='append',default=[])
    return p


if __name__=='__main__':
    print(json.dumps(build(parser().parse_args()),indent=2))
