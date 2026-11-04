"""Portable original-byte verifier only: no network/Git/harness/bots/reducer."""
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
