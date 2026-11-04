"""Portable original-byte verifier, adapted from checkpoint500bc verify.py.
No network, Git, bot, SDK, harness, reducer, runtime preparation or game execution.
Whole recordings are indexed original ZIP members, not duplicate regular files.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
ROOT=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
def safe(root,name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe physical reference')
    result=root.joinpath(*p.parts)
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):raise ValueError('Physical reference escapes archive')
    return result

def verify(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_bytes())
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(m['files'])|{'manifest.json'}:raise ValueError('Physical inventory differs')
    for name,row in m['files'].items():
        raw=safe(root,name).read_bytes()
        if len(raw)!=row['bytes'] or sha(raw)!=row['sha256']:raise ValueError('Physical bytes differ:'+name)
    parts=m['originalZIP']['parts'];inventory=m['originalZIP']['partInventory']
    if len(parts)!=len(inventory):raise ValueError('Part inventory differs')
    chunks=[]
    for i,(name,row) in enumerate(zip(parts,inventory)):
        b=safe(root,name).read_bytes();blob=b'blob '+str(len(b)).encode()+b'\0'+b
        if not 0<len(b)<=2*1024*1024 or row!=dict(index=i,path=name,bytes=len(b),sha256=sha(b),gitBlobSHA=hashlib.sha1(blob).hexdigest()):raise ValueError('Part index/hash differs')
        chunks.append(b)
    raw=b''.join(chunks);del chunks
    if len(raw)!=m['originalZIP']['bytes'] or sha(raw)!=m['originalZIP']['sha256']:raise ValueError('Original ZIP differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members=[x for x in z.infolist() if not x.is_dir()]
        if len(members)!=len(m['originalMembers']):raise ValueError('Original member count differs')
        for i,(item,row) in enumerate(zip(members,m['originalMembers'])):
            b=z.read(item)
            storage=dict(kind='original-provider-zip-member',index=i,filename=item.filename)
            if row!=dict(index=i,filename=item.filename,bytes=len(b),sha256=sha(b),zipCRC32=item.CRC,zipExternalAttributes=item.external_attr,storage=storage):raise ValueError('Original indexed member differs:'+str(i))
        recordings=[i for i,x in enumerate(members) if x.filename.endswith('.battle.gz')]
        if recordings!=[r['index'] for r in m['wholeRecordings']] or len(recordings)!=m['actualRecordingCount']:raise ValueError('Whole recording inventory differs')
        for row in m['wholeRecordings']:
            i=row['index'];item=members[i];b=z.read(item)
            if row!=dict(index=i,filename=item.filename,bytes=len(b),sha256=sha(b),storage=dict(kind='original-provider-zip-member',index=i,filename=item.filename)):raise ValueError('Whole recording differs')
        for group in ('originalDecisions','originalFailures','extras'):
            for row in m.get(group,[]):
                storage=row['storage']
                if storage['kind']=='original-provider-zip-member':
                    i=storage['index'];item=members[i]
                    if storage!=dict(kind='original-provider-zip-member',index=i,filename=item.filename):raise ValueError('Extra index differs')
                    b=z.read(item)
                elif storage['kind']=='direct-file':b=safe(root,storage['path']).read_bytes()
                else:raise ValueError('Unknown extra storage')
                if len(b)!=row['bytes'] or sha(b)!=row['sha256']:raise ValueError('Extra original differs')
    if 'frozenSourceBindings' in m:
        bindings=json.loads(safe(root,m['frozenSourceBindings']).read_bytes())
        if len(bindings)!=90:raise ValueError('Frozen source count differs')
        for name,row in bindings.items():
            b=safe(root,row['path']).read_bytes();blob=b'blob '+str(len(b)).encode()+b'\0'+b
            if row['commit']!=m['expectedPayloadHead'] or row['bytes']!=len(b) or row['sha256']!=sha(b) or row['blobSHA']!=hashlib.sha1(blob).hexdigest():raise ValueError('Frozen source binding differs:'+name)
    if 'checkpointBindings' in m:
        for name,row in json.loads(safe(root,m['checkpointBindings']).read_bytes()).items():
            b=safe(root,row['path']).read_bytes();blob=b'blob '+str(len(b)).encode()+b'\0'+b
            if row['commit']!='500bc1341437afbdb72e5ecce25c0e101858a55e' or row['bytes']!=len(b) or row['sha256']!=sha(b) or row['blobSHA']!=hashlib.sha1(blob).hexdigest():raise ValueError('Admission checkpoint differs:'+name)
    return m

def main():
    m=verify(ROOT)
    if (m['originalZIP']['bytes'],m['originalZIP']['sha256'],len(m['originalMembers']),m['actualRecordingCount'],m['expectedPayloadHead'],m['runId'])!=(351880028,'bcbd2e27ff64dc585f7341aa32ae506ea9819903c1893db8a723ba91c7036ad6',2194,18,'f2a25d3309062a362a13902329de405bf63094a7',41889609469):raise ValueError('Registered provider identity differs')
    print('PASS: exact351880028-byte ZIP,2194 members,18 whole recording byte streams,90 source assets and admission checkpoint; no outcome calculation')
if __name__=='__main__':main()