"""One-download-only raw artifact custody. No battle/private-log interpretation."""
from pathlib import Path
import datetime, hashlib, json, subprocess, zipfile

OUT=Path('/private/tmp/keel-brace-screen-41951522797-final-evidence')
OUT.mkdir(exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
receipts=[]
def fetch(endpoint,name):
    command=['gh','api',endpoint]
    target=OUT/name;partial=Path(str(target)+'.download-part');error=Path(str(target)+'.stderr')
    start=now();result=None
    with partial.open('xb') as stream,error.open('xb') as stderr:
        result=subprocess.run(command,stdout=stream,stderr=stderr,timeout=600)
    record=dict(command=command,startedAtUTC=start,completedAtUTC=now(),exitCode=result.returncode,
                downloadedBytes=partial.stat().st_size,output=name,stderr=error.name)
    receipts.append(record)
    (OUT/'download-commands-and-times.json').write_text(json.dumps(receipts,indent=2)+'\n')
    result.check_returncode();partial.rename(target)
    return target

for suffix,name in [('actions/runs/41951522797','run.json'),('actions/runs/41951522797/jobs','jobs.json'),
                    ('actions/runs/41951522797/artifacts','artifacts.json'),
                    ('actions/artifacts/13383827989','artifact-13383827989.json'),
                    ('actions/artifacts/13383922872','artifact-13383922872.json')]:
    fetch('repos/stephanelee/robocode/'+suffix,name)

specifications=[(13383827989,108899016,'3eb14b072617c3b64d364988c9db8d224cb47e4160214bd14d6e6ef629ca02d8'),
                (13383922872,1366,'ffcee88194c9599be4930058853481c85ee0ee91dd5d96d02f396b6343ef89db')]
for identifier,size,digest in specifications:
    api=json.loads((OUT/f'artifact-{identifier}.json').read_text())
    assert api['id']==identifier and api['size_in_bytes']==size and api['digest']=='sha256:'+digest
    path=fetch(f'repos/stephanelee/robocode/actions/artifacts/{identifier}/zip',f'original-artifact-{identifier}.zip')
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    assert path.stat().st_size==size and actual==digest
    print(json.dumps(dict(downloadedVerifiedZIP=str(path),bytes=size,sha256=actual,memberVerification='starting')),flush=True)
    rows=[]
    with zipfile.ZipFile(path) as zipped:
        for item in zipped.infolist():
            if item.is_dir():continue
            h=hashlib.sha256();read=0
            with zipped.open(item) as stream:
                while block:=stream.read(1024*1024):h.update(block);read+=len(block)
            assert read==item.file_size
            rows.append(dict(path=item.filename,bytes=read,sha256=h.hexdigest(),crc32=f'{item.CRC:08x}',compressedBytes=item.compress_size))
    manifest=dict(artifactID=identifier,originalZIP=path.name,bytes=size,sha256=actual,members=rows,
                  regularMembers=len(rows),uncompressedBytes=sum(r['bytes'] for r in rows),
                  verification='Every member completely streamed; ZipExtFile CRC check passed; no semantic private-log interpretation',completedAtUTC=now())
    (OUT/f'zip-members-{identifier}.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(wholeZIPAndAllMembersVerified=str(path),regularMembers=len(rows),uncompressedBytes=manifest['uncompressedBytes'])),flush=True)
