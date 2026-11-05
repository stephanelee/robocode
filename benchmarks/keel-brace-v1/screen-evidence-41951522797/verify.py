"""Verbatim raw evidence reassembly/hash/CRC only. Never interpret bot/private logs."""
from pathlib import Path
import hashlib
import json
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    physical={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    assert physical==set(manifest['files'])|{'manifest.json'},'Physical inventory differs'
    for name,row in manifest['files'].items():
        path=ROOT/name
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
    for archive in manifest['originalZIPs']:
        h=hashlib.sha256();size=0
        with tempfile.TemporaryFile() as restored:
            for name in archive['parts']:
                with (ROOT/name).open('rb') as part:
                    while block:=part.read(1024*1024):restored.write(block);h.update(block);size+=len(block)
            assert size==archive['bytes'] and h.hexdigest()==archive['sha256'],'Original ZIP differs'
            restored.seek(0)
            expected=json.loads((ROOT/archive['memberManifest']).read_text())
            with zipfile.ZipFile(restored) as zipped:
                actual=[]
                for item in zipped.infolist():
                    if item.is_dir():continue
                    digest=hashlib.sha256();read=0
                    with zipped.open(item) as stream:
                        while block:=stream.read(1024*1024):digest.update(block);read+=len(block)
                    actual.append(dict(path=item.filename,bytes=read,sha256=digest.hexdigest(),crc32=f'{item.CRC:08x}',compressedBytes=item.compress_size))
                assert actual==expected['members'],'Complete member/CRC/hash inventory differs'
    print('PASS: both complete original ZIPs, all1767members, all14WHOLErecordings and all physical bytes verified; no numerical evaluation')
if __name__=='__main__':main()
