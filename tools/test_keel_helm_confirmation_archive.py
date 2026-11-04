"""Synthetic byte-preservation regressions only. No games, SDK or outcome reducer."""
import copy,gzip,hashlib,importlib.util,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
B=Path(__file__).resolve().parents[1]/'benchmarks/keel-helm-v1/confirmation-evidence-41889609469'
old=sys.dont_write_bytecode
try:
 sys.dont_write_bytecode=True
 spec=importlib.util.spec_from_file_location('keel_confirmation_byte_archive',B/'verify.py');verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
finally:sys.dont_write_bytecode=old
class ArchiveFixture(unittest.TestCase):
 def create(self,root):
  buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z:
   z.writestr('stage/game.battle.gz',gzip.compress(b'synthetic only\n'))
   z.writestr('stage/decision.json',b'{"synthetic":true}')
  raw=buf.getvalue();parts=[];inventory=[]
  for i,start in enumerate(range(0,len(raw),64)):
   b=raw[start:start+64];h=hashlib.sha256(b).hexdigest();name='objects/'+h+'.bin';p=root/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b);parts.append(name);inventory.append(dict(index=i,path=name,bytes=len(b),sha256=h,gitBlobSHA=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
  rows=[]
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   for i,item in enumerate(z.infolist()):
    b=z.read(item);rows.append(dict(index=i,filename=item.filename,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),zipCRC32=item.CRC,zipExternalAttributes=item.external_attr,storage=dict(kind='original-provider-zip-member',index=i,filename=item.filename)))
  r=rows[0];record={k:r[k] for k in ('index','filename','bytes','sha256','storage')}
  m=dict(originalZIP=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),parts=parts,partInventory=inventory),originalMembers=rows,wholeRecordings=[record],actualRecordingCount=1,originalDecisions=[],originalFailures=[],extras=[])
  m['files']={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.rglob('*') if p.is_file()};self.write(root,m);return m
 def write(self,root,m):(root/'manifest.json').write_bytes(json.dumps(m).encode())
 def exercise(self,mutate=None):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);m=self.create(root)
   if mutate:
    mutate(root,m);self.write(root,m)
    with self.assertRaises((ValueError,FileNotFoundError,zipfile.BadZipFile)):verifier.verify(root)
   else:verifier.verify(root)
 def test_complete_original_roundtrip(self):self.exercise()
 def test_missing_part_rejected(self):self.exercise(lambda r,m:(r/m['originalZIP']['parts'][0]).unlink())
 def test_reordered_parts_rejected(self):self.exercise(lambda r,m:m['originalZIP']['parts'].reverse())
 def test_wrong_member_index_rejected(self):self.exercise(lambda r,m:m['originalMembers'][0]['storage'].update(index=1))
 def test_wrong_member_name_rejected(self):self.exercise(lambda r,m:m['originalMembers'][0].update(filename='different'))
 def test_wrong_member_hash_rejected(self):self.exercise(lambda r,m:m['originalMembers'][0].update(sha256='0'*64))
 def test_missing_whole_recording_rejected(self):self.exercise(lambda r,m:m['wholeRecordings'].clear())
 def test_registered_provider_identity(self):
  m=json.loads((B/'manifest.json').read_bytes());self.assertEqual((351880028,'bcbd2e27ff64dc585f7341aa32ae506ea9819903c1893db8a723ba91c7036ad6'),(m['originalZIP']['bytes'],m['originalZIP']['sha256']));self.assertEqual(2194,len(m['originalMembers']));self.assertEqual(18,m['actualRecordingCount']);self.assertEqual(168,len(m['originalZIP']['parts']))
if __name__=='__main__':unittest.main()