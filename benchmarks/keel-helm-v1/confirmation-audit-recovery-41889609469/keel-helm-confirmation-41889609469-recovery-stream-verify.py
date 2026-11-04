from pathlib import Path
import hashlib,json,zipfile,stat
b=Path('/private/tmp/keel-helm-confirmation-41889609469');zpath=Path(str(b)+'.zip');h=hashlib.sha256()
with zpath.open('rb') as f:
 for x in iter(lambda:f.read(1048576),b''):h.update(x)
assert zpath.stat().st_size==351880028 and h.hexdigest()=='bcbd2e27ff64dc585f7341aa32ae506ea9819903c1893db8a723ba91c7036ad6'
old=json.load(open(str(b)+'-independent-audit.json'));new=json.load(open(str(b)+'-recovery-independent-audit.json'));assert old==new
prev=json.load(open(str(b)+'-archive-stream-audit.json'));entries={};records=[]
with zipfile.ZipFile(zpath) as z:
 for i in z.infolist():
  assert i.filename not in entries and not i.filename.startswith('/') and '..' not in Path(i.filename).parts and not stat.S_ISLNK(i.external_attr>>16)
  if i.is_dir():continue
  m=hashlib.sha256()
  with z.open(i) as f:
   for c in iter(lambda:f.read(1048576),b''):m.update(c)
  value=dict(bytes=i.file_size,sha256=m.hexdigest());assert prev['files'][i.filename]==value;entries[i.filename]=value
  if i.filename.endswith('.battle.gz'):records.append(dict(member=i.filename,**value))
assert len(entries)==2194 and len(records)==18
out=dict(reviewerTask='/root/helm_final_review_recovery',scope='Fresh post-crash original ZIP stream/CRC/hash verification plus rerun preexisting independent raw reducer and provider-binding audit; no games',originalZIP=str(zpath),originalZIPBytes=zpath.stat().st_size,originalZIPSHA256=h.hexdigest(),members=len(entries),wholeRecordings=records,wholeRecordingBytes=sum(x['bytes'] for x in records),newRawReplayEqualsPriorOutput=True,newBindingsEqualsPriorOutput=json.load(open(str(b)+'-bindings-audit.json'))==json.load(open(str(b)+'-recovery-bindings-audit.json')),passAudit=True)
assert out['newBindingsEqualsPriorOutput'];Path(str(b)+'-recovery-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='wholeRecordings'}))
