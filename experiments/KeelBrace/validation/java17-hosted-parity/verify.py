#!/usr/bin/env python3
"""Offline custody/source/JAR and pure-fixture verification; starts no game/process."""
import hashlib,json,re,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
for name,row in json.loads((ROOT/'manifest.json').read_text()).items():
 b=(ROOT/name).read_bytes();assert len(b)==row['bytes'] and h(b)==row['sha256'],name
artifact=json.loads((ROOT/'artifacts.json').read_text())['artifacts'][0]
b=(ROOT/'original.zip').read_bytes();assert len(b)==artifact['size_in_bytes'] and 'sha256:'+h(b)==artifact['digest']
with zipfile.ZipFile(ROOT/'original.zip') as z:
 assert z.testzip() is None
 for row in json.loads((ROOT/'original-members.json').read_text()):
  raw=z.read(row['path']);assert len(raw)==row['bytes'] and h(raw)==row['sha256']
 jars=[n for n in z.namelist() if n.endswith('.jar')];assert len(jars)==3
 assert all(h(z.read(n))=='8a1f1c3365846b7fb272a5a1885b65fad4f5f6ace29cc770fed04f34287d7eb9' for n in jars)
 tests=[ET.fromstring(z.read(n)) for n in z.namelist() if n.endswith('.xml')]
 assert sum(int(x.attrib['tests']) for x in tests)==30
 assert all(int(x.attrib[k])==0 for x in tests for k in ['failures','errors','skipped'])
 for name,count in [('controller',51),('independent-audit',48),('image-probe',21)]:
  log=z.read('build/keel-brace-bootstrap/'+name+'-tests.log').decode();assert re.search(r'Ran '+str(count)+r' tests? in ',log) and '\nOK\n' in log
print('PASS: original ZIP, 3 identical JARs, 30 semantic tests, 51/48/21 pure fixtures; zero games. No image/runtime admission.')
