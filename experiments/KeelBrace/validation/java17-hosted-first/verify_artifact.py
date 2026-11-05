#!/usr/bin/env python3
"""Offline byte/count/custody check only. Never launches Java/bots/games or grants clearance."""
import collections
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

manifest = json.loads((ROOT / 'original-member-manifest.json').read_text())
archive = ROOT / 'original-artifact.zip'
assert archive.stat().st_size == manifest['originalZIPBytes'] == 1463862
assert digest(archive.read_bytes()) == manifest['originalZIPSHA256'] == '71b036d63fe40bf0e83d10f907ca9f09720b3998f6aa3846c78d68d749bebeb6'
with zipfile.ZipFile(archive) as z:
    entries = [e for e in z.infolist() if not e.is_dir()]
    assert len(entries) == len({e.filename for e in entries}) == len(manifest['members']) == 16
    for e, row in zip(entries, manifest['members']):
        assert e.filename == row['path'] and e.file_size == row['bytes'] and e.CRC == row['CRC32']
        data = z.read(e)  # ZIP CRC is checked while reading.
        assert digest(data) == row['sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['gitBlobSHA']
        assert (ROOT / 'original-members' / row['path']).read_bytes() == data

build = ROOT / 'original-members/build/keel-brace-bootstrap'
first, second = (build / 'KeelBrace-first.jar').read_bytes(), (build / 'KeelBrace.jar').read_bytes()
assert first == second
jar_sha = '8a1f1c3365846b7fb272a5a1885b65fad4f5f6ace29cc770fed04f34287d7eb9'
assert digest(second) == jar_sha and len(second) == 872696
assert (ROOT.parents[1] / 'KeelBrace.jar').read_bytes() == second
entry_manifest = json.loads((ROOT / 'jar-entry-manifest.json').read_text())
with zipfile.ZipFile(build / 'KeelBrace.jar') as z:
    entries = [e for e in z.infolist() if not e.is_dir()]
    assert len(entries) == len(entry_manifest['entries']) == 790
    assert len(z.namelist()) == len(set(z.namelist())) == 826
    for e, row in zip(entries, entry_manifest['entries']):
        assert e.filename == row['path'] and e.file_size == row['bytes'] and digest(z.read(e)) == row['sha256']
    assert 'KeelBrace.class' in z.namelist() and 'RamLedger.class' in z.namelist()
    assert 'Keel.class' not in z.namelist() and not any(n.startswith('org/junit/') or n.endswith('Test.class') for n in z.namelist())

passes = collections.Counter(re.findall(r'^([A-Za-z0-9_]+) > ([^\r\n]+?) PASSED$', (build / 'build.log').read_text(), re.M))
assert len(passes) == 30 and set(passes.values()) == {2} and sum(passes.values()) == 60
xml_tests = 0
for f in (ROOT / 'original-members/experiments/KeelBrace/build/test-results/test').glob('TEST-*.xml'):
    suite = ET.parse(f).getroot()
    assert all(int(suite.attrib[k]) == 0 for k in ('errors', 'failures', 'skipped'))
    xml_tests += int(suite.attrib['tests'])
assert xml_tests == 30

proof = json.loads((build / 'provenance.json').read_text())
run = json.loads((ROOT / 'run-api.json').read_text())
artifact = json.loads((ROOT / 'artifact-api.json').read_text())
assert run['id'] == proof['runId'] == artifact['workflow_run']['id'] == 41934557748
assert run['head_sha'] == proof['headSHA'] == artifact['workflow_run']['head_sha'] == 'c5a4c3f97e11358656dcf6821529e9917c757bf5'
assert run['conclusion'] == 'success' and run['run_attempt'] == proof['runAttempt'] == 1
assert artifact['id'] == 13372474505 and artifact['digest'] == 'sha256:' + manifest['originalZIPSHA256']
assert proof['firstCommittedJarPresent'] is False
assert 'Temurin-17.0.21+9' in proof['javaVersion'] and proof['pythonVersion'] == 'Python 3.13.17'
header = re.search(r'Image: ubuntu-26\.04\n[^\n]*Version: ([^\r\n]+)', (ROOT / 'job.log').read_text())
assert header and header.group(1).strip() == '20261025.179.1'
assert proof['imageVersion'] == '20261026.180.1'
assert proof['imageVersion'] != header.group(1).strip()  # Discrepancy is deliberately retained, never waived.
print('PASS: exact original16members; JAR790entries;30unique tests/twoexecutions. Image-provenance discrepancy retained; NOgame-runtime clearance.')
